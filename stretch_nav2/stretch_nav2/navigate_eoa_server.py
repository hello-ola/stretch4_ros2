#!/usr/bin/env python3
"""
Action server on navigate_eoa_to_pose (nav2_msgs/NavigateToPose).

The goal pose is the desired *eoa* pose in the goal header frame (usually map).
The server converts it to a base_link goal and forwards to Nav2 navigate_to_pose.
"""

from __future__ import annotations

import threading

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

from .transform_utils import compute_base_goal_from_eoa_goal

class NavigateEoaServer(Node):
    def __init__(self) -> None:
        super().__init__("navigate_eoa_server")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("eoa_frame", "grasp_center_link")
        self.declare_parameter("eoa_action_name", "navigate_eoa_to_pose")
        self.declare_parameter("nav2_action_name", "navigate_to_pose")
        self.declare_parameter("tf_timeout_sec", 0.5)
        self.declare_parameter("publish_computed_base_goal", True)

        self._base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self._eoa_frame = (
            self.get_parameter("eoa_frame").get_parameter_value().string_value
        )
        eoa_action = (
            self.get_parameter("eoa_action_name").get_parameter_value().string_value
        )
        nav2_action = self.get_parameter("nav2_action_name").get_parameter_value().string_value
        self._tf_timeout = self.get_parameter("tf_timeout_sec").get_parameter_value().double_value
        self._publish_base_goal = (
            self.get_parameter("publish_computed_base_goal").get_parameter_value().bool_value
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        if self._publish_base_goal:
            self._base_goal_pub = self.create_publisher(
                PoseStamped, "computed_base_goal", 10
            )

        self._nav2_client_cb = MutuallyExclusiveCallbackGroup()
        self._nav2_client = ActionClient(
            self,
            NavigateToPose,
            nav2_action,
            callback_group=self._nav2_client_cb,
        )

        self._server_cb = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            eoa_action,
            execute_callback=self._execute,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._server_cb,
        )

        self._nav2_goal_lock = threading.Lock()
        self._nav2_goal_handle = None

        self.get_logger().info(
            f"'{eoa_action}' ready (eoa frame goal -> Nav2 '{nav2_action}', "
            f"base={self._base_frame}, eoa={self._eoa_frame})"
        )

    def _goal_callback(self, goal_request) -> GoalResponse:
        if not goal_request.pose.header.frame_id:
            self.get_logger().warn("Rejected goal: pose.header.frame_id is empty.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        with self._nav2_goal_lock:
            if self._nav2_goal_handle is not None:
                self._nav2_goal_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    def _lookup_transforms(self, frame: str, stamp):
        timeout = rclpy.duration.Duration(seconds=self._tf_timeout)
        t_map_base = self._tf_buffer.lookup_transform(
            frame, self._base_frame, stamp, timeout=timeout
        )
        t_map_eoa = self._tf_buffer.lookup_transform(
            frame, self._eoa_frame, stamp, timeout=timeout
        )
        return t_map_base, t_map_eoa

    def _execute(self, goal_handle):
        eoa_goal = goal_handle.request.pose
        map_frame = eoa_goal.header.frame_id

        try:
            stamp = rclpy.time.Time.from_msg(eoa_goal.header.stamp)
            if eoa_goal.header.stamp.sec == 0 and eoa_goal.header.stamp.nanosec == 0:
                stamp = rclpy.time.Time()
            t_map_base, t_map_eoa = self._lookup_transforms(map_frame, stamp)
        except TransformException as exc:
            self.get_logger().error(f"TF failed: {exc}")
            goal_handle.abort()
            result = NavigateToPose.Result()
            result.error_code = 1
            result.error_msg = f"TF error: {exc}"
            return result

        base_goal = compute_base_goal_from_eoa_goal(
            eoa_goal, t_map_base, t_map_eoa, self._base_frame
        )
        base_goal.header.stamp = eoa_goal.header.stamp

        if self._publish_base_goal:
            self._base_goal_pub.publish(base_goal)

        self.get_logger().info(
            f"Eoa goal in '{map_frame}' -> base ({base_goal.pose.position.x:.3f}, "
            f"{base_goal.pose.position.y:.3f}, yaw from quat)"
        )

        if not self._nav2_client.wait_for_server(timeout_sec=10.0):
            goal_handle.abort()
            result = NavigateToPose.Result()
            result.error_code = 2
            result.error_msg = "Nav2 NavigateToPose action server not available."
            return result

        nav2_goal = NavigateToPose.Goal()
        nav2_goal.pose = base_goal
        nav2_goal.behavior_tree = goal_handle.request.behavior_tree

        send_future = self._nav2_client.send_goal_async(
            nav2_goal,
            feedback_callback=lambda fb: self._forward_feedback(goal_handle, fb),
        )
        rclpy.spin_until_future_complete(self, send_future)
        nav2_goal_handle = send_future.result()
        if nav2_goal_handle is None or not nav2_goal_handle.accepted:
            goal_handle.abort()
            result = NavigateToPose.Result()
            result.error_code = 3
            result.error_msg = "Nav2 rejected the NavigateToPose goal."
            return result

        with self._nav2_goal_lock:
            self._nav2_goal_handle = nav2_goal_handle

        result_future = nav2_goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        nav2_result = result_future.result().result
        nav2_status = result_future.result().status

        with self._nav2_goal_lock:
            self._nav2_goal_handle = None

        if nav2_status == GoalStatus.STATUS_SUCCEEDED:
            goal_handle.succeed()
            return nav2_result

        if nav2_status == GoalStatus.STATUS_CANCELED:
            goal_handle.canceled()
            return nav2_result

        goal_handle.abort()
        return nav2_result

    def _forward_feedback(self, goal_handle, nav2_feedback_msg) -> None:
        goal_handle.publish_feedback(nav2_feedback_msg.feedback)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigateEoaServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
