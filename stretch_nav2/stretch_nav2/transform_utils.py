"""SE(3) helpers for eoa-goal to base-goal conversion."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(rot: np.ndarray):
    from geometry_msgs.msg import Quaternion

    m = rot
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = Quaternion()
    q.x, q.y, q.z, q.w = x, y, z, w
    return q


def transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    r = transform.transform.rotation
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_to_matrix(r)
    mat[:3, 3] = [t.x, t.y, t.z]
    return mat


def pose_to_matrix(pose: Pose) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_to_matrix(pose.orientation)
    mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return mat


def matrix_to_pose(mat: np.ndarray) -> Pose:
    from geometry_msgs.msg import Point, Pose, Quaternion

    pose = Pose()
    pose.position = Point(x=float(mat[0, 3]), y=float(mat[1, 3]), z=float(mat[2, 3]))
    pose.orientation = _matrix_to_quat(mat[:3, :3])
    return pose
def compute_base_goal_from_eoa_goal(
    eoa_goal: PoseStamped,
    t_map_base: TransformStamped,
    t_map_eoa: TransformStamped,
    base_frame: str,
) -> PoseStamped:
    """
    Compute the base_link goal given a desired eoa pose in the map.

    Uses the current arm configuration via TF:
        T_map_base_goal = T_map_eoa_goal * inv(T_base_eoa)
    where T_base_eoa = inv(T_map_base) * T_map_eoa.
    """
    t_map_base_mat = transform_to_matrix(t_map_base)
    t_map_eoa_mat = transform_to_matrix(t_map_eoa)
    t_base_eoa = np.linalg.inv(t_map_base_mat) @ t_map_eoa_mat

    t_map_eoa_goal = pose_to_matrix(eoa_goal.pose)
    t_map_base_goal = t_map_eoa_goal @ np.linalg.inv(t_base_eoa)

    base_goal = PoseStamped()
    base_goal.header = eoa_goal.header
    base_goal.header.frame_id = eoa_goal.header.frame_id
    base_goal.pose = matrix_to_pose(t_map_base_goal)
    return base_goal