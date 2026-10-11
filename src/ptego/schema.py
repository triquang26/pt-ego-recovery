from __future__ import annotations

import numpy as np

STATE_DIM = 10
ACTION_DIM = 7
IMAGE_SIZE = 224

TASKS = {
    "StackCube-v1": "Stack the red cube on the green cube.",
    "PegInsertionSide-v1": "Insert the peg into the hole on the side of the box.",
}


def advantage_prompt(task: str, positive: bool) -> str:
    return f"{task} Advantage: {'positive' if positive else 'negative'}."


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quat, dtype=np.float64), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def rotation_6d(quat: np.ndarray) -> np.ndarray:
    matrix = quat_to_matrix(quat)
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], -1)


def state_vector(tcp_pose: np.ndarray, gripper_opening: np.ndarray) -> np.ndarray:
    tcp_pose = np.asarray(tcp_pose)
    opening = np.asarray(gripper_opening)[..., None]
    return np.concatenate([tcp_pose[..., :3], rotation_6d(tcp_pose[..., 3:7]), opening], -1).astype(np.float32)
