from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ptego.schema import state_vector


@dataclass
class Episode:
    index: int
    env_id: str
    seed: int
    success: bool
    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    actions: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)


def read_episodes(path: Path) -> Iterator[Episode]:
    meta = json.loads(path.with_suffix(".json").read_text())
    env_id = meta["env_info"]["env_id"]
    with h5py.File(path, "r") as f:
        for info in meta["episodes"]:
            group = f[f"traj_{info['episode_id']}"]
            obs = group["obs"]
            qpos = obs["agent/qpos"][:]
            steps = len(group["actions"])
            yield Episode(
                index=info["episode_id"],
                env_id=env_id,
                seed=info["episode_seed"],
                success=bool(info["success"]),
                image=obs["sensor_data/base_camera/rgb"][:steps],
                wrist_image=obs["sensor_data/hand_camera/rgb"][:steps],
                state=state_vector(obs["extra/tcp_pose"][:steps], qpos[:steps, -2:].sum(-1)),
                actions=group["actions"][:].astype(np.float32),
            )
