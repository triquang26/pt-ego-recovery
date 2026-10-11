from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

from ptego.episodes import read_episodes
from ptego.schema import ACTION_DIM, IMAGE_SIZE, STATE_DIM, TASKS

FPS = 20


def features() -> dict:
    image = {"dtype": "image", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channel"]}
    return {
        "image": image,
        "wrist_image": image,
        "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        "advantage": {"dtype": "float32", "shape": (1,), "names": ["advantage"]},
    }


def convert(trajectories: list[Path], repo_id: str, advantage: float) -> Path:
    output = HF_LEROBOT_HOME / repo_id
    if output.exists():
        shutil.rmtree(output)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=FPS,
        features=features(),
        image_writer_threads=8,
        image_writer_processes=4,
    )
    sources = []
    for path in trajectories:
        for episode in read_episodes(path):
            if not episode.success:
                continue
            for t in range(len(episode)):
                dataset.add_frame(
                    {
                        "image": episode.image[t],
                        "wrist_image": episode.wrist_image[t],
                        "state": episode.state[t],
                        "actions": episode.actions[t],
                        "advantage": np.array([advantage], dtype=np.float32),
                        "task": TASKS[episode.env_id],
                    }
                )
            dataset.save_episode()
            sources.append({"trajectory": str(path), "episode": episode.index, "seed": episode.seed, "steps": len(episode)})
    (output / "meta" / "sources.json").write_text(json.dumps(sources, indent=1))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", type=Path, nargs="+")
    parser.add_argument("--repo-id", default="local/stackcube_initial")
    parser.add_argument("--advantage", type=float, default=1.0)
    args = parser.parse_args()
    print(convert(args.trajectories, args.repo_id, args.advantage))


if __name__ == "__main__":
    main()
