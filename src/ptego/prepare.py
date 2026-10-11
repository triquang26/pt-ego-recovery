from __future__ import annotations

import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import h5py
import numpy as np
from openpi_client.image_tools import resize_with_pad
from PIL import Image

from ptego import reboot
from ptego.store import IMAGE_SIZE, Store

MOTION_TOL = 1e-3
MARGIN = 10
DUPLICATE_TOL = 1e-3
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}


def active_window(actions: np.ndarray) -> tuple[int, int]:
    moving = np.flatnonzero(np.abs(np.diff(actions, axis=0)).max(1) > MOTION_TOL)
    if len(moving) == 0:
        return 0, len(actions)
    return max(int(moving[0]) - MARGIN, 0), min(int(moving[-1]) + 1 + MARGIN, len(actions))


def signature(actions: np.ndarray, points: int = 128) -> np.ndarray:
    grid = np.linspace(0, len(actions) - 1, points)
    return np.stack([np.interp(grid, np.arange(len(actions)), actions[:, d]) for d in range(actions.shape[1])], 1)


def duplicate_groups(trajectories: list[np.ndarray]) -> list[int]:
    signatures = np.stack([signature(t) for t in trajectories])
    parent = list(range(len(trajectories)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(signatures)):
        distance = np.abs(signatures[i + 1 :] - signatures[i]).mean((1, 2))
        for j in np.flatnonzero(distance < DUPLICATE_TOL) + i + 1:
            parent[root(int(j))] = root(i)
    roots = [root(i) for i in range(len(trajectories))]
    index = {r: g for g, r in enumerate(dict.fromkeys(roots))}
    return [index[r] for r in roots]


def assign_splits(groups: list[int], seed: int) -> dict[int, str]:
    unique = np.random.default_rng(seed).permutation(sorted(set(groups)))
    bounds = np.cumsum([round(f * len(unique)) for f in SPLITS.values()])
    return {int(g): name for name, part in zip(SPLITS, np.split(unique, bounds[:-1])) for g in part}


def collect(task: reboot.Task, seed: int) -> tuple[list[dict], dict[str, np.ndarray]]:
    records, states, actions = [], [], []
    offset = 0
    for kind, repo in (("nominal", task.nominal), ("recovery", task.recovery)):
        meta = reboot.episodes(repo)
        data = reboot.lowdim(repo)
        kind_records, trajectories = [], []
        for row in meta.itertuples():
            mask = data["episode"] == row.episode_index
            start, end = active_window(data["action"][mask])
            states.append(data["state"][mask][start:end])
            actions.append(data["action"][mask][start:end])
            trajectories.append(actions[-1])
            kind_records.append(
                {
                    "id": f"{kind}_{row.episode_index:03d}",
                    "kind": kind,
                    "repo": repo,
                    "episode_index": int(row.episode_index),
                    "offset": offset,
                    "length": end - start,
                    "raw_start": start,
                    "videos": {
                        camera: {
                            "chunk": int(getattr(meta, f"videos/observation.images.{camera}/chunk_index")[row.Index]),
                            "file": int(getattr(meta, f"videos/observation.images.{camera}/file_index")[row.Index]),
                            "from": float(
                                getattr(meta, f"videos/observation.images.{camera}/from_timestamp")[row.Index]
                            ),
                        }
                        for camera in reboot.CAMERAS
                    },
                }
            )
            offset += end - start
        groups = duplicate_groups(trajectories)
        splits = assign_splits(groups, seed)
        for record, group in zip(kind_records, groups):
            record["group"] = f"{kind}_{group:03d}"
            record["split"] = splits[group]
        records += kind_records
    return records, {"state": np.concatenate(states), "action": np.concatenate(actions)}


def encode(frame: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(resize_with_pad(frame[None], IMAGE_SIZE, IMAGE_SIZE)[0]).save(buffer, format="JPEG", quality=90)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def extract_camera(root: Path, camera: str, records: list[dict], total: int) -> str:
    cache = root / ".videos"
    files: dict[tuple[str, int, int], list[dict]] = {}
    for record in records:
        source = record["videos"][camera]
        files.setdefault((record["repo"], source["chunk"], source["file"]), []).append(record)
    with h5py.File(root / f"{camera}.h5", "w") as h5:
        frames = h5.create_dataset("jpeg", (total,), dtype=h5py.vlen_dtype(np.uint8))
        for (repo, chunk, file), members in sorted(files.items()):
            path = reboot.video(repo, camera, chunk, file, cache)
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for frame in container.decode(stream):
                    for record in members:
                        index = (
                            round((frame.time - record["videos"][camera]["from"]) * reboot.FPS) - record["raw_start"]
                        )
                        if 0 <= index < record["length"]:
                            frames[record["offset"] + index] = encode(frame.to_ndarray(format="rgb24"))
            path.unlink()
        written = sum(len(frames[i]) > 0 for i in range(total))
    return f"{camera}: {written}/{total} frames"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="usb_a", choices=sorted(reboot.TASKS))
    parser.add_argument("--root", type=Path, default=Path("data/reboot"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.root / args.task
    root.mkdir(parents=True, exist_ok=True)
    task = reboot.TASKS[args.task]
    records, arrays = collect(task, args.seed)
    np.savez(root / "lowdim.npz", **arrays)
    for record in records:
        print(record["id"], record["split"], record["group"], record["length"])
    duplicates = len(records) - len({r["group"] for r in records})
    print(f"episodes {len(records)}, frames {len(arrays['state'])}, duplicates merged {duplicates}")
    with ProcessPoolExecutor(len(reboot.CAMERAS)) as pool:
        jobs = [pool.submit(extract_camera, root, c, records, len(arrays["state"])) for c in reboot.CAMERAS]
        for job in jobs:
            print(job.result())
    for record in records:
        del record["videos"]
    Store.write_manifest(root, {"task": args.task, "prompt": task.prompt, "episodes": records})


if __name__ == "__main__":
    main()
