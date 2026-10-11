from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

ORG = "REBOOT26"
FPS = 30
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LOWDIM = ("action", "observation.state", "episode_index", "frame_index")


@dataclasses.dataclass(frozen=True)
class Task:
    prompt: str
    nominal: str
    recovery: str


TASKS = {
    "usb_a": Task(
        "Pick the USB-A connector and install it on the taskboard.", "usbA-install", "USB-A_recovery_install"
    ),
    "cylinder": Task(
        "Pick the 16mm cylinder and install it on the taskboard.",
        "16mm-cylinder_install",
        "16mm-cylinder_recovery_install",
    ),
}


def filesystem() -> HfFileSystem:
    return HfFileSystem(token=False)


def episodes(repo: str) -> pd.DataFrame:
    fs = filesystem()
    paths = sorted(fs.glob(f"datasets/{ORG}/{repo}/meta/episodes/*/*.parquet"))
    with ThreadPoolExecutor(4) as pool:
        frames = list(pool.map(lambda p: pd.read_parquet(fs.open(p)), paths))
    return pd.concat(frames).sort_values("episode_index").reset_index(drop=True)


def lowdim(repo: str) -> dict[str, np.ndarray]:
    fs = filesystem()
    paths = sorted(fs.glob(f"datasets/{ORG}/{repo}/data/*/*.parquet"))

    def read(path: str) -> pa.Table:
        with fs.open(path, block_size=1 << 20) as f:
            return pq.ParquetFile(f).read(columns=list(LOWDIM))

    with ThreadPoolExecutor(16) as pool:
        table = pa.concat_tables(list(pool.map(read, paths)))
    order = np.lexsort((table["frame_index"].to_numpy(), table["episode_index"].to_numpy()))
    column = lambda name: np.stack(table[name].to_numpy(zero_copy_only=False))[order]
    return {
        "action": column("action").astype(np.float32),
        "state": column("observation.state").astype(np.float32),
        "episode": table["episode_index"].to_numpy()[order],
        "frame": table["frame_index"].to_numpy()[order],
    }


def video(repo: str, camera: str, chunk: int, file: int, cache: Path) -> Path:
    name = f"videos/observation.images.{camera}/chunk-{chunk:03d}/file-{file:03d}.mp4"
    return Path(hf_hub_download(f"{ORG}/{repo}", name, repo_type="dataset", local_dir=cache, token=False))
