from __future__ import annotations

import dataclasses
import io
import json
import threading
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from ptego.reboot import CAMERAS

IMAGE_SIZE = 224
STAGES = ("retract", "realign", "retry")


@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class Recovery:
    repair_start: int
    rejoin: int
    stages: tuple[Stage, ...]
    failure_phase: str
    source: str


@dataclasses.dataclass(frozen=True)
class Episode:
    id: str
    kind: str
    split: str
    group: str
    offset: int
    length: int
    recovery: Recovery | None

    def stage_of(self, t: int) -> int:
        for k, stage in enumerate(self.recovery.stages):
            if stage.start <= t < stage.end:
                return k
        return -1


@dataclasses.dataclass(frozen=True)
class Anchor:
    episode: str
    t: int
    stage: int = -1


def parse_recovery(data: dict | None) -> Recovery | None:
    if not data:
        return None
    stages = tuple(Stage(s["name"], s["start"], s["end"]) for s in data["stages"])
    return Recovery(data["repair_start"], data["rejoin"], stages, data["failure_phase"], data["source"])


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        arrays = np.load(self.root / "lowdim.npz")
        self.state, self.action = arrays["state"], arrays["action"]
        self.episodes = {
            e["id"]: Episode(
                e["id"], e["kind"], e["split"], e["group"], e["offset"], e["length"], parse_recovery(e.get("recovery"))
            )
            for e in self.manifest["episodes"]
        }
        self._local = threading.local()

    @property
    def prompt(self) -> str:
        return self.manifest["prompt"]

    @staticmethod
    def write_manifest(root: Path, manifest: dict) -> None:
        (Path(root) / "manifest.json").write_text(json.dumps(manifest, indent=1))

    def select(self, kind: str | None = None, split: str | None = None) -> list[Episode]:
        return [
            e
            for e in self.episodes.values()
            if (kind is None or e.kind == kind)
            and (split is None or e.split == split)
            and (e.kind != "recovery" or e.recovery is not None)
        ]

    def states(self, episode: Episode) -> np.ndarray:
        return self.state[episode.offset : episode.offset + episode.length]

    def actions(self, episode: Episode) -> np.ndarray:
        return self.action[episode.offset : episode.offset + episode.length]

    def chunk(self, anchor: Anchor, horizon: int) -> np.ndarray:
        episode = self.episodes[anchor.episode]
        index = np.minimum(np.arange(anchor.t, anchor.t + horizon), episode.length - 1)
        return self.action[episode.offset + index]

    def _files(self) -> dict[str, h5py.Dataset]:
        if not hasattr(self._local, "files"):
            self._local.files = {c: h5py.File(self.root / f"{c}.h5", "r")["jpeg"] for c in CAMERAS}
        return self._local.files

    def images(self, episode: Episode, t: int) -> dict[str, np.ndarray]:
        files = self._files()
        return {c: np.asarray(Image.open(io.BytesIO(files[c][episode.offset + t].tobytes()))) for c in CAMERAS}

    def sample(self, anchor: Anchor, horizon: int) -> dict:
        episode = self.episodes[anchor.episode]
        return {
            "images": self.images(episode, anchor.t),
            "state": self.state[episode.offset + anchor.t],
            "actions": self.chunk(anchor, horizon),
            "prompt": self.prompt,
        }

    def frames(self, episodes: list[Episode], start: int = 0) -> list[Anchor]:
        return [Anchor(e.id, t, -1) for e in episodes for t in range(start, e.length)]

    def stage_frames(self, episodes: list[Episode]) -> list[Anchor]:
        return [
            Anchor(e.id, t, k)
            for e in episodes
            for k, stage in enumerate(e.recovery.stages)
            for t in range(stage.start, stage.end)
        ]

    def setup_frames(self, episodes: list[Episode]) -> list[Anchor]:
        return [Anchor(e.id, t, -1) for e in episodes for t in range(e.recovery.repair_start)]
