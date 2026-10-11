from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from ptego import scoring
from ptego.store import STAGES, Episode, Store

COLORS = {"setup": "#bbbbbb", "retract": "#e45756", "realign": "#f2a93b", "retry": "#4c78a8", "after": "#54a24b"}


def phase_at(episode: Episode, t: int) -> str:
    recovery = episode.recovery
    if t < recovery.repair_start:
        return "setup"
    if t >= recovery.rejoin:
        return "after"
    return recovery.stages[episode.stage_of(t)].name


def shade(ax, episode: Episode) -> None:
    for stage in episode.recovery.stages:
        ax.axvspan(stage.start, stage.end, color=COLORS[stage.name], alpha=0.25, lw=0)


def keyframes(episode: Episode) -> list[tuple[str, int]]:
    recovery = episode.recovery
    marks = [("before", max(recovery.repair_start - 30, 0)), ("repair start", recovery.repair_start)]
    marks += [(f"{s.name} end", min(s.end, episode.length - 1)) for s in recovery.stages]
    return marks


def episode_figure(store: Store, episode: Episode, curve: np.ndarray, stride: int, path: Path) -> None:
    marks = keyframes(episode)
    fig = plt.figure(figsize=(3 * len(marks), 8.5))
    grid = fig.add_gridspec(3, len(marks), height_ratios=[1.1, 1.1, 1.4])
    for i, (label, t) in enumerate(marks):
        images = store.images(episode, t)
        for row, camera in enumerate(("cam_high", "cam_right_wrist")):
            ax = fig.add_subplot(grid[row, i])
            ax.imshow(images[camera])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{label}\nt={t} ({phase_at(episode, t)})", fontsize=9)
    ax = fig.add_subplot(grid[2, :])
    states = store.states(episode)
    shade(ax, episode)
    ax.plot(np.arange(len(curve)) * stride, curve, "k", lw=2, label="nominal progress (DTW)")
    for j in range(7, 13):
        ax.plot(states[:, j] / 4 + 0.5, lw=0.6, alpha=0.7)
    ax.plot(states[:, 13] * 20, color="purple", lw=1.5, label="right gripper")
    for _, t in marks:
        ax.axvline(t, color="gray", ls=":", lw=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[n], alpha=0.4) for n in STAGES]
    ax.legend(
        ax.get_legend_handles_labels()[0][:2] + handles,
        ["nominal progress (DTW)", "right gripper", *STAGES],
        loc="upper left",
        fontsize=8,
        ncol=5,
    )
    ax.set_xlim(0, episode.length)
    ax.set_xlabel("frame (30 Hz)")
    rec = episode.recovery
    title = f"{episode.id} [{episode.split}]  failure phase: {rec.failure_phase}"
    fig.suptitle(f"{title}  repair {rec.repair_start}->{rec.rejoin}  source: {rec.source}")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def overview(store: Store, episodes: list[Episode], curves: dict, stride: int, path: Path) -> None:
    cols = 6
    rows = int(np.ceil(len(episodes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 1.9 * rows), squeeze=False)
    for ax, episode in zip(axes.flat, episodes):
        shade(ax, episode)
        curve = np.array(curves[episode.id])
        ax.plot(np.arange(len(curve)) * stride, curve, "k", lw=1)
        ax.plot(store.states(episode)[:, 13] * 20, color="purple", lw=0.8)
        ax.set_title(f"{episode.id} {episode.split} {episode.recovery.failure_phase}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(episodes) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def episode_video(store: Store, episode: Episode, path: Path, step: int = 2) -> None:
    frames = []
    for t in range(0, episode.length, step):
        images = store.images(episode, t)
        frame = Image.fromarray(np.concatenate([images["cam_high"], images["cam_right_wrist"]], 1))
        phase = phase_at(episode, t)
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 447, 18), fill=COLORS[phase])
        draw.text((4, 3), f"{episode.id}  t={t}  {phase}", fill=(0, 0, 0))
        frames.append(np.asarray(frame))
    iio.imwrite(path, np.stack(frames), fps=30 // step, codec="libx264")


def segments(root: Path, out: Path, videos: int) -> None:
    store = Store(root)
    progress = json.loads((root / "progress.json").read_text())
    episodes = store.select("recovery")
    out.mkdir(parents=True, exist_ok=True)
    overview(store, episodes, progress["curves"], progress["stride"], out / "overview.png")
    for episode in episodes:
        episode_figure(
            store, episode, np.array(progress["curves"][episode.id]), progress["stride"], out / f"{episode.id}.png"
        )
    for episode in episodes[:videos]:
        episode_video(store, episode, out / f"{episode.id}.mp4")


def scores(root: Path, round_dir: Path) -> None:
    store = Store(root)
    table = scoring.stage_table(scoring.load(round_dir / "scores.json"))
    chosen = set(json.loads((round_dir / "selection.json").read_text()).get("episodes", []))
    order = sorted(table, key=lambda e: scoring.priority(table[e]), reverse=True)
    matrix = np.array([table[e] for e in order])
    fig, ax = plt.subplots(figsize=(6, 0.22 * len(order) + 1.5))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(STAGES)), STAGES)
    mark = lambda e: "* " if e in chosen else ""
    phase = lambda e: store.episodes[e].recovery.failure_phase
    labels = [f"{mark(e)}{e} ({phase(e)}) S={scoring.priority(table[e]):.2f}" for e in order]
    ax.set_yticks(range(len(order)), labels, fontsize=7)
    fig.colorbar(image, label="energy score")
    ax.set_title("stage energy score, * = selected")
    fig.tight_layout()
    fig.savefig(round_dir / "scores.png", dpi=80)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    seg = sub.add_parser("segments")
    seg.add_argument("--root", type=Path, default=Path("data/reboot/usb_a"))
    seg.add_argument("--out", type=Path, default=Path("runs/usb_a/viz/segments"))
    seg.add_argument("--videos", type=int, default=4)
    sco = sub.add_parser("scores")
    sco.add_argument("round_dir", type=Path)
    sco.add_argument("--root", type=Path, default=Path("data/reboot/usb_a"))
    args = parser.parse_args()
    if args.command == "segments":
        segments(args.root, args.out, args.videos)
    else:
        scores(args.root, args.round_dir)


if __name__ == "__main__":
    main()
