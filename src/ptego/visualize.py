from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptego.episodes import Episode, read_episodes
from ptego.schema import TASKS

ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def figure_array(fig) -> np.ndarray:
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def signal_panel(episode: Episode, step: int, width: int) -> np.ndarray:
    fig, axes = plt.subplots(1, 3, figsize=(width / 100, 2.2), dpi=100)
    time = np.arange(len(episode))
    for i, label in enumerate(ACTION_LABELS[:3]):
        axes[0].plot(time, episode.actions[:, i], label=label)
    axes[0].set_title("translation action")
    for i, label in enumerate(ACTION_LABELS[3:6], 3):
        axes[1].plot(time, episode.actions[:, i], label=label)
    axes[1].set_title("rotation action")
    axes[2].plot(time, episode.actions[:, 6], label="command")
    axes[2].plot(time, episode.state[:, 9] / max(episode.state[:, 9].max(), 1e-6), label="opening")
    axes[2].set_title("gripper")
    for ax in axes:
        ax.axvline(step, color="black", lw=1)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    return figure_array(fig)


def frame(episode: Episode, step: int, panel: np.ndarray) -> np.ndarray:
    cameras = np.concatenate([episode.image[step], episode.wrist_image[step]], 1)
    scale = panel.shape[1] // cameras.shape[1]
    cameras = cameras.repeat(scale, 0).repeat(scale, 1)
    pad = panel.shape[1] - cameras.shape[1]
    cameras = np.pad(cameras, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), constant_values=255)
    return np.concatenate([cameras, panel], 0)


def video(episode: Episode, out: Path, stride: int) -> None:
    width = episode.image.shape[2] * 2 * 2
    frames = [frame(episode, t, signal_panel(episode, t, width)) for t in range(0, len(episode), stride)]
    iio.imwrite(out, np.stack(frames), fps=max(1, 20 // stride), codec="libx264")


def contact_sheet(episodes: list[Episode], out: Path, columns: int) -> None:
    fig, axes = plt.subplots(len(episodes), columns, figsize=(columns * 1.6, len(episodes) * 1.7), squeeze=False)
    for row, episode in zip(axes, episodes):
        steps = np.linspace(0, len(episode) - 1, columns).astype(int)
        for ax, step in zip(row, steps):
            ax.imshow(episode.image[step])
            ax.set_title(f"t={step}", fontsize=6)
            ax.axis("off")
        row[0].text(-10, 112, f"seed {episode.seed}\n{'ok' if episode.success else 'fail'}", fontsize=7, ha="right", va="center")
    fig.suptitle(TASKS.get(episodes[0].env_id, episodes[0].env_id), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument("--sheet", type=int, default=6)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = []
    for episode in read_episodes(args.trajectory):
        if episode.index < args.videos:
            video(episode, args.out / f"episode_{episode.index:03d}.mp4", args.stride)
        if episode.index < args.sheet:
            episodes.append(episode)
    contact_sheet(episodes, args.out / "sheet.png", columns=8)


if __name__ == "__main__":
    main()
