from __future__ import annotations

import argparse
import json
from pathlib import Path

import numba
import numpy as np

from ptego.store import STAGES, Episode, Store

STRIDE = 2
NEIGHBOURS = 5
GRIPPERS = (6, 13)
WEIGHTS = np.array([0.3] * 6 + [8.0] + [1.0] * 6 + [8.0], dtype=np.float32)
MIN_REPAIR = 15


@numba.njit(cache=True)
def dtw(cost: np.ndarray) -> tuple[float, np.ndarray]:
    n, m = cost.shape
    total = np.full((n + 1, m + 1), np.inf)
    total[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            total[i, j] = cost[i - 1, j - 1] + min(total[i - 1, j], total[i, j - 1], total[i - 1, j - 1])
    match = np.zeros(n, dtype=np.int64)
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        match[i - 1] = max(match[i - 1], j - 1)
        steps += 1
        options = np.array([total[i - 1, j - 1], total[i - 1, j], total[i, j - 1]])
        move = np.argmin(options)
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return total[n, m] / steps, match


def features(states: np.ndarray) -> np.ndarray:
    return (states * WEIGHTS)[::STRIDE]


def align(query: np.ndarray, template: np.ndarray) -> tuple[float, np.ndarray]:
    cost = np.sqrt(((query[:, None] - template[None]) ** 2).sum(-1))
    score, match = dtw(cost)
    return score, match / (len(template) - 1)


def progress(query: np.ndarray, templates: list[np.ndarray]) -> np.ndarray:
    results = sorted((align(query, t) for t in templates), key=lambda r: r[0])[:NEIGHBOURS]
    return np.median(np.stack([r[1] for r in results]), 0)


def longest_stall(curve: np.ndarray, pace: float) -> tuple[int, int, float]:
    gain = 0.5 - pace * np.diff(curve)
    best, start, best_span, running = 0.0, 0, (0, 0), 0.0
    for t, g in enumerate(gain):
        if running <= 0:
            running, start = 0.0, t
        running += g
        if running > best:
            best, best_span = running, (start, t + 1)
    return best_span[0], best_span[1], float(best)


def split_stages(arm: np.ndarray, start: int, end: int) -> list[tuple[int, int]]:
    window = arm[start : end + 1]
    away = np.linalg.norm(window - window[0], axis=1)
    back = np.linalg.norm(window - window[-1], axis=1)
    first = start + int(np.argmax(away >= 0.5 * away.max()))
    second = start + len(window) - 1 - int(np.argmax((back >= 0.5 * back.max())[::-1]))
    shortest = max(3, (end - start) // 10)
    if min(first - start, second - first, end - second) < shortest:
        first, second = start + (end - start) // 3, start + 2 * (end - start) // 3
    return [(start, first), (first, second), (second, end)]


def gripper_events(actions: np.ndarray) -> tuple[float, float]:
    gripper = actions[:, GRIPPERS].max(1)
    opened = gripper > 0.5 * gripper.max()
    edges = np.flatnonzero(opened[1:] & ~opened[:-1]) + 1
    closes = np.flatnonzero(~opened[1:] & opened[:-1]) + 1
    grasp = closes[0] if len(closes) else len(actions) // 3
    release = edges[-1] if len(edges) else 2 * len(actions) // 3
    return grasp / (len(actions) - 1), release / (len(actions) - 1)


def segment(store: Store, episode: Episode, templates: list[np.ndarray], pace: float, boundary: float) -> dict:
    states = store.states(episode)
    curve = progress(features(states), templates)
    start, end, strength = longest_stall(curve, pace)
    start, end = start * STRIDE, min(end * STRIDE, episode.length - 1)
    if strength * STRIDE < MIN_REPAIR:
        return {"curve": curve.tolist(), "found": False}
    stages = split_stages(states * WEIGHTS, start, end)
    return {
        "curve": curve.tolist(),
        "found": True,
        "recovery": {
            "repair_start": start,
            "rejoin": end,
            "stages": [{"name": n, "start": int(a), "end": int(b)} for n, (a, b) in zip(STAGES, stages)],
            "failure_phase": "pick" if curve[start // STRIDE] < boundary else "place",
            "outcome": "recovered",
            "source": "auto",
            "strength": strength * STRIDE,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/reboot/usb_a"))
    args = parser.parse_args()
    store = Store(args.root)
    nominal = store.select("nominal", "train")
    templates = [features(store.states(e)) for e in nominal]
    pace = float(np.median([len(t) for t in templates]))
    events = np.array([gripper_events(store.actions(e)) for e in nominal])
    boundary = float(np.median(events.mean(1)))
    curves = {}
    for record in store.manifest["episodes"]:
        if record["kind"] != "recovery" or record.get("recovery", {}).get("source") == "manual":
            continue
        result = segment(store, store.episodes[record["id"]], templates, pace, boundary)
        curves[record["id"]] = result["curve"]
        record.pop("recovery", None)
        if result["found"]:
            record["recovery"] = result["recovery"]
        summary = result.get("recovery", {})
        print(
            record["id"],
            record["split"],
            summary.get("failure_phase"),
            [(s["start"], s["end"]) for s in summary.get("stages", [])],
        )
    Store.write_manifest(args.root, store.manifest)
    (args.root / "progress.json").write_text(json.dumps({"stride": STRIDE, "boundary": boundary, "curves": curves}))


if __name__ == "__main__":
    main()
