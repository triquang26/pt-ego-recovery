from __future__ import annotations

from collections import defaultdict

import numpy as np

from ptego.scoring import priority, stage_table
from ptego.store import Anchor, Episode


def frames(episode: Episode) -> int:
    return episode.recovery.rejoin - episode.recovery.repair_start


def budget(episodes: list[Episode], fraction: float) -> float:
    return fraction * sum(frames(e) for e in episodes)


def fill(ordered: list[Episode], limit: float) -> list[Episode]:
    chosen, total = [], 0
    for episode in ordered:
        if total >= limit:
            break
        chosen.append(episode)
        total += frames(episode)
    return chosen


def uniform(episodes: list[Episode], fraction: float, rng: np.random.Generator) -> list[Episode]:
    return fill([episodes[i] for i in rng.permutation(len(episodes))], budget(episodes, fraction))


def brr(episodes: list[Episode], results: list[dict], fraction: float) -> list[Episode]:
    scores = {e: priority(s) for e, s in stage_table(results).items()}
    categories: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        categories[episode.recovery.failure_phase].append(episode)
    chosen = []
    for members in categories.values():
        ordered = sorted(members, key=lambda e: scores[e.id], reverse=True)
        chosen += fill(ordered, budget(members, fraction))
    return chosen


def es_chunk(episodes: list[Episode], results: list[dict], fraction: float, stride: int) -> list[Anchor]:
    lookup = {e.id: e for e in episodes}
    limit, chosen = budget(episodes, fraction), []
    for r in sorted(results, key=lambda r: r["energy"], reverse=True):
        if len(chosen) >= limit:
            break
        stage = lookup[r["episode"]].recovery.stages[r["stage"]]
        chosen += [Anchor(r["episode"], t, r["stage"]) for t in range(r["t"], min(r["t"] + stride, stage.end))]
    return chosen
