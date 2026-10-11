from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np

from ptego.policy import ACTION_DIM, Adapter, Sampler
from ptego.store import Anchor, Episode, Store


def energy_score(samples: np.ndarray, target: np.ndarray) -> float:
    x = samples.reshape(len(samples), -1)
    spread = np.linalg.norm(x[:, None] - x[None], axis=-1).sum() / (len(x) * (len(x) - 1))
    return float(np.linalg.norm(x - target.reshape(-1), axis=-1).mean() - 0.5 * spread)


def anchors(episodes: list[Episode], stride: int) -> list[Anchor]:
    if episodes and episodes[0].recovery is None:
        return [Anchor(e.id, t, -1) for e in episodes for t in range(0, e.length, stride)]
    return [
        Anchor(e.id, t, k)
        for e in episodes
        for k, stage in enumerate(e.recovery.stages)
        for t in range(stage.start, stage.end, stride)
    ]


def score(
    store: Store,
    sampler: Sampler,
    adapter: Adapter,
    targets: list[Anchor],
    horizon: int,
    samples: int = 10,
    batch: int = 8,
    advantage: bool | None = None,
    seed: int = 0,
) -> list[dict]:
    results, rng = [], jax.random.key(seed)
    for i in range(0, len(targets), batch):
        group = targets[i : i + batch]
        padded = group + [group[-1]] * (batch - len(group))
        observation, actions = Adapter.collate([adapter(store.sample(a, horizon), advantage) for a in padded])
        predicted = sampler(jax.random.fold_in(rng, i), observation, samples)
        for anchor, draws, target in zip(group, predicted, actions[..., :ACTION_DIM]):
            results.append(
                {
                    "episode": anchor.episode,
                    "t": anchor.t,
                    "stage": anchor.stage,
                    "energy": energy_score(draws, target),
                    "error": float(((draws - target) ** 2).mean()),
                }
            )
    return results


def stage_table(results: list[dict], key: str = "energy") -> dict[str, list[float]]:
    cells: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        cells[r["episode"]][r["stage"]].append(r[key])
    return {e: [float(np.mean(v[k])) for k in sorted(v)] for e, v in cells.items()}


def priority(stages: list[float]) -> float:
    return 0.5 * float(np.mean(stages)) + 0.5 * float(np.max(stages))


def save(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results))


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())
