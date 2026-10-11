from __future__ import annotations

import dataclasses
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import jax.numpy as jnp
import numpy as np

from ptego.losses import LOSSES, STRUCTURED
from ptego.policy import Adapter
from ptego.store import STAGES, Anchor, Episode, Store


@dataclasses.dataclass(frozen=True)
class Pool:
    anchors: tuple[Anchor, ...] = ()
    episodes: tuple[Episode, ...] = ()


@dataclasses.dataclass(frozen=True)
class Term:
    name: str
    weight: float
    loss: str
    size: int
    per_stage: int = 2
    advantage: bool | None = None

    @property
    def structured(self) -> bool:
        return self.loss in STRUCTURED

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.size, len(STAGES), self.per_stage) if self.structured else (self.size,)

    @property
    def count(self) -> int:
        return int(np.prod(self.shape))

    def draw(self, pool: Pool, rng: np.random.Generator) -> list[Anchor]:
        if not self.structured:
            return [pool.anchors[i] for i in rng.integers(len(pool.anchors), size=self.count)]
        anchors = []
        for i in rng.integers(len(pool.episodes), size=self.size):
            episode = pool.episodes[i]
            for k, stage in enumerate(episode.recovery.stages):
                anchors += [Anchor(episode.id, int(t), k) for t in rng.integers(stage.start, stage.end, self.per_stage)]
        return anchors


def objective(terms: tuple[Term, ...], per_sample: jnp.ndarray) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    parts, offset = {}, 0
    for term in terms:
        parts[term.name] = LOSSES[term.loss](per_sample[offset : offset + term.count].reshape(term.shape))
        offset += term.count
    return sum(term.weight * parts[term.name] for term in terms), parts


class Batches:
    def __init__(
        self,
        store: Store,
        adapter: Adapter,
        horizon: int,
        terms: tuple[Term, ...],
        pools: dict[str, Pool],
        seed: int,
        prefetch: int = 4,
        workers: int = 16,
    ):
        self.store, self.adapter, self.horizon = store, adapter, horizon
        self.terms, self.pools = terms, pools
        self.rng = np.random.default_rng(seed)
        self.executor = ThreadPoolExecutor(workers)
        self.queue: queue.Queue = queue.Queue(prefetch)
        threading.Thread(target=self._fill, daemon=True).start()

    def _item(self, anchor: Anchor, advantage: bool | None) -> dict:
        return self.adapter(self.store.sample(anchor, self.horizon), advantage)

    def _fill(self) -> None:
        while True:
            jobs = [
                (anchor, term.advantage) for term in self.terms for anchor in term.draw(self.pools[term.name], self.rng)
            ]
            self.queue.put(Adapter.collate(list(self.executor.map(lambda job: self._item(*job), jobs))))

    def __iter__(self):
        return self

    def __next__(self):
        return self.queue.get()
