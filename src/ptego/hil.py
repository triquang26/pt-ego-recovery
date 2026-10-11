from __future__ import annotations

from ptego.mixture import Pool, Term
from ptego.store import Anchor, Episode, Store


def acbc(
    store: Store, nominal: list[Episode], recoveries: list[Episode], selected: list[Anchor]
) -> tuple[tuple[Term, ...], dict[str, Pool]]:
    terms = (
        Term("nominal", 0.5, "mean", 6, advantage=True),
        Term("selected", 0.4, "mean", 6, advantage=True),
        Term("negative", 0.1, "mean", 6, advantage=False),
    )
    pools = {
        "nominal": Pool(tuple(store.frames(nominal))),
        "selected": Pool(tuple(selected)),
        "negative": Pool(tuple(store.setup_frames(recoveries))),
    }
    return terms, pools
