from __future__ import annotations

import jax.numpy as jnp


def mean(per_sample: jnp.ndarray) -> jnp.ndarray:
    return per_sample.mean()


def stage_balanced(per_sample: jnp.ndarray) -> jnp.ndarray:
    return per_sample.mean(-1).mean()


def brr(per_sample: jnp.ndarray) -> jnp.ndarray:
    stages = per_sample.mean(-1)
    return (0.5 * stages.mean(-1) + 0.5 * stages.max(-1)).mean()


LOSSES = {"mean": mean, "stage_balanced": stage_balanced, "brr": brr}
STRUCTURED = {"stage_balanced", "brr"}
