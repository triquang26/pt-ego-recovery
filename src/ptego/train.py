from __future__ import annotations

import dataclasses
import functools
import json
import time
from pathlib import Path

import flax.nnx as nnx
import jax
import optax
from openpi.models import pi0_config
from openpi.training import optimizer as _optimizer
from openpi.training import sharding

from ptego import policy
from ptego.mixture import Batches, Pool, Term, objective
from ptego.store import Store


@dataclasses.dataclass(frozen=True)
class Schedule:
    steps: int
    warmup: int
    peak_lr: float = 2.5e-5
    end_lr: float = 2.5e-6

    def optimizer(self) -> optax.GradientTransformation:
        lr = _optimizer.CosineDecaySchedule(self.warmup, self.peak_lr, self.steps, self.end_lr)
        return _optimizer.create_optimizer(_optimizer.AdamW(), lr)


def train(
    store: Store,
    config: pi0_config.Pi0Config,
    terms: tuple[Term, ...],
    pools: dict[str, Pool],
    schedule: Schedule,
    init: Path | None,
    out: Path,
    seed: int,
    log_every: int = 50,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng = jax.random.key(seed)
    params = jax.device_put(policy.create(config, rng, init), replicated)
    learnable = policy.trainable(config)
    tx = schedule.optimizer()
    opt_state = tx.init(params.filter(learnable))
    graphdef = policy.graphdef(config)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt_state, rng, observation, actions):
        model = nnx.merge(graphdef, params)
        model.train()

        def loss_fn(model):
            per_sample = model.compute_loss(rng, observation, actions, train=True).mean(-1)
            return objective(terms, per_sample)

        (loss, parts), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, learnable), has_aux=True)(model)
        current = params.filter(learnable)
        updates, opt_state = tx.update(grads, opt_state, current)
        nnx.update(model, optax.apply_updates(current, updates))
        return nnx.state(model), opt_state, {"loss": loss, "grad_norm": optax.global_norm(grads), **parts}

    batches = Batches(
        store, policy.Adapter(policy.norm_stats(store), config), config.action_horizon, terms, pools, seed
    )
    history, start = [], time.time()
    with (out / "train.jsonl").open("w") as log, sharding.set_mesh(mesh):
        for i in range(schedule.steps):
            observation, actions = jax.device_put(next(batches), data_sharding)
            params, opt_state, info = step(params, opt_state, jax.random.fold_in(rng, i), observation, actions)
            history.append(info)
            if (i + 1) % log_every == 0 or i == schedule.steps - 1:
                mean = {k: float(sum(float(h[k]) for h in history) / len(history)) for k in history[0]}
                record = {"step": i + 1, "seconds": round(time.time() - start, 1), **mean}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                history = []
    path = out / "lora.npz"
    policy.save_lora(params, config, path)
    return path
