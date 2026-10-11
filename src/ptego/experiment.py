from __future__ import annotations

import argparse
import dataclasses
import gc
import json
from pathlib import Path

import jax
import numpy as np

from ptego import evaluate, hil, policy, scoring, selection
from ptego.mixture import Pool, Term
from ptego.store import Anchor, Episode, Store
from ptego.train import Schedule, train

SCORE_STRIDE = 6


@dataclasses.dataclass(frozen=True)
class Run:
    name: str
    selection: str
    loss: str
    acbc: bool = False


RUNS = {
    "A": Run("nominal_sft", "none", "mean"),
    "B": Run("uniform_recovery", "uniform", "mean"),
    "C": Run("es_chunk", "es_chunk", "mean"),
    "D": Run("hil_umi_offline", "es_chunk", "mean", acbc=True),
    "E": Run("brr_selection", "brr", "mean"),
    "F": Run("brr_loss", "uniform", "brr"),
    "G": Run("brr_full", "brr", "brr"),
}


def recovery_term(name: str, weight: float, loss: str) -> Term:
    return Term(name, weight, loss, 1 if loss != "mean" else 6)


def episode_pool(store: Store, episodes: list[Episode]) -> Pool:
    return Pool(tuple(store.stage_frames(episodes)), tuple(episodes))


def mixture(store: Store, run: Run, chosen: list[Episode] | list[Anchor]) -> tuple[tuple[Term, ...], dict[str, Pool]]:
    nominal = store.select("nominal", "train")
    recoveries = store.select("recovery", "train")
    if run.selection == "none":
        return (Term("nominal", 1.0, "mean", 18),), {"nominal": Pool(tuple(store.frames(nominal)))}
    if run.acbc:
        return hil.acbc(store, nominal, recoveries, chosen)
    selected = Pool(tuple(chosen)) if run.selection == "es_chunk" else episode_pool(store, chosen)
    terms = (
        Term("nominal", 0.5, "mean", 6),
        recovery_term("selected", 0.4, run.loss),
        recovery_term("random", 0.1, run.loss),
    )
    pools = {
        "nominal": Pool(tuple(store.frames(nominal))),
        "selected": selected,
        "random": episode_pool(store, recoveries),
    }
    return terms, pools


def score_train(store: Store, lora: Path, advantage: bool | None, seed: int) -> list[dict]:
    config = policy.model_config()
    sampler = policy.Sampler(config, policy.create(config, jax.random.key(seed), lora))
    adapter = policy.Adapter(policy.norm_stats(store), config)
    targets = scoring.anchors(store.select("recovery", "train"), SCORE_STRIDE)
    results = scoring.score(store, sampler, adapter, targets, config.action_horizon, advantage=advantage, seed=seed)
    del sampler
    gc.collect()
    return results


def choose(store: Store, run: Run, results: list[dict] | None, fraction: float, rng: np.random.Generator):
    recoveries = store.select("recovery", "train")
    if run.selection == "uniform":
        return selection.uniform(recoveries, fraction, rng)
    if run.selection == "brr":
        return selection.brr(recoveries, results, fraction)
    if run.selection == "es_chunk":
        return selection.es_chunk(recoveries, results, fraction, SCORE_STRIDE)
    return []


def describe(chosen: list) -> dict:
    if chosen and isinstance(chosen[0], Anchor):
        return {"anchors": len(chosen), "episodes": sorted({a.episode for a in chosen})}
    return {"episodes": [e.id for e in chosen], "frames": sum(selection.frames(e) for e in chosen)}


def run_rounds(
    store: Store, key: str, base: Path, out: Path, seed: int, rounds: int, steps: int, fraction: float
) -> None:
    run = RUNS[key]
    advantage = True if run.acbc else None
    current, rng = base, np.random.default_rng(seed)
    for r in range(rounds):
        folder = out / f"round{r}"
        folder.mkdir(parents=True, exist_ok=True)
        results = None
        if run.selection in ("brr", "es_chunk"):
            results = score_train(store, current, advantage, seed)
            scoring.save(folder / "scores.json", results)
        chosen = choose(store, run, results, fraction, rng)
        (folder / "selection.json").write_text(json.dumps(describe(chosen), indent=1))
        terms, pools = mixture(store, run, chosen)
        schedule = Schedule(steps=steps, warmup=min(100, steps // 10))
        previous = current
        current = train(store, policy.model_config(), terms, pools, schedule, current, folder, seed * 100 + r)
        if previous != base:
            previous.unlink()
        gc.collect()
    evaluate.evaluate(store, current, out, advantage, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("base")
    base.add_argument("--steps", type=int, default=10_000)
    run = sub.add_parser("run")
    run.add_argument("key", choices=sorted(RUNS))
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--steps", type=int, default=3_000)
    run.add_argument("--fraction", type=float, default=0.25)
    for p in (base, run):
        p.add_argument("--root", type=Path, default=Path("data/reboot/usb_a"))
        p.add_argument("--runs", type=Path, default=Path("runs/usb_a"))
    args = parser.parse_args()
    store = Store(args.root)
    if args.command == "base":
        out = args.runs / "base"
        terms, pools = mixture(store, RUNS["A"], [])
        lora = train(store, policy.model_config(), terms, pools, Schedule(args.steps, 500), None, out, 0)
        evaluate.evaluate(store, lora, out)
        return
    out = args.runs / f"{args.key}_{RUNS[args.key].name}" / f"frac{args.fraction:g}" / f"seed{args.seed}"
    run_rounds(store, args.key, args.runs / "base" / "lora.npz", out, args.seed, args.rounds, args.steps, args.fraction)


if __name__ == "__main__":
    main()
