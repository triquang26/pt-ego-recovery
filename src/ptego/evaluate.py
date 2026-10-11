from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np

from ptego import policy, scoring
from ptego.store import STAGES, Store

RECOVERY_STRIDE = 5
NOMINAL_STRIDE = 15


def episode_means(results: list[dict], key: str, stage: int | None = None) -> np.ndarray:
    cells = defaultdict(list)
    for r in results:
        if stage is None or r["stage"] == stage:
            cells[r["episode"]].append(r[key])
    return np.array([np.mean(v) for v in cells.values()])


def interval(values: np.ndarray, seed: int = 0, resamples: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (resamples, len(values))).mean(1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(boot, 0.025)),
        "high": float(np.quantile(boot, 0.975)),
    }


def summarize(recovery: list[dict], nominal: list[dict]) -> dict:
    summary = {}
    for key in ("energy", "error"):
        worst = np.array([max(s) for s in scoring.stage_table(recovery, key).values()])
        summary[key] = {
            "recovery": interval(episode_means(recovery, key)),
            "stages": {name: interval(episode_means(recovery, key, k)) for k, name in enumerate(STAGES)},
            "worst_stage": interval(worst),
            "nominal": interval(episode_means(nominal, key)),
        }
    return summary


def evaluate(
    store: Store, lora: Path, out: Path, advantage: bool | None = None, samples: int = 10, seed: int = 0
) -> dict:
    config = policy.model_config()
    sampler = policy.Sampler(config, policy.create(config, jax.random.key(seed), lora))
    adapter = policy.Adapter(policy.norm_stats(store), config)
    run = lambda targets: scoring.score(
        store, sampler, adapter, targets, config.action_horizon, samples, advantage=advantage, seed=seed
    )
    recovery = run(scoring.anchors(store.select("recovery", "test"), RECOVERY_STRIDE))
    nominal = run(scoring.anchors(store.select("nominal", "test"), NOMINAL_STRIDE))
    summary = summarize(recovery, nominal)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps({"summary": summary, "recovery": recovery, "nominal": nominal}))
    return summary


def collect(root: Path) -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for path in sorted(root.glob("**/eval.json")):
        parent = path.parent.relative_to(root)
        key = parent.parent if parent.name.startswith("seed") else parent
        runs[str(key)].append(json.loads(path.read_text())["summary"])
    return runs


def report(root: Path) -> str:
    rows = [
        "| run | seeds | recovery ES | worst stage ES | retract | realign | retry | recovery MSE | nominal ES |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    cell = lambda summaries, pick: f"{np.mean([pick(s)['mean'] for s in summaries]):.3f}"
    for name, summaries in sorted(collect(root).items()):
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(len(summaries)),
                    cell(summaries, lambda s: s["energy"]["recovery"]),
                    cell(summaries, lambda s: s["energy"]["worst_stage"]),
                    *[cell(summaries, lambda s, n=n: s["energy"]["stages"][n]) for n in STAGES],
                    cell(summaries, lambda s: s["error"]["recovery"]),
                    cell(summaries, lambda s: s["energy"]["nominal"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("checkpoint")
    one.add_argument("lora", type=Path)
    one.add_argument("--root", type=Path, default=Path("data/reboot/usb_a"))
    one.add_argument("--advantage", action="store_true")
    table = sub.add_parser("report")
    table.add_argument("runs", type=Path)
    args = parser.parse_args()
    if args.command == "checkpoint":
        summary = evaluate(Store(args.root), args.lora, args.lora.parent, True if args.advantage else None)
        print(json.dumps(summary, indent=1))
    else:
        text = report(args.runs)
        (args.runs / "report.md").write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
