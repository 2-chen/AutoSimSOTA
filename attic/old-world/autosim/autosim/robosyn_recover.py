"""Recover an interrupted single-recipe RoboSyn ablation without retraining.

The normal runner intentionally records failures and exits.  This recovery
entry point is narrower: it accepts a completed development run with exactly
one recipe, reuses valid checkpoints and baseline metrics, and fills only the
missing evaluation/training rungs.  Earlier failure records remain in the
audit trail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from autosim.robosyn_mvp import (
    MVPConfig,
    Recipe,
    RoboSynMVPRunner,
    acceptance_decision,
    gpu_lock,
    metric_order,
)


def recover_run(config: MVPConfig, run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("evaluation_protocol") != "development":
        raise ValueError("recovery is restricted to development runs")
    records = list(summary.get("records") or [])
    names = {record.get("name") for record in records if record.get("name")}
    if len(names) != 1:
        raise ValueError("recovery requires exactly one recorded recipe")
    name = next(iter(names))
    source_record = next(record for record in records if record.get("name") == name)
    recipe = Recipe(name=name, params=dict(source_record["params"]))

    runner = RoboSynMVPRunner(config, run_dir=run_dir)
    if summary.get("frozen_hashes") != runner.guard.before:
        raise RuntimeError("frozen evaluator hashes changed since the original run")
    runner.records = records
    runner.events.append("recovery_started", existing_records=len(records), recipe=name)

    rungs = config.eval_defaults["rungs"]
    seeds = config.eval_defaults["seeds"]
    training_steps = config.eval_defaults["training_steps"]
    baseline_by_rung = dict(summary.get("baseline_by_episodes") or {})
    candidate_dir = run_dir / "candidates" / name
    train_dir = candidate_dir / "train"

    for rung_index, episodes in enumerate(rungs):
        already_completed = [
            record
            for record in runner.records
            if record.get("name") == name
            and record.get("status") == "completed"
            and int(record.get("rung", -1)) == rung_index
        ]
        if already_completed:
            continue

        rung_seed = seeds[rung_index]
        baseline_key = f"rung_{rung_index}_{episodes}ep_seed_{rung_seed}"
        if baseline_key not in baseline_by_rung:
            baseline_by_rung[baseline_key] = runner.evaluate(
                config.baseline_checkpoint,
                episodes,
                "baseline",
                seed=rung_seed,
            )

        target_steps = training_steps[rung_index]
        checkpoint = train_dir / "checkpoints" / f"{target_steps:06d}" / "pretrained_model"
        if not (checkpoint / "model.safetensors").is_file():
            prior_models = list(train_dir.glob("checkpoints/*/pretrained_model/model.safetensors"))
            checkpoint = runner.train(
                recipe,
                steps=target_steps,
                resume=bool(prior_models),
            )
        metrics = runner.evaluate(
            checkpoint,
            episodes,
            name,
            seed=rung_seed,
        )
        runner._record(
            name=name,
            status="completed",
            recovered=True,
            rung=rung_index,
            episodes=episodes,
            training_steps=target_steps,
            score=metrics["score"],
            metrics=metrics,
            params=recipe.params,
            checkpoint=str(checkpoint),
        )
        runner.guard.assert_unchanged()

    final_rung = len(rungs) - 1
    finalists = [
        record
        for record in runner.records
        if record.get("status") == "completed"
        and int(record.get("rung", -1)) == final_rung
    ]
    if not finalists:
        raise RuntimeError("recovery did not produce the final confirmation rung")
    best = max(finalists, key=lambda record: metric_order(record["metrics"]))
    final_key = f"rung_{final_rung}_{rungs[final_rung]}ep_seed_{seeds[final_rung]}"
    decision = acceptance_decision(
        best["metrics"], baseline_by_rung[final_key], config.eval_defaults
    )
    result = dict(summary)
    result.update(
        {
            "status": "completed",
            "baseline_by_episodes": baseline_by_rung,
            "best": best,
            "accepted": bool(decision["accepted"]),
            "acceptance_decision": decision,
            "records": runner.records,
            "recovered": True,
        }
    )
    runner._write_summary(result)
    runner.events.append("recovery_finished", summary=str(summary_path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    config = MVPConfig.load(args.config)
    with gpu_lock(config.gpu_id):
        result = recover_run(config, args.run_dir)
    best = result["best"]
    print(
        f"Recovered {args.run_dir}: rung={best['rung']} "
        f"score={best['metrics']['success_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
