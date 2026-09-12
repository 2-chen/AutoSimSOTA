"""Resumable single-GPU successive-halving experiments for RoboSyn v3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from autosim.robosyn_mvp import MVPConfig, Recipe, RoboSynMVPRunner, gpu_lock
from autosim.robosyn_v3 import aggregate_replicates, load_metrics


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _checkpoint(run_dir: Path, steps: int) -> Path:
    return (
        run_dir
        / "candidates/policy/train/checkpoints"
        / f"{steps:06d}"
        / "pretrained_model"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_order(candidate: dict[str, Any]) -> tuple[Any, ...]:
    development = candidate["development"]
    aggregates = [development["development_a"], development["development_b"]]
    rates = [rate for aggregate in aggregates for rate in aggregate["success_rates"]]
    return (
        sum(rates) / len(rates),
        min(rates),
        -sum(item["mean_average_action_steps"] for item in aggregates) / 2,
        candidate["name"],
    )


def run_processing_ablation(
    config_path: str | Path,
    protocol_path: str | Path,
    specification_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    base_config = MVPConfig.load(config_path)
    protocol = json.loads(Path(protocol_path).expanduser().resolve().read_text())
    specification = json.loads(
        Path(specification_path).expanduser().resolve().read_text()
    )
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "processing_ablation_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("status") == "completed":
            return progress
    else:
        progress = {
            "schema_version": 3,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "candidates": {},
            "confirmation": {},
            "internal_frozen_v3_queried": False,
        }
        _write_json_atomic(progress_path, progress)

    runners = {}
    recipes = {}
    for candidate_spec in specification["candidates"]:
        name = str(candidate_spec["name"])
        config = copy.deepcopy(base_config)
        config.name = name
        config.dataset_mixture_manifest = Path(
            candidate_spec["dataset_mixture_manifest"]
        ).expanduser().resolve()
        run_dir = output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        runner = RoboSynMVPRunner(config, run_dir=run_dir)
        if runner.guard.before != protocol["frozen_files"]:
            raise RuntimeError(f"{name}: evaluator hashes differ from v3 protocol")
        params = dict(config.train_defaults)
        params.update(candidate_spec.get("train_overrides") or {})
        params["steps"] = 20_000
        recipe = Recipe(name="policy", params=params)
        runners[name] = runner
        recipes[name] = recipe
        record = progress["candidates"].setdefault(
            name,
            {
                "name": name,
                "hypothesis": candidate_spec.get("hypothesis"),
                "dataset_mixture_manifest": str(config.dataset_mixture_manifest),
                "train_params": params,
                "training": {},
                "evaluations": {},
            },
        )
        checkpoint_5k = _checkpoint(run_dir, 5_000)
        if not (checkpoint_5k / "model.safetensors").is_file():
            checkpoint_5k = runner.train(recipe, steps=5_000, resume=False)
        record["training"]["5000"] = {
            "checkpoint": str(checkpoint_5k),
            "mechanical_check": "passed",
            "ranking_score_used": False,
        }
        _write_json_atomic(progress_path, progress)
        checkpoint_20k = _checkpoint(run_dir, 20_000)
        if not (checkpoint_20k / "model.safetensors").is_file():
            checkpoint_20k = runner.train(recipe, steps=20_000, resume=True)
        record["training"]["20000"] = {"checkpoint": str(checkpoint_20k)}
        _write_json_atomic(progress_path, progress)

        development = {}
        for bank_name in ("development_a", "development_b"):
            bank = protocol["banks"][bank_name]
            replicates = []
            for replicate in (1, 2):
                key = f"{bank_name}_rep{replicate}"
                if key in record["evaluations"]:
                    result = load_metrics(record["evaluations"][key]["metrics_path"])
                else:
                    result = runner.evaluate(
                        checkpoint_20k,
                        int(bank["episodes"]),
                        f"{name}_{key}",
                        seed=int(bank["master_seed"]),
                    )
                    record["evaluations"][key] = result
                    _write_json_atomic(progress_path, progress)
                replicates.append(result)
            development[bank_name] = aggregate_replicates(replicates)
        record["development"] = development
        runner.guard.assert_unchanged()
        _write_json_atomic(progress_path, progress)

    ranked = sorted(
        progress["candidates"].values(), key=_candidate_order, reverse=True
    )
    top_two = [item["name"] for item in ranked[:2]]
    progress["development_ranking"] = [
        {
            "name": item["name"],
            "order": _candidate_order(item)[:3],
        }
        for item in ranked
    ]
    progress["promoted_to_40000"] = top_two
    _write_json_atomic(progress_path, progress)

    bank = protocol["banks"]["confirmation"]
    for name in top_two:
        runner = runners[name]
        record = progress["candidates"][name]
        checkpoint_40k = _checkpoint(output_dir / name, 40_000)
        if not (checkpoint_40k / "model.safetensors").is_file():
            checkpoint_40k = runner.train(recipes[name], steps=40_000, resume=True)
        record["training"]["40000"] = {"checkpoint": str(checkpoint_40k)}
        if name in progress["confirmation"]:
            confirmation = load_metrics(
                progress["confirmation"][name]["metrics_path"]
            )
        else:
            confirmation = runner.evaluate(
                checkpoint_40k,
                int(bank["episodes"]),
                f"{name}_confirmation",
                seed=int(bank["master_seed"]),
            )
            progress["confirmation"][name] = confirmation
        _write_json_atomic(progress_path, progress)

    def confirmation_order(name):
        metrics = progress["confirmation"][name]
        return (
            float(metrics["success_rate"]),
            -float(metrics["average_action_steps"]),
            name,
        )

    selected = max(top_two, key=confirmation_order)
    progress.update(
        {
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "selected": {
                "name": selected,
                "checkpoint": progress["candidates"][selected]["training"]["40000"][
                    "checkpoint"
                ],
                "confirmation": progress["confirmation"][selected],
            },
            "selection_rule": [
                "higher independent confirmation success rate",
                "lower confirmation average action steps",
                "lexical name tie-break",
            ],
            "frozen_evaluation_performed": False,
        }
    )
    _write_json_atomic(progress_path, progress)
    return progress


def train_and_validate_selected(
    config_path: str | Path,
    protocol_path: str | Path,
    specification_path: str | Path,
    selection_path: str | Path,
    incumbent_diagnostics_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = MVPConfig.load(config_path)
    protocol = json.loads(Path(protocol_path).expanduser().resolve().read_text())
    specification = json.loads(Path(specification_path).expanduser().resolve().read_text())
    selection_path = Path(selection_path).expanduser().resolve()
    selection = json.loads(selection_path.read_text())
    incumbent_diagnostics = json.loads(
        Path(incumbent_diagnostics_path).expanduser().resolve().read_text()
    )
    if selection.get("status") != "completed" or selection.get(
        "frozen_evaluation_performed"
    ) is not False:
        raise ValueError("selection must be completed without a frozen query")
    selected_name = selection["selected"]["name"]
    candidate_spec = next(
        item for item in specification["candidates"] if item["name"] == selected_name
    )
    record = selection["candidates"][selected_name]
    config.name = selected_name
    config.dataset_mixture_manifest = Path(
        candidate_spec["dataset_mixture_manifest"]
    ).expanduser().resolve()
    params = dict(config.train_defaults)
    params.update(candidate_spec.get("train_overrides") or {})
    params["steps"] = 80_000
    params["save_freq"] = 5_000
    recipe = Recipe(name="policy", params=params)
    run_dir = selection_path.parent / selected_name
    runner = RoboSynMVPRunner(config, run_dir=run_dir)
    if runner.guard.before != protocol["frozen_files"]:
        raise RuntimeError("evaluator hashes changed before final training")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "final_training_validation.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "schema_version": 3,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "selected": selected_name,
            "source_checkpoint": record["training"]["40000"]["checkpoint"],
            "evaluations": {},
            "internal_frozen_v3_queried": False,
        }
    )
    if progress.get("status") == "completed":
        return progress
    checkpoint_80k = _checkpoint(run_dir, 80_000)
    if not (checkpoint_80k / "model.safetensors").is_file():
        checkpoint_80k = runner.train(recipe, steps=80_000, resume=True)
    progress["checkpoint"] = str(checkpoint_80k)
    progress["model_sha256"] = _sha256(checkpoint_80k / "model.safetensors")
    progress["train_params"] = params
    _write_json_atomic(progress_path, progress)

    def evaluate_one(key, bank_name, checkpoint, diagnostic_profile=None):
        if key in progress["evaluations"]:
            return load_metrics(progress["evaluations"][key]["metrics_path"])
        bank = protocol["banks"][bank_name]
        result = runner.evaluate(
            checkpoint,
            int(bank["episodes"]),
            key,
            diagnostic_profile=diagnostic_profile,
            seed=int(bank["master_seed"]),
        )
        progress["evaluations"][key] = result
        _write_json_atomic(progress_path, progress)
        return result

    incumbent_checkpoint = config.baseline_checkpoint
    incumbent_confirmation = evaluate_one(
        "incumbent_confirmation", "confirmation", incumbent_checkpoint
    )
    candidate_confirmation = evaluate_one(
        "candidate_confirmation", "confirmation", checkpoint_80k
    )
    factor_results = {}
    factor_drops = {}
    for factor in ("appearance", "camera", "robot_pose", "clutter"):
        candidate = evaluate_one(
            f"candidate_diagnostic_{factor}",
            f"diagnostic_{factor}",
            checkpoint_80k,
            diagnostic_profile=factor,
        )
        incumbent = incumbent_diagnostics["evaluations"][
            f"incumbent_diagnostic_{factor}"
        ]
        factor_results[factor] = candidate
        factor_drops[factor] = float(candidate["success_rate"]) - float(
            incumbent["success_rate"]
        )
    candidate_contact = evaluate_one(
        "candidate_diagnostic_contact", "diagnostic_contact", checkpoint_80k
    )
    incumbent_contact = incumbent_diagnostics["evaluations"][
        "incumbent_diagnostic_contact"
    ]
    factor_results["contact"] = candidate_contact
    factor_drops["contact"] = float(candidate_contact["success_rate"]) - float(
        incumbent_contact["success_rate"]
    )
    max_drop = float(protocol["acceptance"]["maximum_factor_slice_drop"])
    reasons = [
        f"{factor} slice dropped by {delta:.4f}"
        for factor, delta in factor_drops.items()
        if delta < -max_drop
    ]
    confirmation_delta = float(candidate_confirmation["success_rate"]) - float(
        incumbent_confirmation["success_rate"]
    )
    if confirmation_delta < -max_drop:
        reasons.append(
            f"confirmation success rate dropped by {confirmation_delta:.4f}"
        )
    runner.guard.assert_unchanged()
    progress.update(
        {
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "candidate_confirmation": candidate_confirmation,
            "incumbent_confirmation": incumbent_confirmation,
            "confirmation_delta": confirmation_delta,
            "factor_results": factor_results,
            "factor_deltas": factor_drops,
            "ready_for_internal_frozen_v3": not reasons,
            "pre_frozen_gate_reasons": reasons,
            "frozen_hashes": runner.guard.before,
        }
    )
    _write_json_atomic(progress_path, progress)
    return progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["ablation", "final"], default="ablation"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--specification", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--incumbent-diagnostics", type=Path)
    args = parser.parse_args()
    config = MVPConfig.load(args.config)
    with gpu_lock(config.gpu_id):
        if args.mode == "ablation":
            result = run_processing_ablation(
                args.config, args.protocol, args.specification, args.output_dir
            )
        else:
            if args.selection is None or args.incumbent_diagnostics is None:
                parser.error("--mode final requires --selection and --incumbent-diagnostics")
            result = train_and_validate_selected(
                args.config,
                args.protocol,
                args.specification,
                args.selection,
                args.incumbent_diagnostics,
                args.output_dir,
            )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
