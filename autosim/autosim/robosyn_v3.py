"""Protocol and analysis utilities for the third RoboSyn AutoResearch round."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from autosim.robosyn_data import evaluation_seed_bank
from autosim.robosyn_mvp import (
    MVPConfig,
    RoboSynMVPRunner,
    gpu_lock,
    paired_success_comparison,
)


BANKS = {
    "development_a": {"master_seed": 1_031_001, "episodes": 40},
    "development_b": {"master_seed": 1_032_001, "episodes": 40},
    "confirmation": {"master_seed": 1_033_001, "episodes": 100},
    "diagnostic_appearance": {"master_seed": 1_034_101, "episodes": 30},
    "diagnostic_camera": {"master_seed": 1_034_201, "episodes": 30},
    "diagnostic_robot_pose": {"master_seed": 1_034_301, "episodes": 30},
    "diagnostic_clutter": {"master_seed": 1_034_401, "episodes": 30},
    "diagnostic_contact": {"master_seed": 1_034_501, "episodes": 30},
    "internal_frozen_v3": {"master_seed": 1_039_001, "episodes": 200},
}

COLLECTION_MASTERS = {
    "targeted_camera": 2_031_001,
    "targeted_clutter": 2_032_001,
    "targeted_recovery": 2_033_001,
    "composite_contact": 2_034_001,
}

DEFAULT_FROZEN_FILES = [
    "scripts/eval_policy.py",
    "policy/act/deploy_policy.yml",
    "policy/act/deploy_policy.py",
    "policy/inference_timing.py",
    "robosynchallenge/tasks/click_bell/click_bell.py",
    "configs/click_bell/random/gym_config.json",
    "configs/click_bell/action_config.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def create_protocol(repo: str | Path, output: str | Path) -> dict[str, Any]:
    repo = Path(repo).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    banks = {}
    seen: dict[int, str] = {}
    overlaps = []
    for name, spec in BANKS.items():
        seeds = evaluation_seed_bank(spec["master_seed"], spec["episodes"])
        for seed in seeds:
            if seed in seen:
                overlaps.append({"seed": seed, "first": seen[seed], "second": name})
            seen[seed] = name
        banks[name] = {
            **spec,
            "episode_seed_sha256": hashlib.sha256(
                json.dumps(seeds, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    if overlaps:
        raise RuntimeError(f"v3 evaluation seed banks overlap: {overlaps[:3]}")
    frozen_hashes = {
        name: _sha256(repo / name) for name in DEFAULT_FROZEN_FILES
    }
    payload = {
        "schema_version": 3,
        "kind": "robosyn_autoresearch_protocol",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "locked",
        "policy_observation_contract": "14D joint state + 3 RGB cameras",
        "old_frozen_seed_zero_status": "retired_postmortem_only",
        "banks": banks,
        "collection_master_seeds": COLLECTION_MASTERS,
        "seed_bank_overlaps": overlaps,
        "development_process_replicates": 2,
        "acceptance": {
            "baseline": "v2_stratified_80k_incumbent",
            "minimum_internal_frozen_episodes": 200,
            "minimum_success_rate_delta": 0.05,
            "maximum_one_sided_sign_test_p_value": 0.05,
            "maximum_average_action_steps_ratio": 1.10,
            "maximum_factor_slice_drop": 0.05,
        },
        "targeted_episode_fraction_cap": 0.60,
        "frozen_files": frozen_hashes,
        "frozen_evaluation_performed": False,
    }
    _write_json_atomic(output, payload)
    return payload


def load_metrics(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    metrics = dict(payload["summary"])
    metrics["episodes"] = list(payload.get("episodes") or [])
    metrics["config"] = dict(payload.get("config") or {})
    metrics["metrics_path"] = str(path)
    return metrics


def aggregate_replicates(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(metrics)
    if not items:
        raise ValueError("at least one metrics object is required")
    episode_counts = {int(item["episode_count"]) for item in items}
    seed_lists = {
        tuple(int(row["episode_seed"]) for row in item.get("episodes", []))
        for item in items
    }
    if len(episode_counts) != 1 or len(seed_lists) != 1:
        raise ValueError("replicates must have identical episode and seed coverage")
    rates = [float(item["success_rate"]) for item in items]
    steps = [float(item["average_action_steps"]) for item in items]
    return {
        "replicate_count": len(items),
        "episodes_per_replicate": next(iter(episode_counts)),
        "mean_success_rate": sum(rates) / len(rates),
        "minimum_success_rate": min(rates),
        "maximum_success_rate": max(rates),
        "success_rates": rates,
        "mean_average_action_steps": sum(steps) / len(steps),
        "metrics_paths": [item.get("metrics_path") for item in items],
    }


def failure_stage_report(metrics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(metrics)
    counter: Counter[str] = Counter()
    depths = []
    for item in items:
        for episode in item.get("episodes", []):
            counter[str(episode.get("failure_stage", "unclassified"))] += 1
            depth = episode.get("max_button_press_depth_m")
            if depth is not None:
                depths.append(float(depth))
    total = sum(counter.values())
    return {
        "episode_count": total,
        "stage_counts": dict(sorted(counter.items())),
        "stage_fractions": {
            key: value / total for key, value in sorted(counter.items())
        } if total else {},
        "mean_max_button_press_depth_m": sum(depths) / len(depths) if depths else None,
    }


def _integer_allocation(weights: dict[str, float], total: int) -> dict[str, int]:
    normalized = {key: max(float(value), 0.0) for key, value in weights.items()}
    mass = sum(normalized.values())
    if mass <= 0:
        normalized = {key: 1.0 for key in normalized}
        mass = len(normalized)
    raw = {key: total * value / mass for key, value in normalized.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(raw, key=lambda key: (raw[key] - result[key], key), reverse=True)
    for key in order[:remaining]:
        result[key] += 1
    return result


def correction_allocation(
    diagnostic_success_rates: dict[str, float], *, total: int = 400
) -> dict[str, Any]:
    required = {"appearance", "camera", "robot_pose", "clutter", "contact"}
    missing = required - set(diagnostic_success_rates)
    if missing:
        raise ValueError(f"missing diagnostic factors: {sorted(missing)}")
    # A non-zero floor retains coverage even when a small diagnostic happens to
    # score well under process noise. Failure deficit supplies the remaining mass.
    weights = {
        "targeted_camera": 0.05
        + (
            max(0.0, 1.0 - float(diagnostic_success_rates["appearance"]))
            + max(0.0, 1.0 - float(diagnostic_success_rates["camera"]))
        )
        / 2,
        "targeted_recovery": 0.05
        + max(0.0, 1.0 - float(diagnostic_success_rates["robot_pose"])),
        "targeted_clutter": 0.05
        + max(0.0, 1.0 - float(diagnostic_success_rates["clutter"])),
        "composite_contact": 0.05
        + max(0.0, 1.0 - float(diagnostic_success_rates["contact"])),
    }
    allocation = _integer_allocation(weights, int(total))
    return {
        "total": int(total),
        "diagnostic_success_rates": diagnostic_success_rates,
        "weights": weights,
        "episode_allocation": allocation,
    }


def _assert_protocol_hashes(config: MVPConfig, protocol: dict[str, Any]) -> None:
    runner = RoboSynMVPRunner(config, dry_run=True)
    if runner.guard.before != protocol["frozen_files"]:
        raise RuntimeError("current evaluator hashes differ from locked v3 protocol")


def run_incumbent_diagnostics(
    config_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = MVPConfig.load(config_path)
    protocol = json.loads(Path(protocol_path).expanduser().resolve().read_text())
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "incumbent_diagnostics_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("status") == "completed":
            return progress
    else:
        progress = {
            "schema_version": 3,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "checkpoint": str(config.baseline_checkpoint),
            "evaluations": {},
        }
        _write_json_atomic(progress_path, progress)
    runner_dir = output_dir / "runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    runner = RoboSynMVPRunner(config, run_dir=runner_dir)
    if runner.guard.before != protocol["frozen_files"]:
        raise RuntimeError("current evaluator hashes differ from locked v3 protocol")

    def evaluate_one(key, bank_name, diagnostic_profile=None):
        if key in progress["evaluations"]:
            return load_metrics(progress["evaluations"][key]["metrics_path"])
        bank = protocol["banks"][bank_name]
        result = runner.evaluate(
            config.baseline_checkpoint,
            int(bank["episodes"]),
            key,
            diagnostic_profile=diagnostic_profile,
            seed=int(bank["master_seed"]),
        )
        progress["evaluations"][key] = result
        _write_json_atomic(progress_path, progress)
        runner.guard.assert_unchanged()
        return result

    development = {}
    development_items = []
    for bank_name in ("development_a", "development_b"):
        replicates = [
            evaluate_one(f"incumbent_{bank_name}_rep{replicate}", bank_name)
            for replicate in (1, 2)
        ]
        development[bank_name] = aggregate_replicates(replicates)
        development_items.extend(replicates)

    diagnostics = {}
    for factor in ("appearance", "camera", "robot_pose", "clutter"):
        diagnostics[factor] = evaluate_one(
            f"incumbent_diagnostic_{factor}",
            f"diagnostic_{factor}",
            diagnostic_profile=factor,
        )
    diagnostics["contact"] = evaluate_one(
        "incumbent_diagnostic_contact", "diagnostic_contact"
    )
    rates = {
        factor: float(metrics["success_rate"])
        for factor, metrics in diagnostics.items()
    }
    progress.update(
        {
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "development": development,
            "development_failure_stages": failure_stage_report(development_items),
            "diagnostic_success_rates": rates,
            "correction_allocation": correction_allocation(rates),
            "frozen_hashes": runner.guard.before,
            "old_frozen_seed_zero_queried": False,
            "internal_frozen_v3_queried": False,
        }
    )
    _write_json_atomic(progress_path, progress)
    return progress


def recompute_correction_allocation(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("status") != "completed":
        raise ValueError("diagnostics must be completed before allocation")
    payload["correction_allocation"] = correction_allocation(
        payload["diagnostic_success_rates"]
    )
    _write_json_atomic(path, payload)
    return payload["correction_allocation"]


def evaluate_internal_frozen_once(
    config_path: str | Path,
    protocol_path: str | Path,
    checkpoint: str | Path,
    incumbent_checkpoint: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = MVPConfig.load(config_path)
    protocol = json.loads(Path(protocol_path).expanduser().resolve().read_text())
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "internal_frozen_v3_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        raise RuntimeError(
            "internal frozen-v3 is already started or completed: "
            f"{state.get('status')} at {state_path}"
        )
    checkpoint = Path(checkpoint).expanduser().resolve()
    incumbent_checkpoint = Path(incumbent_checkpoint).expanduser().resolve()
    for label, path in (("candidate", checkpoint), ("incumbent", incumbent_checkpoint)):
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"missing {label} checkpoint: {path}")
    runner_dir = output_dir / "runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    runner = RoboSynMVPRunner(config, run_dir=runner_dir)
    if runner.guard.before != protocol["frozen_files"]:
        raise RuntimeError("current evaluator hashes differ from locked v3 protocol")
    bank = protocol["banks"]["internal_frozen_v3"]
    started = {
        "schema_version": 3,
        "status": "started",
        "started_at": datetime.now().astimezone().isoformat(),
        "bank_open_count": 1,
        "planned_policy_query_count": 2,
        "completed_policy_query_count": 0,
        "master_seed": int(bank["master_seed"]),
        "episodes_per_policy": int(bank["episodes"]),
        "candidate_checkpoint": str(checkpoint),
        "candidate_model_sha256": _sha256(checkpoint / "model.safetensors"),
        "incumbent_checkpoint": str(incumbent_checkpoint),
        "incumbent_model_sha256": _sha256(
            incumbent_checkpoint / "model.safetensors"
        ),
    }
    _write_json_atomic(state_path, started)
    incumbent = runner.evaluate(
        incumbent_checkpoint,
        int(bank["episodes"]),
        "internal_frozen_v3_incumbent",
        seed=int(bank["master_seed"]),
    )
    _write_json_atomic(
        state_path, {**started, "completed_policy_query_count": 1}
    )
    candidate = runner.evaluate(
        checkpoint,
        int(bank["episodes"]),
        "internal_frozen_v3_candidate",
        seed=int(bank["master_seed"]),
    )
    runner.guard.assert_unchanged()
    comparison = paired_success_comparison(candidate, incumbent)
    gate = protocol["acceptance"]
    speed_ratio = (
        float(candidate["average_action_steps"])
        / float(incumbent["average_action_steps"])
    )
    reasons = []
    if int(candidate["episode_count"]) < int(gate["minimum_internal_frozen_episodes"]):
        reasons.append("insufficient frozen episodes")
    if comparison["success_rate_delta"] < float(gate["minimum_success_rate_delta"]):
        reasons.append("success-rate delta below gate")
    if comparison["one_sided_sign_test_p_value"] > float(
        gate["maximum_one_sided_sign_test_p_value"]
    ):
        reasons.append("paired sign-test p-value above gate")
    if speed_ratio > float(gate["maximum_average_action_steps_ratio"]):
        reasons.append("average action steps regressed beyond gate")
    result = {
        "schema_version": 3,
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(),
        "bank_open_count": 1,
        "policy_query_count": 2,
        "master_seed": int(bank["master_seed"]),
        "episodes_per_policy": int(bank["episodes"]),
        "candidate": candidate,
        "incumbent": incumbent,
        "candidate_vs_incumbent": comparison,
        "average_action_steps_ratio": speed_ratio,
        "accepted": not reasons,
        "reasons": reasons,
        "frozen_hashes": runner.guard.before,
        "old_frozen_seed_zero_queried": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json_atomic(summary_path, result)
    _write_json_atomic(
        state_path,
        {
            **started,
            "status": "completed",
            "completed_at": result["completed_at"],
            "completed_policy_query_count": 2,
            "summary": str(summary_path),
            "accepted": result["accepted"],
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    protocol = subparsers.add_parser("protocol")
    protocol.add_argument("--repo", required=True, type=Path)
    protocol.add_argument("--output", required=True, type=Path)
    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("metrics", nargs="+", type=Path)
    diagnose.add_argument("--output", required=True, type=Path)
    incumbent = subparsers.add_parser("incumbent")
    incumbent.add_argument("--config", required=True, type=Path)
    incumbent.add_argument("--protocol", required=True, type=Path)
    incumbent.add_argument("--output-dir", required=True, type=Path)
    frozen = subparsers.add_parser("frozen")
    frozen.add_argument("--config", required=True, type=Path)
    frozen.add_argument("--protocol", required=True, type=Path)
    frozen.add_argument("--checkpoint", required=True, type=Path)
    frozen.add_argument("--incumbent-checkpoint", required=True, type=Path)
    frozen.add_argument("--output-dir", required=True, type=Path)
    reallocate = subparsers.add_parser("reallocate")
    reallocate.add_argument("--diagnostics", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "protocol":
        payload = create_protocol(args.repo, args.output)
    elif args.mode == "diagnose":
        payload = failure_stage_report(load_metrics(path) for path in args.metrics)
        _write_json_atomic(args.output.resolve(), payload)
    elif args.mode == "incumbent":
        with gpu_lock(MVPConfig.load(args.config).gpu_id):
            payload = run_incumbent_diagnostics(
                args.config, args.protocol, args.output_dir
            )
    elif args.mode == "frozen":
        with gpu_lock(MVPConfig.load(args.config).gpu_id):
            payload = evaluate_internal_frozen_once(
                args.config,
                args.protocol,
                args.checkpoint,
                args.incumbent_checkpoint,
                args.output_dir,
            )
    else:
        payload = recompute_correction_allocation(args.diagnostics)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
