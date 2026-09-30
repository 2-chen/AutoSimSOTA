"""Train the selected RoboSyn candidate and enforce one frozen evaluation query."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from autosim.robosyn_mvp import (
    MVPConfig,
    Recipe,
    RoboSynMVPRunner,
    acceptance_decision,
    gpu_lock,
    paired_success_comparison,
)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_selected(
    config: MVPConfig,
    selection_path: str | Path,
    *,
    target_steps: int = 80_000,
) -> Dict[str, Any]:
    selection_path = Path(selection_path).expanduser().resolve()
    selection = json.loads(selection_path.read_text())
    if selection.get("frozen_evaluation_performed") is not False:
        raise ValueError("selection manifest is not pre-frozen-evaluation")
    selected = selection["selected"]
    if selected.get("experiment") != config.name:
        raise ValueError("selected experiment does not match final training config")
    run_dir = Path(selected["run_dir"]).resolve()
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("frozen_hashes") is None:
        raise ValueError("selected run has no frozen evaluator hashes")
    record = summary["best"]
    if int(record["training_steps"]) >= int(target_steps):
        raise ValueError("selected checkpoint already meets or exceeds target steps")
    recipe_params = dict(record["params"])
    recipe_params["save_freq"] = 5_000
    recipe = Recipe(name=record["name"], params=recipe_params)
    runner = RoboSynMVPRunner(config, run_dir=run_dir)
    if runner.guard.before != summary["frozen_hashes"]:
        raise RuntimeError("frozen evaluator hashes changed after development selection")
    runner.events.append(
        "final_training_started",
        selection=str(selection_path),
        source_checkpoint=record["checkpoint"],
        target_steps=target_steps,
    )
    checkpoint = runner.train(recipe, steps=target_steps, resume=True)
    model_path = checkpoint / "model.safetensors"
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "final_training_completed",
        "selection_path": str(selection_path),
        "selected_experiment": config.name,
        "source_checkpoint": record["checkpoint"],
        "source_training_steps": int(record["training_steps"]),
        "target_training_steps": int(target_steps),
        "checkpoint": str(checkpoint),
        "model_sha256": _sha256(model_path),
        "frozen_hashes": runner.guard.before,
        "frozen_evaluation_performed": False,
    }
    manifest_path = selection_path.parent / "final_training_v2.json"
    _write_json_atomic(manifest_path, result)
    runner.events.append(
        "final_training_finished", checkpoint=str(checkpoint), manifest=str(manifest_path)
    )
    return result


def _load_metrics(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    metrics = dict(payload["summary"])
    metrics["episodes"] = list(payload.get("episodes") or [])
    metrics["evaluation_config"] = dict(payload.get("config") or {})
    metrics["score"] = float(metrics["success_rate"])
    metrics["metrics_path"] = str(Path(path).expanduser().resolve())
    return metrics


def evaluate_frozen_once(
    config: MVPConfig,
    training_manifest_path: str | Path,
    official_metrics_path: str | Path,
    targeted_metrics_path: str | Path,
) -> Dict[str, Any]:
    training_manifest_path = Path(training_manifest_path).expanduser().resolve()
    state_path = training_manifest_path.parent / "frozen_evaluation_state_v2.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        raise RuntimeError(
            "frozen evaluation is already started or completed: "
            f"{state.get('status')} at {state_path}"
        )
    training = json.loads(training_manifest_path.read_text())
    if training.get("status") != "final_training_completed":
        raise ValueError("final training is not completed")
    checkpoint = Path(training["checkpoint"]).resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"incomplete final checkpoint: {checkpoint}")

    config.evaluation.update(
        {
            "protocol": "frozen_final",
            "rungs": [100],
            "training_steps": [int(training["target_training_steps"])],
            "seeds": [0],
        }
    )
    final_dir = training_manifest_path.parent / "frozen_final_v2"
    final_dir.mkdir(parents=True, exist_ok=True)
    runner = RoboSynMVPRunner(config, run_dir=final_dir)
    if runner.guard.before != training["frozen_hashes"]:
        raise RuntimeError("frozen evaluator hashes changed before final evaluation")
    started = {
        "schema_version": 1,
        "status": "started",
        "query_count": 1,
        "started_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": _sha256(checkpoint / "model.safetensors"),
        "master_seed": 0,
        "episodes": 100,
    }
    _write_json_atomic(state_path, started)
    runner.events.append("frozen_evaluation_started", **started)
    candidate = runner.evaluate(checkpoint, 100, "data_v2_final", seed=0)
    runner.guard.assert_unchanged()
    official = _load_metrics(official_metrics_path)
    targeted = _load_metrics(targeted_metrics_path)
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "frozen_evaluation_completed",
        "ranking_eligible": True,
        "query_count": 1,
        "master_seed": 0,
        "episodes": 100,
        "checkpoint": str(checkpoint),
        "candidate": candidate,
        "official80": official,
        "targeted80": targeted,
        "candidate_vs_official80": paired_success_comparison(candidate, official),
        "candidate_vs_targeted80": paired_success_comparison(candidate, targeted),
        "acceptance_vs_targeted80": acceptance_decision(
            candidate, targeted, config.eval_defaults
        ),
        "frozen_hashes": runner.guard.before,
    }
    summary_path = final_dir / "summary.json"
    _write_json_atomic(summary_path, result)
    _write_json_atomic(
        state_path,
        {
            **started,
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "summary": str(summary_path),
            "success_rate": candidate["success_rate"],
        },
    )
    runner.events.append("frozen_evaluation_finished", summary=str(summary_path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("config", type=Path)
    train_parser.add_argument("selection", type=Path)
    train_parser.add_argument("--target-steps", type=int, default=80_000)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("config", type=Path)
    eval_parser.add_argument("training_manifest", type=Path)
    eval_parser.add_argument("official_metrics", type=Path)
    eval_parser.add_argument("targeted_metrics", type=Path)
    args = parser.parse_args()
    config = MVPConfig.load(args.config)
    with gpu_lock(config.gpu_id):
        if args.mode == "train":
            result = train_selected(
                config, args.selection, target_steps=args.target_steps
            )
            print(f"Final checkpoint: {result['checkpoint']}")
        else:
            result = evaluate_frozen_once(
                config,
                args.training_manifest,
                args.official_metrics,
                args.targeted_metrics,
            )
            print(f"Frozen success rate: {result['candidate']['success_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
