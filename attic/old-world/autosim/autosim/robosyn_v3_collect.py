"""Collect and audit v3 policy-induced correction shards."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from autosim.robosyn_data import audit_collection_manifests
from autosim.robosyn_mvp import MVPConfig, RoboSynMVPRunner, gpu_lock
from autosim.robosyn_v3 import BANKS, COLLECTION_MASTERS


PROFILE_MAP = {
    "targeted_camera": "targeted_camera",
    "targeted_clutter": "targeted_clutter",
    "targeted_recovery": "targeted_recovery",
    "composite_contact": "composite_hard",
}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def run_collection(
    config_path: str | Path,
    diagnostics_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = MVPConfig.load(config_path)
    diagnostics = json.loads(Path(diagnostics_path).expanduser().resolve().read_text())
    if diagnostics.get("status") != "completed":
        raise ValueError("incumbent diagnostics must complete before collection")
    allocation = diagnostics["correction_allocation"]["episode_allocation"]
    if set(allocation) != set(PROFILE_MAP) or sum(allocation.values()) != 400:
        raise ValueError(f"unexpected correction allocation: {allocation}")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "collection_progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "schema_version": 3,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "allocation": allocation,
            "shards": {},
        }
    )
    if progress.get("status") == "completed":
        return progress
    runner_dir = output_dir / "runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    runner = RoboSynMVPRunner(config, run_dir=runner_dir)
    if runner.guard.before != diagnostics.get("frozen_hashes"):
        raise RuntimeError("evaluator hashes changed after incumbent diagnostics")
    dataset_parent = (
        config.repo / "lerobot_dataset" / "autosim_v3"
    )
    manifest_paths = []
    for label, episode_count in allocation.items():
        manifest = output_dir / f"{label}_{episode_count}.json"
        log = output_dir / f"{label}_{episode_count}.log"
        if not (manifest.exists() and json.loads(manifest.read_text()).get("status") == "completed"):
            save_parent = dataset_parent / f"{label}_{episode_count}"
            command = [
                str(config.python),
                "scripts/run_env.py",
                "--gym_config",
                "configs/click_bell/random/gym_config.json",
                "--action_config",
                "configs/click_bell/action_config.json",
                "--headless",
                "--max_episodes",
                str(int(episode_count)),
                "--collection_profile",
                PROFILE_MAP[label],
                "--collection_mode",
                "policy_correction",
                "--collection_seed",
                str(COLLECTION_MASTERS[label]),
                "--dataset_save_path",
                str(save_parent),
                "--collection_manifest",
                str(manifest),
                "--collection_max_attempts",
                str(max(1_000, int(episode_count) * 20)),
                "--collection_quiet",
                "--correction_policy_config",
                "policy/act/deploy_policy.yml",
                "--correction_checkpoint",
                str(config.baseline_checkpoint),
                "--correction_device",
                "cuda",
                "--correction_prefix_min",
                "40",
                "--correction_prefix_max",
                "160",
                "--correction_replan_steps",
                "10",
                "--correction_safe_return_steps",
                "20",
            ]
            for attempt in range(3):
                attempt_log = (
                    log
                    if attempt == 0
                    else log.with_name(f"{log.stem}_attempt{attempt + 1}.log")
                )
                try:
                    runner._run_command(command, attempt_log, 43_200)
                    break
                except RuntimeError as exc:
                    retryable = "exit code -6" in str(exc) or "exit code -11" in str(exc)
                    if not retryable or attempt == 2:
                        raise
        payload = json.loads(manifest.read_text())
        if payload.get("status") != "completed":
            raise RuntimeError(f"collection shard failed: {manifest}")
        if len(payload.get("successful_episode_seeds", [])) != int(episode_count):
            raise RuntimeError(f"collection shard is incomplete: {manifest}")
        if len(payload.get("dataset_paths", [])) != 1:
            raise RuntimeError(f"expected one dataset path in {manifest}")
        dataset_path = Path(payload["dataset_paths"][0])
        qc_path = output_dir / f"{label}_{episode_count}_qc.json"
        qc_command = [
            str(config.python),
            str(Path(__file__).resolve().parent / "robosyn_data.py"),
            "qc",
            "--dataset-root",
            str(dataset_path),
            "--collection-manifest",
            str(manifest),
            "--output",
            str(qc_path),
            "--eval-master-seed",
            str(BANKS["internal_frozen_v3"]["master_seed"]),
            "--eval-episodes",
            str(BANKS["internal_frozen_v3"]["episodes"]),
        ]
        runner._run_command(
            qc_command, output_dir / f"{label}_{episode_count}_qc.log", 3_600
        )
        qc = json.loads(qc_path.read_text())
        if not qc["passed"]:
            raise RuntimeError(f"QC failed for {dataset_path}")
        progress["shards"][label] = {
            "episodes": int(episode_count),
            "manifest": str(manifest),
            "dataset_root": str(dataset_path),
            "qc": str(qc_path),
        }
        _write_json_atomic(progress_path, progress)
        manifest_paths.append(manifest)

    excluded = {
        name: (int(spec["master_seed"]), int(spec["episodes"]))
        for name, spec in BANKS.items()
    }
    seed_audit = audit_collection_manifests(
        manifest_paths,
        output=output_dir / "seed_audit.json",
        eval_master_seed=BANKS["internal_frozen_v3"]["master_seed"],
        eval_episodes=BANKS["internal_frozen_v3"]["episodes"],
        excluded_seed_banks=excluded,
    )
    if not seed_audit["passed"]:
        raise RuntimeError("v3 collection seed audit failed")
    progress.update(
        {
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "seed_audit": str(output_dir / "seed_audit.json"),
            "total_successful_episodes": sum(allocation.values()),
        }
    )
    _write_json_atomic(progress_path, progress)
    return progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = MVPConfig.load(args.config)
    with gpu_lock(config.gpu_id):
        result = run_collection(args.config, args.diagnostics, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
