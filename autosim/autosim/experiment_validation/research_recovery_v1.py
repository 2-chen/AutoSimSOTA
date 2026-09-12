"""Versioned recovery for a research task rejected by full video decoding.

The failed experiment remains immutable.  A fresh experiment root reuses only
the already-audited official-data checkpoint and its exact fixed-bank
development evaluations.  New collection uses the original master seeds and a
strict full-decode gate before any trajectory can enter training.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import time
from pathlib import Path

from autosim.robosyn_data import evaluation_seed_bank
from autosim.research.common import atomic_json, digest, freeze_files, now, read_json
from autosim.research.controller import ResearchConfig, ResearchController
from autosim.research.runtime import Runtime

from .checkpoint_audit import audit_checkpoint
from .dataset_video_integrity_v1 import audit_dataset
from .queue_manifest import versioned_manifest
from .research_comparison import write_report


TASK = "water_pouring"


def _copy_json(source: Path, destination: Path) -> None:
    atomic_json(destination, read_json(source))


def _backup_ledger(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)


def prepare_experiment_root(prior: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("task_inventory.json", "asset_status.json", "smoke_status.json",
                 "official_policy_smoke_status.json"):
        source = prior / name
        if source.is_file():
            destination = output / name
            if destination.exists() and read_json(destination) != read_json(source):
                raise RuntimeError(f"recovery input changed: {name}")
            _copy_json(source, destination)
    _backup_ledger(prior / "seeds.sqlite", output / "seeds.sqlite")
    source_versions, destination_versions = prior / "data_versions", output / "data_versions"
    if source_versions.is_dir() and not destination_versions.exists():
        shutil.copytree(source_versions, destination_versions)


def verify_reused_evaluation(source: Path, checkpoint: Path, *, task: str,
                             episodes: int, master_seed: int, purpose: str) -> dict:
    metrics = read_json(source / "evaluation_metrics.json")
    request = read_json(source / "evaluation_request.json")
    rows = metrics.get("episodes", [])
    expected_seeds = evaluation_seed_bank(master_seed, episodes)
    checks = {
        "real_simulation": metrics.get("execution_mode") == "real_simulation",
        "purpose": metrics.get("purpose") == purpose,
        "task": metrics.get("config", {}).get("task") == task,
        "episode_count": len(rows) == episodes,
        "ordered_seed_bank": [int(row.get("episode_seed", -1)) for row in rows] == expected_seeds,
        "checkpoint_path": Path(request.get("checkpoint", "")).resolve() == checkpoint.resolve(),
        "checkpoint_hash": request.get("weight_sha256") == digest(checkpoint / "model.safetensors"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"reused evaluation contract failed: {checks}")
    return {"checks": checks, "metrics": metrics,
            "metrics_sha256": digest(source / "evaluation_metrics.json"),
            "request_sha256": digest(source / "evaluation_request.json")}


class FullDecodeRecoveryRuntime(Runtime):
    def __init__(self, *args, prior_task: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_task = prior_task
        self.reused_baseline = (prior_task /
            "controlled_baseline/train/checkpoints/080000/pretrained_model")

    @staticmethod
    def _video_keys(spec) -> list[str]:
        return [f"observation.images.{camera}" for camera in spec.cameras]

    def prepare_data(self, spec, root: Path, output: Path) -> dict:
        result = super().prepare_data(spec, root, output)
        info = read_json(root / "meta/info.json")
        video = audit_dataset(root, expected_episodes=int(info["total_episodes"]),
                              expected_video_keys=self._video_keys(spec), workers=8)
        atomic_json(output / "video_integrity_v1.json", video)
        if not video["passed"]:
            raise RuntimeError(f"full video decode admission rejected: {output / 'video_integrity_v1.json'}")
        result["full_video_decode_admission"] = str(output / "video_integrity_v1.json")
        atomic_json(output / "data_audit.json", result)
        return result

    def collect(self, spec, destination: Path, *, episodes: int, master_seed: int,
                profile: str = "full_random", timeout=1800) -> Path:
        attempts = []
        for index in range(1, 4):
            attempt = destination / f"serialization_attempt_{index}"
            shard = super().collect(spec, attempt, episodes=episodes, master_seed=master_seed,
                                    profile=profile, timeout=timeout)
            audit = audit_dataset(shard, expected_episodes=episodes,
                                  expected_video_keys=self._video_keys(spec), workers=8)
            atomic_json(attempt / "video_integrity_v1.json", audit)
            attempts.append({"attempt": index, "dataset": str(shard), "passed": audit["passed"],
                             "video_set_sha256": audit.get("video_set_sha256"),
                             "failed_videos": audit.get("failed_videos", [])})
            atomic_json(destination / "collection_admission_v1.json", {
                "kind": "same_seed_serialization_retry_v1", "task": spec.name,
                "master_seed": master_seed, "target_episodes": episodes,
                "profile": profile, "attempts": attempts,
                "environment_or_success_modified": False,
                "performance_conditioned_retry": False,
                "status": "admitted" if audit["passed"] else "retrying",
            })
            if audit["passed"]:
                return shard
        raise RuntimeError(f"three same-seed collections failed full video decoding: {destination}")

    def train(self, spec, root: Path, output: Path, *, steps: int,
              params: dict | None = None, mixture: Path | None = None,
              seed: int = 1000, resume: bool = False) -> Path:
        if (output.name == "controlled_baseline" and mixture is None and steps == 80000
                and seed == 1000 and not resume):
            audit = audit_checkpoint(self.reused_baseline, 80000, 1000)
            if not audit["passed"]:
                raise RuntimeError("prior controlled baseline failed checkpoint audit")
            atomic_json(output / "reused_checkpoint_v1.json", {
                "kind": "audited_checkpoint_reuse_v1", "created_at": now(),
                "source": str(self.reused_baseline), "model_sha256": digest(
                    self.reused_baseline / "model.safetensors"),
                "checkpoint_audit": audit, "retrained": False,
                "reason": "same frozen official data, recipe, train seed and 80k budget",
                "policy_performance_claim": False,
            })
            return self.reused_baseline
        return super().train(spec, root, output, steps=steps, params=params, mixture=mixture,
                             seed=seed, resume=resume)

    def evaluate(self, spec, checkpoint: Path, output: Path, *, episodes: int,
                 master_seed: int, purpose: str = "development", policy: str = "act") -> dict:
        reusable = {
            "controlled_baseline_development_bank_1":
                self.prior_task / "evaluations/controlled_baseline_development_bank_1",
            "released_checkpoint_development_bank_1":
                self.prior_task / "evaluations/released_checkpoint_development_bank_1",
        }
        source = reusable.get(output.name)
        if source is not None and purpose == "development" and episodes == 40 and master_seed == 82_000_001:
            verified = verify_reused_evaluation(source, checkpoint, task=spec.name,
                                                episodes=episodes, master_seed=master_seed,
                                                purpose=purpose)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() or output.is_symlink():
                if not output.is_symlink() or output.resolve() != source.resolve():
                    raise RuntimeError(f"reused evaluation destination already differs: {output}")
            else:
                output.symlink_to(source, target_is_directory=True)
            atomic_json(output.parent / f"{output.name}_reuse_v1.json", {
                "kind": "verified_evaluation_reuse_v1", "created_at": now(),
                "source": str(source), "checkpoint": str(checkpoint),
                "metrics_sha256": verified["metrics_sha256"],
                "request_sha256": verified["request_sha256"],
                "checks": verified["checks"], "rerun": False,
            })
            return verified["metrics"]
        return super().evaluate(spec, checkpoint, output, episodes=episodes,
                                master_seed=master_seed, purpose=purpose, policy=policy)


def _link_view(view: Path, task: str, source: Path) -> None:
    view.mkdir(parents=True, exist_ok=True)
    destination = view / task
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise RuntimeError(f"combined research view changed: {destination}")
        return
    destination.symlink_to(source, target_is_directory=True)


def run_recovery(workspace: Path, root: Path, experiment_output: Path,
                 prior_output: Path, main_state: Path, *, max_hours: float,
                 poll_seconds: float) -> dict:
    local = Path(__file__).resolve()
    manifest = {
        "kind": "water_pouring_video_integrity_recovery_v1",
        "failed_experiment_preserved": True,
        "same_pre_registered_seed_banks": True,
        "serialization_retry_uses_same_master_seed": True,
        "final_test_used": False,
        "sota_claim": False,
        "sources": freeze_files([
            local, local.with_name("dataset_video_integrity_v1.py"),
            local.with_name("research_comparison.py"), local.with_name("checkpoint_audit.py"),
            local.with_name("queue_manifest.py"),
            workspace / "autosim/autosim/research/controller.py",
            workspace / "autosim/autosim/research/runtime.py",
        ]),
    }
    versioned_manifest(root, manifest)
    state_path = root / "state.json"
    prior_state = read_json(state_path) if state_path.is_file() else {}
    if prior_state.get("status") in {"three_task_comparison_complete", "requires_review",
                                    "queue_wait_budget_exhausted"}:
        return prior_state
    deadline = float(prior_state.get("deadline_epoch", time.time() + max_hours * 3600))
    state = {**prior_state, "status": "running", "started_at": prior_state.get("started_at", now()),
             "deadline_epoch": deadline, "formal_milestone_complete": False}
    prior_research = prior_output / "research/train_seed_1000"
    while time.time() < deadline:
        upstream = read_json(main_state) if main_state.is_file() else {}
        drawer = read_json(prior_research / "drawer_open_place/state.json") if (
            prior_research / "drawer_open_place/state.json").is_file() else {}
        if upstream.get("status") not in {"requires_review", "three_task_comparison_complete"}:
            state.update(stage="waiting_upstream_drawer_open_place",
                         upstream_status=upstream.get("status", "missing"),
                         drawer_status=drawer.get("status", "missing"), updated_at=now())
            atomic_json(state_path, state)
            time.sleep(poll_seconds)
            continue
        if drawer.get("status") != "completed_development":
            state.update(status="requires_review", stage="upstream_drawer_incomplete",
                         drawer_status=drawer.get("status", "missing"), updated_at=now())
            atomic_json(state_path, state)
            return state
        prepare_experiment_root(prior_output, experiment_output)
        runtime = FullDecodeRecoveryRuntime(workspace, experiment_output, gpu="0",
                                            prior_task=prior_research / TASK)
        config = ResearchConfig(rounds=2, screen_steps=20000, final_steps=80000,
                                development_episodes=40, confirmation_episodes=100,
                                collect_per_round=50, train_seed=1000,
                                hours_per_task=40, allow_llm=False, pilot=False)
        result = ResearchController(runtime, config).run([TASK])[TASK]
        if result.get("status") != "completed_development":
            state.update(status="requires_review", stage="water_pouring_recovery_incomplete",
                         water_pouring_status=result.get("status"), error=result.get("error"),
                         updated_at=now())
            atomic_json(state_path, state)
            return state
        view = root / "combined_research_view"
        _link_view(view, "click_bell", prior_research / "click_bell")
        _link_view(view, "drawer_open_place", prior_research / "drawer_open_place")
        _link_view(view, TASK, experiment_output / "research/train_seed_1000" / TASK)
        report = write_report(root, view)
        established = bool(report.get("automatic_decision_beats_random_control_established"))
        state.update(status="three_task_comparison_complete",
                     stage="awaiting_replication_or_final_test",
                     formal_milestone_complete=established,
                     automatic_decision_beats_random_control_established=established,
                     combined_research_view=str(view), updated_at=now())
        atomic_json(state_path, state)
        return state
    state.update(status="queue_wait_budget_exhausted", updated_at=now())
    atomic_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment-output", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path, required=True)
    parser.add_argument("--main-state", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=336)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    run_recovery(args.workspace.absolute(), args.root.absolute(),
                 args.experiment_output.absolute(), args.prior_output.absolute(),
                 args.main_state.absolute(), max_hours=args.max_hours,
                 poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
