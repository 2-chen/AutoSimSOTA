"""Bounded training-only expert diagnostics, never policy ranking results."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from autosim.robosyn_mvp import gpu_lock
from .common import atomic_json, exclusive, read_json, redact, run_command
from .registry import TASK_IDS, load_task
from .runtime import Runtime


def diagnose(runtime, tasks, label="expert_diagnostics_v1", *, stepwise_history=False, settle_steps=0):
    output = runtime.output / label
    results = {}
    with exclusive(runtime.output / "suite.lock"), gpu_lock(runtime.gpu):
        for task in tasks:
            spec = load_task(runtime.repo, task)
            destination = output / task
            env = runtime.environment()
            env["AUTOSIM_EXPERT_PROBE_OUTPUT"] = str(destination / "checks.jsonl")
            env["ROBOSYN_DEBUG_SUCCESS_FLAGS"] = "1"
            env["AUTOSIM_EXPERT_STEPWISE_HISTORY"] = "1" if stepwise_history else "0"
            env["AUTOSIM_EXPERT_SETTLE_STEPS"] = str(settle_steps)
            command = [str(runtime.python), "-m", "autosim.research.expert_probe",
                       "--gym_config", spec.gym_config, "--action_config", spec.action_config,
                       "--num_envs", "1", "--headless", "--max_episodes", "1",
                       "--collection_seed", str(190_000_000 + list(TASK_IDS).index(task) * 1000),
                       "--collection_profile", "full_random", "--collection_mode", "expert",
                       "--collection_manifest", str(destination / "collection.json"),
                       "--dataset_save_path", str(destination / "data"),
                       "--collection_max_attempts", "4", "--collection_quiet"]
            result = {"task": task, "kind": "training_expert_diagnostic_not_policy_score",
                      "stepwise_success_history": stepwise_history, "terminal_hold_max_steps": settle_steps}
            try:
                run_command(command, cwd=runtime.repo, env=env, output=destination / "process", timeout=300)
            except Exception as exc:
                result["process_error"] = redact(f"{type(exc).__name__}: {exc}")
            checks = destination / "checks.jsonl"
            rows = [json.loads(line) for line in checks.read_text().splitlines()] if checks.exists() else []
            result["check_counts"] = dict(Counter(
                f"{r['event']}:{r.get('ik_valid', r.get('returned_valid'))}" for r in rows))
            manifest = destination / "collection.json"
            if manifest.exists():
                data = read_json(manifest)
                result.update(attempts=data.get("expert_attempt_count"),
                              successful_trajectories=len(data.get("successful_episode_seeds", [])),
                              collector_error=data.get("error"))
            result["checks_are_not_independent_episode_samples"] = True
            results[task] = result
            atomic_json(output / "summary.json", results)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+", default=["water_pouring", "item_assembly", "sample_loading"], choices=list(TASK_IDS))
    parser.add_argument("--label", default="expert_diagnostics_v1")
    parser.add_argument("--stepwise-history", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=0)
    args = parser.parse_args()
    diagnose(Runtime(args.workspace, args.output or args.workspace / "autosim/output/robosyn_general_20260905"), args.tasks, args.label,
             stepwise_history=args.stepwise_history, settle_steps=args.settle_steps)
