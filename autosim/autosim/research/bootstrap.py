"""Freeze existing evidence before modifying execution code; no credential copies."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .common import atomic_json, digest, freeze_files, now
from .registry import inventory


def bootstrap(workspace: Path, output: Path) -> dict:
    destination = output / "bootstrap.json"
    if destination.exists():
        raise FileExistsError(f"baseline snapshot already exists: {destination}")
    repo = workspace / "AutoSimSOTA/RoboSynChallenge"
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [repo / p for p in ["scripts/run_env.py", "scripts/eval_policy.py",
                    "policy/act/scripts/train.py", "policy/act/deploy_policy.py",
                    "robosynchallenge/managers/datasets.py"]]
    source_paths += list((workspace / "autosim/autosim").glob("robosyn*.py"))
    for path in source_paths:
        target = output / "source_before" / path.relative_to(workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    anchors = [repo / "checkpoints/ACT_sim_click_bell/model.safetensors",
               workspace / "autosim/output/robosyn_click_bell_v3/mixture_ablation/M1_append/candidates/policy/train/checkpoints/080000/pretrained_model/model.safetensors",
               workspace / "autosim/output/official_vs_2500_clean_20260902/summary.json"]
    record = {"created_at": now(), "workspace": str(workspace),
              "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "source_hashes": freeze_files(source_paths), "evidence_hashes": freeze_files(anchors),
              "disk_free_bytes": shutil.disk_usage(workspace).free,
              "credentials_copied": False, "tasks": inventory(repo)}
    atomic_json(destination, record)
    atomic_json(output / "task_inventory.json", record["tasks"])
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bootstrap(args.workspace.absolute(), args.output.absolute())
    print({"snapshot": str(args.output), "tasks": len(result["tasks"]),
           "contracts_passed": sum(t["contract_status"] == "passed" for t in result["tasks"])})
