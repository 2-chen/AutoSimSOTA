"""Strict recovery for native collection crashes before the first scene reset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .common import atomic_json, digest, now, read_json
from .score_push_pipeline import production


NATIVE_STARTUP_CODES = {-11, -6}
MAX_STARTUP_ATTEMPTS = 3


def _verify_startup_failure(destination: Path) -> tuple[dict, int]:
    process_dir = destination / "process"
    record_path = process_dir / "process.json"
    if not record_path.is_file():
        raise RuntimeError("no failed process record to recover")
    record = read_json(record_path)
    if record.get("status") != "failed" or record.get("returncode") not in NATIVE_STARTUP_CODES:
        raise RuntimeError("recovery is restricted to native startup crashes")
    if (destination / "collection.json").exists() or (destination / "scene_resets.jsonl").exists():
        raise RuntimeError("collection reached an auditable reset/manifest; startup recovery forbidden")
    prior = sorted(destination.glob("startup_attempt_*/process/process.json"))
    attempt = len(prior) + 1
    if attempt >= MAX_STARTUP_ATTEMPTS:
        raise RuntimeError("native startup retry limit exhausted")
    for index, path in enumerate(prior, start=1):
        if path.parent.parent.name != f"startup_attempt_{index}":
            raise RuntimeError("non-contiguous startup attempt archive")
        old = read_json(path)
        if old.get("command") != record.get("command") or old.get("cwd") != record.get("cwd"):
            raise RuntimeError("startup retry command/cwd changed")
    return record, attempt


def recover_production(workspace: Path, root: Path, task: str) -> dict:
    workspace, root = workspace.resolve(), root.resolve()
    destination = root / task / "production_collection"
    record, attempt = _verify_startup_failure(destination)
    archive = destination / f"startup_attempt_{attempt}" / "process"
    archive.parent.mkdir(parents=True, exist_ok=False)
    decision = {
        "kind": "native_collection_startup_retry", "created_at": now(),
        "task": task, "failed_attempt": attempt, "next_attempt": attempt + 1,
        "returncode": record["returncode"], "elapsed_seconds": record.get("elapsed_seconds"),
        "completed_scene_resets": 0, "collection_manifest_exists": False,
        "same_command": True, "same_seed_bank": True, "same_target": True,
        "failed_process_sha256": digest(destination / "process/process.json"),
        "failed_log_sha256": digest(destination / "process/stdout.log"),
    }
    atomic_json(destination / f"startup_attempt_{attempt}" / "retry_decision.json", decision)
    shutil.move(str(destination / "process"), str(archive))
    result = production(workspace, root, task)
    new_record = read_json(destination / "process/process.json")
    if new_record.get("command") != record.get("command") or new_record.get("cwd") != record.get("cwd"):
        raise RuntimeError("recovered collection changed command/cwd")
    receipt = {**decision, "status": "completed", "result": result,
               "recovered_at": now(), "new_process_sha256": digest(destination / "process/process.json")}
    atomic_json(destination / "startup_recovery_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    print(recover_production(args.workspace, args.root, args.task))


if __name__ == "__main__":
    main()
