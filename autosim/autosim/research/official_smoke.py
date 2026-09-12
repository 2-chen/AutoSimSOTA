"""Policy-only integration gate for tasks whose training expert is unavailable.

Never upgrades the separate full-collection smoke status.
"""
from pathlib import Path

from autosim.robosyn_mvp import gpu_lock
from .common import atomic_json, exclusive, now, read_json, redact
from .registry import load_task


def official_assets(runtime, task):
    item = next(row for row in read_json(runtime.output / "task_inventory.json") if row["task"] == task)
    asset_file = runtime.output / "asset_status.json"
    assets = read_json(asset_file)["tasks"].get(task, {}) if asset_file.exists() else {}
    data = (Path(item["official_dataset"]) if item.get("dataset_available") else
            Path(assets["dataset"]["path"]) if assets.get("dataset", {}).get("status") == "completed" else None)
    model = (Path(item["official_checkpoint"]) if item.get("checkpoint_available") else
             Path(assets["model"]["path"]) if assets.get("model", {}).get("status") == "completed" else None)
    return data, model


def official_smoke(runtime, tasks, *, steps=200):
    from .registry import TASK_IDS

    path = runtime.output / "official_policy_smoke_status.json"
    with exclusive(runtime.output / "suite.lock"), gpu_lock(runtime.gpu):
        state = read_json(path) if path.exists() else {"tasks": {}}
        for task in tasks:
            row = state["tasks"].setdefault(task, {"automatic_collection_validated": False})
            if row.get("status") in {"passed", "failed"}:
                continue
            root, _ = official_assets(runtime, task)
            if root is None:
                row.update(status="waiting_assets", reason="complete official dataset is not available")
                atomic_json(path, state)
                continue
            destination = runtime.output / "official_policy_smoke" / task
            spec = load_task(runtime.repo, task)
            try:
                row.update(status="running", stage="official_data_audit", dataset=str(root), started_at=now())
                atomic_json(path, state)
                runtime.prepare_data(spec, root, destination)
                checkpoint = runtime.train(spec, root, destination / "candidate", steps=steps)
                row.update(stage="evaluation", checkpoint=str(checkpoint))
                atomic_json(path, state)
                metrics = runtime.evaluate(spec, checkpoint, destination / "evaluation", episodes=3,
                    master_seed=61_000_000 + list(TASK_IDS).index(task) * 100_000, purpose="smoke")
                row.update(status="passed", stage="complete", summary=metrics["summary"], finished_at=now())
            except Exception as exc:
                row.update(status="failed", error=redact(f"{type(exc).__name__}: {exc}"), finished_at=now())
            atomic_json(path, state)
    return state
