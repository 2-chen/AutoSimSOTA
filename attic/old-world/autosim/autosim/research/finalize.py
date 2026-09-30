"""Lock selected checkpoints before a fixed 500-episode local final evaluation.

This is not an official leaderboard submission or a three-training-seed study.
Final traces are not fed back to the research controller.
"""
from __future__ import annotations

import math
from pathlib import Path

from autosim.robosyn_mvp import gpu_lock
from .common import assert_frozen, atomic_json, digest, exclusive, now, read_json, redact
from .controller import import_known_seeds
from .ledger import SeedLedger, compare, holm
from .registry import TASK_IDS, load_task


def wilson(successes: int, count: int) -> list[float]:
    if not 0 <= successes <= count or count <= 0:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p, denominator = successes / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [0. if successes == 0 else max(0., center - half),
            1. if successes == count else min(1., center + half)]


def lock_task(runtime, task: str, train_seed: int, directory: Path) -> dict:
    state_path = runtime.output / "research" / f"train_seed_{train_seed}" / task / "state.json"
    state = read_json(state_path)
    if state.get("status") != "completed_development" or state.get("pilot_only"):
        raise ValueError("only completed non-pilot development runs are eligible")
    if state.get("completed_rounds", 0) < 2:
        raise ValueError("at least two completed research rounds required")
    frozen_path = state_path.parent / "frozen_protocol.json"
    assert_frozen(read_json(frozen_path))
    checkpoints = {"controlled_baseline": state["baseline_checkpoint"],
                   **{arm: row["checkpoint"] for arm, row in state["selected"].items()}}
    inventory = read_json(runtime.output / "task_inventory.json")
    item = next(row for row in inventory if row["task"] == task)
    if item["checkpoint_available"]:
        checkpoints["released_act"] = item["official_checkpoint"]
    else:
        asset_path = runtime.output / "asset_status.json"
        assets = read_json(asset_path)["tasks"].get(task, {}) if asset_path.exists() else {}
        if assets.get("model", {}).get("status") != "completed":
            raise ValueError("released ACT checkpoint is required for the final comparison")
        checkpoints["released_act"] = assets["model"]["path"]
    contract = {"task": task, "train_seed": train_seed, "episodes": 500,
                "master_seed": 170_000_000 + list(TASK_IDS).index(task) * 10000 + train_seed,
                "minimum_practical_gain": 0.03,
                "primary_comparison": "auto_vs_random", "family_size": len(TASK_IDS),
                "checkpoint_paths": checkpoints,
                "checkpoint_hashes": {key: digest(Path(path) / "model.safetensors") for key, path in checkpoints.items()},
                "config_hashes": {key: digest(Path(path) / "config.json") for key, path in checkpoints.items()},
                "development_state_sha256": digest(state_path), "frozen_protocol": str(frozen_path)}
    path = directory / "locked_selection.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError("final selection changed; existing final bank is retired and cannot be retuned")
    atomic_json(path, contract)
    return contract


def finalize(runtime, tasks: list[str], *, train_seed=1000) -> dict:
    root = runtime.output / "final" / f"train_seed_{train_seed}"
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "suite_status.json"
    with exclusive(runtime.output / "suite.lock"), gpu_lock(runtime.gpu):
        status = read_json(status_path) if status_path.exists() else {"created_at": now(), "tasks": {}}
        ledger = SeedLedger(runtime.output / "seeds.sqlite")
        try:
            import_known_seeds(runtime, ledger)
            for task in tasks:
                directory = root / task
                row = status["tasks"].setdefault(task, {})
                try:
                    contract = lock_task(runtime, task, train_seed, directory)
                    ledger.reserve(task, f"final:train_seed_{train_seed}", "final",
                                   contract["master_seed"], contract["episodes"])
                    row.update(status="running", significance="not_established")
                    atomic_json(status_path, status)
                    results = {}
                    for label, path in contract["checkpoint_paths"].items():
                        assert_frozen(read_json(Path(contract["frozen_protocol"])))
                        if digest(Path(path) / "model.safetensors") != contract["checkpoint_hashes"][label]:
                            raise ValueError("locked checkpoint changed")
                        results[label] = runtime.evaluate(load_task(runtime.repo, task), Path(path),
                            directory / label, episodes=500, master_seed=contract["master_seed"], purpose="final")
                    comparisons = {label: compare(results["auto"], results[label])
                                   for label in ("controlled_baseline", "random", "released_act")}
                    row.update(status="completed", finished_at=now(), comparisons=comparisons,
                               scores={label: {**data["summary"], "wilson_95_interval": wilson(
                                   data["summary"]["success_count"], 500)} for label, data in results.items()},
                               training_repeats=1, official_score=False,
                               deterministic_pairing_verified=False,
                               significance="provisional_seed_paired_test_requires_determinism_audit")
                except Exception as exc:
                    row.update(status="failed", error=redact(f"{type(exc).__name__}: {exc}"))
                atomic_json(status_path, status)
            # Missing tasks stay in the predeclared family with p=1.
            p_values = {task: status["tasks"].get(task, {}).get("comparisons", {}).get("random", {}).get(
                "one_sided_p_value", 1.) for task in TASK_IDS}
            status["exploratory_holm_rejections"] = holm(p_values)
            status["all_tasks_improved"] = False  # Requires determinism audit and training repeats.
            status["unmet_acceptance_gates"] = ["three_fixed_recipe_training_seeds",
                                                "determinism_or_valid_unpaired_inference",
                                                "semantic_video_review"]
            atomic_json(status_path, status)
        finally:
            ledger.close()
    return status
