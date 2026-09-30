"""Audit recorded initial observations and repeated same-policy rollouts.

Matching a small audit is evidence, never a universal determinism guarantee.
"""
import json
from pathlib import Path

from autosim.robosyn_mvp import gpu_lock
from autosim.research.common import assert_frozen, atomic_json, digest, exclusive, freeze_files, immutable_json, read_json
from autosim.research.ledger import SeedLedger, compare
from autosim.research.registry import TASK_IDS, load_task


def initial_physics(directory: Path):
    metrics = read_json(directory / "evaluation_metrics.json")
    path = Path(metrics.get("artifact_directory", directory)) / "telemetry.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    initial = [row for row in rows if row["step"] == 0]
    if len({row["seed"] for row in initial}) != len(initial):
        raise ValueError("duplicate initial physics records")
    return {row["seed"]: row for row in initial}


def numeric_initial_audit(first: Path, second: Path):
    """Describe differences without inventing a post-hoc pass tolerance."""
    import numpy as np

    ma, _ = initial_observations(first)
    mb, _ = initial_observations(second)
    compare(ma, mb)
    a, b = initial_physics(first), initial_physics(second)
    details = []
    for episode in ma["episodes"]:
        seed = episode["episode_seed"]
        if seed not in a or seed not in b:
            details.append({"seed": seed, "status": "missing_initial_physics"})
            continue
        x, y = a[seed], b[seed]
        entities = {}
        for key in set(x["entities"]) & set(y["entities"]):
            p = np.asarray(x["entities"][key]["pose"]).reshape(-1, 4, 4)
            q = np.asarray(y["entities"][key]["pose"]).reshape(-1, 4, 4)
            if p.shape != q.shape:
                raise ValueError("different initial entity batch shapes")
            translation = np.linalg.norm(p[:, :3, 3] - q[:, :3, 3], axis=-1)
            entities[key] = {"translation_difference_m": float(translation.max()),
                             "rotation_matrix_max_absolute_difference": float(np.abs(p[:, :3, :3] - q[:, :3, :3]).max())}
        details.append({"seed": seed, "status": "measured",
                        "joint_max_absolute_difference": float(np.abs(np.asarray(x["robot_qpos"]) - np.asarray(y["robot_qpos"])).max()),
                        "entities": entities,
                        "missing_entities": sorted(set(x["entities"]) ^ set(y["entities"]))})
    return {"episodes": details, "pass_tolerance": "not_predeclared_no_pass_claim",
            "image_difference": "not_separable_from_existing_combined_observation_hash",
            "hidden_physics_state_verified": False}


def initial_observations(directory: Path):
    metrics = read_json(directory / "evaluation_metrics.json")
    root = Path(metrics.get("artifact_directory", directory))
    path = root / "initializations.jsonl"
    if not path.exists():
        return metrics, None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate initialization records")
    if seeds != [row["episode_seed"] for row in metrics["episodes"]]:
        raise ValueError("initialization coverage/order differs from evaluated episodes")
    return metrics, [row["allowed_observation_sha256"] for row in rows]


def audit_pair(first: Path, second: Path, *, same_checkpoint=False):
    a, ia = initial_observations(first)
    b, ib = initial_observations(second)
    compare(a, b)  # Strict task/protocol/seed/summary consistency; do not use its p-value here.
    if ia is None or ib is None:
        return {"status": "insufficient_records", "universal_determinism_verified": False}
    equal = [x == y for x, y in zip(ia, ib)]
    results_equal = [(x["success"], x["action_steps"]) == (y["success"], y["action_steps"])
                     for x, y in zip(a["episodes"], b["episodes"])]
    if same_checkpoint:
        pa = read_json(Path(a.get("artifact_directory", first)) / "protocol.json")
        pb = read_json(Path(b.get("artifact_directory", second)) / "protocol.json")
        if pa["checkpoint_sha256"] != pb["checkpoint_sha256"]:
            raise ValueError("cannot label different checkpoints a same-policy repeat")
    return {"status": "observations_match" if all(equal) else "initial_observations_differ",
            "episode_count": len(equal), "matching_initial_observations": sum(equal),
            "mismatching_episode_seeds": [r["episode_seed"] for r, ok in zip(a["episodes"], equal) if not ok],
            "same_checkpoint": same_checkpoint,
            "same_policy_matching_outcomes_and_steps": sum(results_equal) if same_checkpoint else None,
            "small_audit_passed": all(equal) and (all(results_equal) if same_checkpoint else True),
            "universal_determinism_verified": False,
            "limitation": "RGB/state hashes do not certify the complete hidden physics/RNG state"}


def run_audit(runtime, task, checkpoint, *, episodes=5):
    directory = runtime.output / "validation" / task / "determinism"
    if not 3 <= episodes <= 20:
        raise ValueError("determinism audit requires 3–20 predeclared development episodes")
    with exclusive(runtime.output / "suite.lock"), gpu_lock(runtime.gpu):
        master = 210_000_000 + list(TASK_IDS).index(task) * 1000
        immutable_json(directory / "request.json", {"task": task, "checkpoint": str(checkpoint),
                       "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
                       "config_sha256": digest(checkpoint / "config.json"),
                       "episodes": episodes, "master_seed": master, "purpose": "development"})
        source_path = directory / "validation_source.json"
        immutable_json(source_path, freeze_files(list(Path(__file__).parent.glob("*.py"))))
        ledger = SeedLedger(runtime.output / "seeds.sqlite")
        try:
            ledger.reserve(task, "validation:determinism", "development", master, episodes)
            for label in ("first", "second"):
                assert_frozen(read_json(source_path))
                runtime.evaluate(load_task(runtime.repo, task), checkpoint, directory / label,
                                 episodes=episodes, master_seed=master, purpose="development")
            result = audit_pair(directory / "first", directory / "second", same_checkpoint=True)
            result["numeric_initial_audit"] = numeric_initial_audit(directory / "first", directory / "second")
            assert_frozen(read_json(source_path))
            atomic_json(directory / "audit.json", result)
            return result
        finally:
            ledger.close()
