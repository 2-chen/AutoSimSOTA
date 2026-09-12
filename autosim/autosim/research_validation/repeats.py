"""Repeat locked recipes, not the search, with two additional training seeds."""
from pathlib import Path
import time

from autosim.robosyn_mvp import gpu_lock
from autosim.research.common import (assert_frozen, atomic_json, digest, exclusive,
    freeze_files, immutable_json, now, read_json, redact)
from autosim.research.controller import task_cost
from autosim.research.data_version import training_versions
from autosim.research.registry import load_task


def recipe_for_checkpoint(checkpoint: Path) -> tuple[Path, dict]:
    if not checkpoint.parent.name.isdigit():
        raise ValueError("checkpoint must identify its exact training step")
    steps = int(checkpoint.parent.name)
    path = checkpoint.parents[3] / f"recipe_{steps}.json"
    recipe = read_json(path)
    if recipe["steps"] != steps or "training_data_content_ids" not in recipe:
        raise ValueError("checkpoint recipe lacks matching steps or data content lineage")
    return path, recipe


def recipe_schedule(checkpoint: Path):
    """Preserve screen→resume boundaries as well as the final update count."""
    final_path, final = recipe_for_checkpoint(checkpoint)
    paths = [final_path]
    if final.get("resume", False):
        paths = sorted((p for p in final_path.parent.glob("recipe_*.json")
                        if int(p.stem.split("_")[-1]) <= final["steps"]),
                       key=lambda p: int(p.stem.split("_")[-1]))
    stages = [{"path": str(p), "sha256": digest(p), "recipe": read_json(p)} for p in paths]
    if stages[0]["recipe"].get("resume", False):
        raise ValueError("resumed source recipe is missing its initial training stage")
    for index, stage in enumerate(stages):
        recipe = stage["recipe"]
        if index and not recipe.get("resume", False):
            raise ValueError("ambiguous training schedule: later stage restarts")
        for key in ("seed", "params", "dataset", "mixture", "training_data_content_ids"):
            if recipe[key] != final[key]:
                raise ValueError(f"training schedule changes {key}; not a fixed recipe")
    return stages


def lock_recipes(runtime, task: str, *, source_seed=1000, repeat_seeds=(1001, 1002)):
    if len(repeat_seeds) != 2 or len({source_seed, *repeat_seeds}) != 3:
        raise ValueError("require the source seed plus exactly two distinct new training seeds")
    directory = runtime.output / "validation" / task / "fixed_recipe_repeats"
    source = runtime.output / "research" / f"train_seed_{source_seed}" / task / "state.json"
    state = read_json(source)
    if state.get("status") != "completed_development" or state.get("pilot_only") is not False:
        raise ValueError("only completed production research can lock final training repeats")
    if state.get("completed_rounds", 0) < 2 or state.get("final_evaluation") != "not_run":
        raise ValueError("lock repeats after two rounds and before any final evaluation")
    if list((runtime.output / "final").glob(f"train_seed_*/{task}/locked_selection.json")):
        raise ValueError("a final selection already exists; cannot add training repeats retrospectively")
    frozen_path = source.parent / "frozen_protocol.json"
    assert_frozen(read_json(frozen_path))
    checkpoints = {"controlled_baseline": state["baseline_checkpoint"],
                   **{arm: row["checkpoint"] for arm, row in state["selected"].items()}}
    if set(checkpoints) != {"controlled_baseline", "auto", "random"}:
        raise ValueError("three declared training arms are required")
    arms = {}
    for arm, checkpoint in checkpoints.items():
        path, recipe = recipe_for_checkpoint(Path(checkpoint))
        if recipe["seed"] != source_seed:
            raise ValueError("source checkpoint seed differs from locked source study")
        arms[arm] = {"recipe": recipe, "recipe_path": str(path), "recipe_sha256": digest(path),
                     "schedule": recipe_schedule(Path(checkpoint)),
                     "source_checkpoint": checkpoint,
                     "source_weight_sha256": digest(Path(checkpoint) / "model.safetensors"),
                     "source_config_sha256": digest(Path(checkpoint) / "config.json")}
    if len({row["recipe"]["steps"] for row in arms.values()}) != 1:
        raise ValueError("training arms have unequal final update budgets")
    lock = {"task": task, "source_seed": source_seed, "repeat_seeds": list(repeat_seeds),
            "source_state_sha256": digest(source), "source_state": str(source),
            "source_frozen_protocol": str(frozen_path), "arms": arms,
            "search_repeated": False, "new_collection": False,
            "training_initialization": "from_scratch_with_same_pretrained_backbone_not_from_selected_policy",
            "resume_rule": "reproduce_source_stage_boundaries_using_only_same_repeat_seed_checkpoints",
            "selection_rule": "report_all_three_training_seeds_never_select_on_final_scores"}
    immutable_json(directory / "locked_recipes.json", lock)
    return directory, lock


def run_repeats(runtime, task, *, hours=48):
    if not 0 < hours <= 72:
        raise ValueError("repeat-training process budget must be in (0,72] hours")
    with exclusive(runtime.output / "suite.lock"), gpu_lock(runtime.gpu):
        directory, lock = lock_recipes(runtime, task)
        path = directory / "state.json"
        state = read_json(path) if path.exists() else {"task": task, "created_at": now(), "checkpoints": {}, "status": "pending"}
        if state["status"] == "completed_training_repeats":
            assert_frozen(read_json(directory / "validation_source.json"))
            assert_frozen(state["checkpoint_weight_hashes"])
            assert_frozen(state["checkpoint_config_hashes"])
            return state
        if state["status"] == "failed":
            return state
        immutable_json(directory / "budget.json", {"process_hours": hours})
        source_lock = directory / "validation_source.json"
        if not source_lock.exists():
            atomic_json(source_lock, freeze_files(list(Path(__file__).parent.glob("*.py"))))
        try:
            for arm, row in lock["arms"].items():
                recipe = row["recipe"]
                assert_frozen(read_json(Path(lock["source_frozen_protocol"])))
                assert_frozen(read_json(source_lock))
                if digest(Path(row["recipe_path"])) != row["recipe_sha256"]:
                    raise ValueError("locked training recipe changed")
                root = Path(recipe["dataset"])
                mixture = Path(recipe["mixture"]) if recipe["mixture"] else None
                versions = training_versions(root, mixture, runtime.output / "data_versions")
                if {v["root"]: v["content_id"] for v in versions} != recipe["training_data_content_ids"]:
                    raise ValueError("locked training bytes changed")
                if mixture and digest(mixture) != recipe["mixture_sha256"]:
                    raise ValueError("locked mixture changed")
                checkpoints = state["checkpoints"].setdefault(arm, {})
                checkpoints[str(lock["source_seed"])] = row["source_checkpoint"]
                for seed in lock["repeat_seeds"]:
                    assert_frozen(read_json(Path(lock["source_frozen_protocol"])))
                    assert_frozen(read_json(source_lock))
                    cost = task_cost(directory)
                    if cost >= hours * 3600:
                        raise RuntimeError("fixed-recipe repeats process budget exhausted")
                    runtime.deadline = time.monotonic() + hours * 3600 - cost
                    state.update(status="running", stage=f"train:{arm}:seed_{seed}")
                    atomic_json(path, state)
                    for stage in row["schedule"]:
                        assert_frozen(read_json(Path(lock["source_frozen_protocol"])))
                        assert_frozen(read_json(source_lock))
                        if digest(Path(stage["path"])) != stage["sha256"]:
                            raise ValueError("locked training stage changed")
                        remaining = hours * 3600 - task_cost(directory)
                        if remaining <= 0:
                            raise RuntimeError("fixed-recipe repeats process budget exhausted")
                        runtime.deadline = time.monotonic() + remaining
                        checkpoint = runtime.train(load_task(runtime.repo, task), root,
                            directory / arm / f"train_seed_{seed}", steps=stage["recipe"]["steps"],
                            params=recipe["params"], mixture=mixture, seed=seed,
                            resume=stage["recipe"].get("resume", False))
                    assert_frozen(read_json(Path(lock["source_frozen_protocol"])))
                    assert_frozen(read_json(source_lock))
                    checkpoints[str(seed)] = str(checkpoint)
                    atomic_json(path, state)
            checkpoint_hashes = freeze_files([Path(checkpoint) / "model.safetensors"
                for arm in state["checkpoints"].values() for checkpoint in arm.values()])
            config_hashes = freeze_files([Path(checkpoint) / "config.json"
                for arm in state["checkpoints"].values() for checkpoint in arm.values()])
            state.update(status="completed_training_repeats", completed_at=now(),
                         checkpoint_weight_hashes=checkpoint_hashes,
                         checkpoint_config_hashes=config_hashes,
                         elapsed_process_seconds=task_cost(directory), training_seed_count=3,
                         final_evaluation="not_run", improvement="not_established")
        except Exception as exc:
            state.update(status="failed", error=redact(f"{type(exc).__name__}: {exc}"), finished_at=now())
        finally:
            runtime.deadline = None
            atomic_json(path, state)
        return state
