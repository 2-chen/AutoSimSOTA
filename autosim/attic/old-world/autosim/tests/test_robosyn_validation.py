"""Temporary CPU fixtures only: never benchmark score evidence."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autosim.research.common import atomic_json, digest
from autosim.research_validation.determinism import audit_pair, initial_observations, numeric_initial_audit
from autosim.research_validation.repeats import lock_recipes, recipe_for_checkpoint, recipe_schedule, run_repeats


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def evaluation(self, name, hashes=("a", "b"), checkpoint="same"):
        path = self.root / name
        path.mkdir()
        metrics = {"test_fixture_only": True, "execution_mode": "real_simulation", "purpose": "development",
                   "harness": "test_fixture_only", "config": {"task": "click_bell", "setting": "random", "timeout_action_steps": 361, "seed": 10},
                   "episodes": [{"episode_seed": seed, "success": False, "action_steps": 361} for seed in (1, 2)],
                   "summary": {"episode_count": 2, "success_count": 0}}
        atomic_json(path / "evaluation_metrics.json", metrics)
        atomic_json(path / "protocol.json", {"checkpoint_sha256": checkpoint})
        from autosim.research.common import event
        for seed, h in zip((1, 2), hashes):
            event(path / "initializations.jsonl", "reset", seed=seed, allowed_observation_sha256=h)
        return path

    def test_exact_hash_audit_is_not_a_universal_determinism_claim(self):
        a, b = self.evaluation("a"), self.evaluation("b")
        result = audit_pair(a, b, same_checkpoint=True)
        self.assertTrue(result["small_audit_passed"])
        self.assertFalse(result["universal_determinism_verified"])

    def test_different_initial_observations_and_checkpoint_rejected(self):
        a, b = self.evaluation("a"), self.evaluation("b", hashes=("x", "b"), checkpoint="different")
        self.assertFalse(audit_pair(a, b)["small_audit_passed"])
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            audit_pair(a, b, same_checkpoint=True)

    def test_duplicate_initializations_are_not_silently_overwritten(self):
        from autosim.research.common import event
        path = self.evaluation("a")
        event(path / "initializations.jsonl", "reset", seed=1, allowed_observation_sha256="a")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            initial_observations(path)

    def test_missing_physics_is_unknown_not_zero_difference(self):
        result = numeric_initial_audit(self.evaluation("a"), self.evaluation("b"))
        self.assertTrue(all(r["status"] == "missing_initial_physics" for r in result["episodes"]))
        self.assertFalse(result["hidden_physics_state_verified"])

    def production_fixture(self):
        from types import SimpleNamespace
        runtime = SimpleNamespace(output=self.root, repo=self.root / "repo", gpu="fixture", deadline=None)
        directory = self.root / "research/train_seed_1000/click_bell"
        source = self.root / "fixture_source.py"
        source.write_text("# CPU fixture only\n")
        atomic_json(directory / "frozen_protocol.json", {str(source): digest(source)})
        checkpoints = {}
        for arm in ("controlled_baseline", "auto", "random"):
            folder = directory / arm
            checkpoint = folder / "train/checkpoints/080000/pretrained_model"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"fixture-not-a-real-model")
            atomic_json(checkpoint / "config.json", {"test_fixture_only": True})
            atomic_json(folder / "recipe_80000.json", {"steps": 80000, "seed": 1000,
                "training_data_content_ids": {"fixture_root": "fixture_content"}, "dataset": "fixture_root",
                "mixture": None, "mixture_sha256": None, "params": {"optimizer_lr": 1e-5}})
            checkpoints[arm] = str(checkpoint)
        state = {"task": "click_bell", "status": "completed_development", "pilot_only": False,
                 "completed_rounds": 2, "final_evaluation": "not_run", "baseline_checkpoint": checkpoints["controlled_baseline"],
                 "selected": {arm: {"checkpoint": checkpoints[arm]} for arm in ("auto", "random")}}
        atomic_json(directory / "state.json", state)
        return runtime, directory, state

    def test_pilot_and_post_final_repeats_are_rejected(self):
        runtime, directory, state = self.production_fixture()
        state["pilot_only"] = True
        atomic_json(directory / "state.json", state)
        with self.assertRaisesRegex(ValueError, "production"):
            lock_recipes(runtime, "click_bell")
        state["pilot_only"] = False
        atomic_json(directory / "state.json", state)
        atomic_json(self.root / "final/train_seed_1000/click_bell/locked_selection.json", {})
        with self.assertRaisesRegex(ValueError, "retrospectively"):
            lock_recipes(runtime, "click_bell")

    def test_repeat_seed_identity_and_recipe_immutability(self):
        runtime, directory, state = self.production_fixture()
        with self.assertRaisesRegex(ValueError, "distinct"):
            lock_recipes(runtime, "click_bell", repeat_seeds=(1000, 1001))
        _, lock = lock_recipes(runtime, "click_bell")
        self.assertFalse(lock["search_repeated"])
        path, recipe = recipe_for_checkpoint(Path(state["selected"]["auto"]["checkpoint"]))
        recipe["params"]["optimizer_lr"] = 2e-5
        atomic_json(path, recipe)
        with self.assertRaisesRegex(ValueError, "immutable artifact changed"):
            lock_recipes(runtime, "click_bell")

    def test_repeats_train_six_times_without_collection_or_search(self):
        runtime, _, _ = self.production_fixture()
        calls = []
        def train(spec, root, output, **kwargs):
            calls.append(kwargs)
            checkpoint = output / "train/checkpoints/080000/pretrained_model"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"fixture-repeat-not-a-real-model")
            atomic_json(checkpoint / "config.json", {"test_fixture_only": True})
            return checkpoint
        runtime.train = train
        with patch("autosim.research_validation.repeats.gpu_lock", return_value=nullcontext()), \
             patch("autosim.research_validation.repeats.load_task", return_value=None), \
             patch("autosim.research_validation.repeats.training_versions", return_value=[{"root": "fixture_root", "content_id": "fixture_content"}]):
            result = run_repeats(runtime, "click_bell")
            self.assertEqual(result["status"], "completed_training_repeats")
            self.assertEqual(len(calls), 6)
            self.assertTrue(all(not r["resume"] for r in calls))
            self.assertEqual([r["seed"] for r in calls], [1001, 1002] * 3)
            run_repeats(runtime, "click_bell")
            self.assertEqual(len(calls), 6)

    def test_resumed_source_requires_and_preserves_screening_boundary(self):
        _, _, state = self.production_fixture()
        checkpoint = Path(state["selected"]["auto"]["checkpoint"])
        path, recipe = recipe_for_checkpoint(checkpoint)
        recipe["resume"] = True
        atomic_json(path, recipe)
        with self.assertRaisesRegex(ValueError, "missing its initial"):
            recipe_schedule(checkpoint)
        atomic_json(path.parent / "recipe_20000.json", {**recipe, "steps": 20000, "resume": False})
        stages = recipe_schedule(checkpoint)
        self.assertEqual([(s["recipe"]["steps"], s["recipe"]["resume"]) for s in stages], [(20000, False), (80000, True)])

    def test_segmented_repeats_resume_only_their_own_seed_output(self):
        runtime, _, state = self.production_fixture()
        for arm in ("auto", "random"):
            path, recipe = recipe_for_checkpoint(Path(state["selected"][arm]["checkpoint"]))
            atomic_json(path, {**recipe, "resume": True})
            atomic_json(path.parent / "recipe_20000.json", {**recipe, "steps": 20000, "resume": False})
        calls, previous = [], {}
        def train(spec, root, output, **kwargs):
            if kwargs["resume"]:
                self.assertEqual(previous[output], (kwargs["seed"], 20000))
            else:
                self.assertNotIn(output, previous)
            previous[output] = kwargs["seed"], kwargs["steps"]
            calls.append(kwargs)
            checkpoint = output / "train/checkpoints" / f"{kwargs['steps']:06d}" / "pretrained_model"
            atomic_json(checkpoint / "config.json", {"test_fixture_only": True})
            (checkpoint / "model.safetensors").write_bytes(b"fixture-model")
            return checkpoint
        runtime.train = train
        with patch("autosim.research_validation.repeats.gpu_lock", return_value=nullcontext()), \
             patch("autosim.research_validation.repeats.load_task", return_value=None), \
             patch("autosim.research_validation.repeats.training_versions", return_value=[{"root": "fixture_root", "content_id": "fixture_content"}]):
            result = run_repeats(runtime, "click_bell")
        self.assertEqual(result["status"], "completed_training_repeats", result.get("error"))
        self.assertEqual(len(calls), 10)
        self.assertEqual(sum(bool(c["resume"]) for c in calls), 4)

    def test_priority_assets_publish_only_after_both_pinned_downloads(self):
        from types import SimpleNamespace
        from autosim.research_validation.priority_assets import acquire_task
        from autosim.research.common import read_json
        entry = {"task": "item_assembly", **{kind: {"repository": f"fixture/{kind}",
                 "revision": "fixture-pinned-revision", "available": True} for kind in ("model", "dataset")}}
        atomic_json(self.root / "hub_inventory.json", [entry])
        atomic_json(self.root / "task_inventory.json", [{"task": "item_assembly", "dataset_available": False}])
        atomic_json(self.root / "asset_status.json", {"fixture": "main downloader untouched"})
        before = digest(self.root / "asset_status.json")
        seen = []
        def download(repo, **kwargs):
            self.assertEqual(kwargs["revision"], "fixture-pinned-revision")
            self.assertFalse(read_json(self.root / "task_inventory.json")[0]["dataset_available"])
            seen.append(kwargs["repo_type"])
            marker = kwargs["local_dir"] / ("model.safetensors" if kwargs["repo_type"] == "model" else "meta/info.json")
            atomic_json(marker, {"test_fixture_only": True})
        with patch("huggingface_hub.get_token", return_value=None), \
             patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.snapshot_download", side_effect=download), \
             patch("autosim.research_validation.priority_assets.shutil.disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
            api.return_value.repo_info.return_value.siblings = [SimpleNamespace(size=20)]
            result = acquire_task(self.root, "item_assembly")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(seen, ["model", "dataset"])
        self.assertTrue(read_json(self.root / "task_inventory.json")[0]["dataset_available"])
        self.assertEqual(digest(self.root / "asset_status.json"), before)

    def test_validation_queue_waits_for_production_and_never_runs_final(self):
        from autosim.research_validation.continuation import next_action
        research = {"status": "completed_development", "pilot_only": False, "completed_rounds": 2}
        self.assertEqual(next_action({"status": "running"}, research, {}, {}), "waiting_production")
        done = {"status": "gated_handoff"}
        self.assertEqual(next_action(done, {**research, "pilot_only": True}, {}, {}), "blocked_production_incomplete")
        self.assertEqual(next_action(done, research, {}, {}), "determinism")
        audit = {"status": "initial_observations_differ", "small_audit_passed": False}
        self.assertEqual(next_action(done, research, audit, {}), "train-repeats")
        self.assertEqual(next_action(done, research, audit, {"status": "failed"}), "blocked_repeat_training")
        self.assertEqual(next_action(done, research, audit, {"status": "completed_training_repeats"}),
                         "awaiting_final_protocol_and_semantic_review")


if __name__ == "__main__":
    unittest.main()
