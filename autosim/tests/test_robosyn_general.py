import copy
import json
import tempfile
import unittest
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np

from autosim.research.common import assert_frozen, freeze_files, redact
from autosim.research.ledger import SeedLedger, compare, holm
from autosim.research.policy_rpc import observation_digest, policy_observation
from autosim.research.registry import TASK_IDS, filter_training_events, load_task
from autosim.research.proposals import Proposal, propose
from autosim.research.analysis import analyze
from autosim.research.common import atomic_json
from autosim.research.common import read_json, run_command
from autosim.research.evaluation import TraceEnv
from autosim.research.finalize import wilson, lock_task
from autosim.research.runtime import Runtime, retryable_evaluation_startup
from autosim.research.controller import make_mixture, ResearchConfig, ResearchController
from autosim.research.collection_worker import explicit_joint_flip_contract
from autosim.research.data_version import fingerprint_dataset


REPO = Path(__file__).resolve().parents[2] / "RoboSynChallenge"


class GeneralResearchTest(unittest.TestCase):
    def test_bounded_collection_preserves_partial_yield_and_attempt_budget(self):
        class FixtureRuntime(Runtime):
            def run(self, command, output, timeout, **kwargs):
                manifest = Path(command[command.index("--collection_manifest") + 1])
                destination = manifest.parent
                data_root = destination / "data/fixture"
                atomic_json(data_root / "meta/info.json", {"total_episodes": 1})
                attempts = [
                    {"seed": 10, "saved": False, "reason": "invalid"},
                    {"seed": 11, "saved": True, "reason": "saved_successful_episode"},
                    {"seed": 12, "saved": False, "reason": "invalid"},
                ]
                atomic_json(manifest, {
                    "status": "failed",
                    "error": "RuntimeError: Collection exceeded 3 expert attempts.",
                    "task": "click_bell",
                    "profile": "targeted_camera", "attempts": attempts,
                    "successful_episode_seeds": [11],
                    "dataset_paths": [str(data_root)],
                })
                rows = []
                for attempt in attempts:
                    rows.append(json.dumps({
                        "seed": attempt["seed"], "task": "click_bell",
                        "requested_profile": "targeted_camera",
                        "entities": {"button": {"pose": [[1.0]]}},
                        "robot_qpos": [0.0],
                        "privileged_training_diagnostics_only": True,
                    }))
                (destination / "scene_resets.jsonl").write_text("\n".join(rows) + "\n")
                return {"status": "completed"}

        with tempfile.TemporaryDirectory() as temp:
            runtime = FixtureRuntime(REPO.parents[1], Path(temp))
            result = runtime.collect_bounded(
                load_task(REPO, "click_bell"), Path(temp) / "probe",
                attempt_budget=3, target_episodes=3, master_seed=1,
                profile="targeted_camera")
            self.assertEqual(result["capability_state"], "partial_yield")
            self.assertEqual(result["attempts_consumed"], 3)
            self.assertEqual(result["accepted_episodes"], 1)
            self.assertEqual(result["termination"], "attempt_budget_exhausted")
            self.assertTrue(Path(result["dataset_root"]).is_dir())
            self.assertTrue(read_json(
                Path(temp) / "probe/scene_evidence_audit.json")["passed"])

    def test_data_version_detects_content_change_with_identical_episode_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            atomic_json(root / "meta/info.json", {"total_episodes": 1})
            atomic_json(root / "data/episode_0.parquet", {"synthetic_fixture": 1})
            store = Path(temp) / "versions"
            first = fingerprint_dataset(root, store)
            self.assertEqual(first, fingerprint_dataset(root, store))
            atomic_json(root / "data/episode_0.parquet", {"synthetic_fixture": 2})
            changed = fingerprint_dataset(root, store)
            self.assertNotEqual(first["content_id"], changed["content_id"])

    def test_native_retry_requires_positive_pre_reset_handshake(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root / "process/process.json", {"status": "failed", "returncode": -11})
            self.assertFalse(retryable_evaluation_startup(root))
            atomic_json(root / "startup.json", {"phase": "environment_constructing"})
            self.assertTrue(retryable_evaluation_startup(root))
            atomic_json(root / "process/process.json", {
                "status": "interrupted", "returncode": -9, "error": "TimeoutExpired"})
            self.assertTrue(retryable_evaluation_startup(root))
            atomic_json(root / "startup.json", {"phase": "evaluation_reset_started"})
            self.assertFalse(retryable_evaluation_startup(root))
            atomic_json(root / "startup.json", {"phase": "environment_ready"})
            atomic_json(root / "evaluation_metrics.json", {"episodes": []})
            self.assertFalse(retryable_evaluation_startup(root))

    def test_successful_startup_retry_is_reused_without_revisiting_failed_attempt(self):
        class FixtureRuntime(Runtime):
            def run(self, command, output, timeout, **kwargs):
                destination = Path(command[command.index("--output") + 1])
                self.calls.append(destination.name)
                atomic_json(destination / "protocol.json", {"frozen_files": {}})
                if destination.name == "eval":
                    atomic_json(destination / "process/process.json", {"status": "failed", "returncode": -11})
                    atomic_json(destination / "startup.json", {"phase": "environment_constructing"})
                    raise RuntimeError("synthetic native startup failure")
                atomic_json(destination / "evaluation_metrics.json", {
                    "execution_mode": "real_simulation", "purpose": "smoke", "test_fixture_only": True,
                    "episodes": [{"episode_seed": 1, "success": False}],
                    "config": {"task": "click_bell", "timeout_action_steps": 361}})
                return {"status": "completed"}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = FixtureRuntime(REPO.parents[1], root)
            runtime.calls = []
            checkpoint = root / "model"
            atomic_json(checkpoint / "config.json", {})
            atomic_json(checkpoint / "model.safetensors", {"synthetic_fixture": True})
            kwargs = dict(episodes=1, master_seed=1, purpose="smoke")
            for _ in range(2):
                result = runtime.evaluate(load_task(REPO, "click_bell"), checkpoint, root / "eval", **kwargs)
                self.assertEqual(result["startup_attempt_count"], 2)
            self.assertEqual(runtime.calls, ["eval", "startup_attempt_2", "startup_attempt_2"])

    def test_production_contracts_cover_ten_tasks(self):
        self.assertEqual(len(TASK_IDS), 10)
        for task in TASK_IDS:
            spec = load_task(REPO, task)
            self.assertEqual(spec.action_dim, 14)
            self.assertEqual(len(spec.cameras), 3)
            self.assertEqual(spec.correction_supported, task == "click_bell")

    def test_task_horizons_are_not_click_bell_defaults(self):
        self.assertEqual(load_task(REPO, "drawer_open_place").max_episode_steps, 900)
        self.assertEqual(load_task(REPO, "manipulate_pipette").max_episode_steps, 1000)

    def test_expert_missing_attributes_come_from_published_validation(self):
        config = read_json(REPO / "configs/handle_basket/action_config.json")
        contract = explicit_joint_flip_contract(config)
        self.assertEqual(contract["agent_qpos_flip_ids"], [3, 4])
        self.assertAlmostEqual(contract["agent_qpos_flip_threshold"], 1.1 * np.pi)
        config["action_config"] = config  # Official launcher adds a recursive alias.
        self.assertEqual(explicit_joint_flip_contract(config), contract)
        with self.assertRaises(ValueError):
            explicit_joint_flip_contract({})

    def test_nonproduction_and_clear_are_rejected(self):
        for task in ("open_pan", "ClickBellTest", "../click_bell"):
            with self.assertRaises(ValueError):
                load_task(REPO, task)
        with self.assertRaises(ValueError):
            load_task(REPO, "click_bell", "clear")

    def test_training_slice_preserves_constraints_unknown_and_placements(self):
        events = {"light": {"func": "randomize_light"}, "camera": {"func": "randomize_camera_intrinsics"},
                  "attach": {"func": "create_rigid_constraint"}, "place": {"func": "randomize_rigid_object_pose"},
                  "new": {"func": "future_semantic_event"}}
        self.assertEqual(set(filter_training_events(events, {"camera"})), {"camera", "attach", "place", "new"})
        with self.assertRaises(ValueError):
            filter_training_events(events, {"fake"})

    def test_policy_wire_strips_all_privileged_fields(self):
        obs = {"robot": {"qpos": np.zeros(7), "eef_pose": "secret"},
               "sensor": {"eye": {"color": np.zeros((2, 3, 4), dtype=np.uint8), "depth": "secret"}},
               "object_pose": "secret", "seed": 17}
        spec = {"state_dim": 7, "cameras": ["eye"], "camera_shapes": {"eye": [2, 3, 3]}}
        selected = policy_observation(obs, spec)
        self.assertEqual(set(selected), {"observation.state", "observation.images.eye"})
        self.assertEqual(selected["observation.images.eye"].shape, (1, 2, 3, 3))
        self.assertEqual(observation_digest(selected), observation_digest(selected))
        obs["robot"]["qpos"][0] = np.nan
        with self.assertRaises(ValueError):
            policy_observation(obs, spec)

    def test_missing_frozen_file_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                freeze_files([Path(temp) / "missing"])
            path = Path(temp) / "judge.py"
            path.write_text("original")
            hashes = freeze_files([path])
            path.write_text("changed")
            with self.assertRaises(RuntimeError):
                assert_frozen(hashes)

    def test_seed_banks_idempotent_but_never_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = SeedLedger(Path(temp) / "seeds.sqlite")
            seeds = ledger.reserve("task", "train", "collection", 17, 20)
            self.assertEqual(seeds, ledger.reserve("task", "train", "collection", 17, 20))
            with self.assertRaises(ValueError):
                ledger.reserve("task", "final", "final", 17, 20)
            with self.assertRaises(ValueError):
                ledger.reserve("task", "train", "collection", 18, 20)
            ledger.close()

    def result(self, successes):
        return {"execution_mode": "real_simulation", "harness": "test", "purpose": "confirmation",
                "config": {"task": "task", "setting": "random", "timeout_action_steps": 500, "seed": 0},
                "summary": {"episode_count": len(successes), "success_count": sum(successes)},
                "episodes": [{"episode_seed": i, "success": s} for i, s in enumerate(successes)]}

    def test_comparison_rejects_partial_duplicate_and_mock_results(self):
        a, b = self.result([1, 1, 0]), self.result([0, 0, 1])
        self.assertEqual(compare(a, b)["candidate_only_successes"], 2)
        for mutation in ("duplicate", "mock", "partial"):
            bad = copy.deepcopy(b)
            if mutation == "duplicate":
                bad["episodes"][0]["episode_seed"] = 1
            elif mutation == "mock":
                bad["execution_mode"] = "mock"
            else:
                bad["episodes"].pop()
            with self.assertRaises(ValueError):
                compare(a, bad)

    def test_holm_does_not_hide_failed_tasks(self):
        self.assertEqual(holm({"a": 0.001, "b": 0.04, "c": 0.04}), {"a": True, "b": False, "c": False})

    def test_credentials_are_redacted(self):
        self.assertNotIn("abcdefghijklmnopqrstuv", redact("hf_abcdefghijklmnopqrstuv"))

    def test_proposal_bounds_reject_evaluator_edits_and_invalid_windows(self):
        for params in ({"max_episode_steps": 5000}, {"optimizer_lr": 100},
                       {"chunk_size": 25, "n_action_steps": 50}):
            with self.assertRaises(ValueError):
                Proposal("test", "development", params).validate()

    def test_rule_loop_uses_feedback_and_avoids_repeated_experiments(self):
        evidence = {"source": "development", "categories": {"incomplete_after_motion": 5}, "summary": {}}
        first, provider = propose(evidence, set())
        second, _ = propose(evidence, {first.signature})
        self.assertEqual(first.params["n_action_steps"], 10)
        self.assertNotEqual(first.signature, second.signature)
        self.assertEqual(provider["used"], "rule")

    def test_final_failures_cannot_feed_research(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root / "evaluation_metrics.json", {"purpose": "final"})
            with self.assertRaises(ValueError):
                analyze(root)

    def test_trace_never_queries_success_or_advances_extra_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = SimpleNamespace(name="fixture", roles={"family": "test", "objects": [], "articulations": []},
                as_dict=lambda: {"state_dim": 1, "cameras": [], "camera_shapes": {}})
            obs = {"robot": {"qpos": np.zeros((1, 1))}}
            base = Mock()
            base.unwrapped = base
            base.reset.return_value = (obs, {})
            base.step.return_value = (obs, 0., False, False, {})
            traced = TraceEnv(base, spec, Path(temp), None, "development")
            rng_before = np.random.get_state()
            traced.reset(seed=123)
            for _ in range(20):
                traced.step(np.zeros(1))
            base.reset.assert_called_once_with(seed=123)
            self.assertEqual(base.step.call_count, 20)
            base.is_task_success.assert_not_called()
            base.get_wrapper_attr.assert_not_called()
            rng_after = np.random.get_state()
            np.testing.assert_array_equal(rng_before[1], rng_after[1])
            self.assertEqual(rng_before[2:], rng_after[2:])

    def test_process_cache_matches_command_not_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            command = [sys.executable, "-c", "print('fixture')"]
            first = run_command(command, cwd=root, env=os.environ.copy(), output=root / "job", timeout=5)
            cached = run_command(command, cwd=root, env=os.environ.copy(), output=root / "job", timeout=2)
            self.assertEqual(first, cached)
            with self.assertRaises(RuntimeError):
                run_command([sys.executable, "-c", "print('changed')"], cwd=root,
                            env=os.environ.copy(), output=root / "job", timeout=5)

    def test_failed_process_is_preserved_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kwargs = dict(cwd=root, env=os.environ.copy(), output=root / "failed", timeout=3)
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    run_command([sys.executable, "-c", "raise SystemExit(2)"], **kwargs)
            self.assertEqual(read_json(root / "failed/process.json")["returncode"], 2)

    def test_pilot_data_is_never_labelled_official(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root / "data/meta/info.json", {"total_episodes": 10, "total_frames": 500})
            path = make_mixture([root / "data"], root / "mixture.json", ["full_random"], pilot=True)
            self.assertEqual(read_json(path)["datasets"][0]["source_kind"], "self_collected_pilot")
            with self.assertRaises(RuntimeError):
                make_mixture([root / "data"], path, ["full_random"], pilot=False)

    def test_pilot_is_ineligible_for_final_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = Runtime(REPO.parents[1], root)
            atomic_json(root / "research/train_seed_1000/click_bell/state.json",
                        {"status": "completed_development", "pilot_only": True})
            with self.assertRaises(ValueError):
                lock_task(runtime, "click_bell", 1000, root / "final")

    def test_wilson_handles_boundary_scores(self):
        low, high = wilson(0, 500)
        self.assertEqual(low, 0.)
        self.assertGreater(high, 0.)
        low, high = wilson(500, 500)
        self.assertLess(low, 1.)
        self.assertAlmostEqual(high, 1.)

    def test_two_round_controller_and_completed_restart_cpu_fixture(self):
        """Synthetic results stay in a temporary unit-test directory, never a report."""
        # Production outputs now contain these real pilot banks. A synthetic
        # controller test must not import the user's live experiment history.
        history = patch("autosim.research.controller.import_known_seeds")
        history.start()
        self.addCleanup(history.stop)
        class FixtureRuntime(Runtime):
            def prepare_data(self, spec, root, output):
                return {"passed": True}

            def collect(self, spec, destination, **kwargs):
                root = destination / "data"
                atomic_json(root / "meta/info.json", {"total_episodes": kwargs["episodes"], "total_frames": 100})
                return root

            def train(self, spec, root, output, **kwargs):
                self.train_calls.append(kwargs)
                return output / "train/checkpoints" / str(kwargs["steps"]) / "pretrained_model"

            def evaluate(self, spec, checkpoint, output, **kwargs):
                from autosim.robosyn_data import evaluation_seed_bank
                count = kwargs["episodes"]
                successes = 0 if "controlled_baseline" in checkpoint.parts else 1
                result = {"execution_mode": "real_simulation", "test_fixture_only": True,
                          "harness": "synthetic_unit_fixture", "purpose": kwargs["purpose"],
                          "config": {"task": spec.name, "setting": "random", "timeout_action_steps": 361,
                                     "seed": kwargs["master_seed"]},
                          "summary": {"episode_count": count, "success_count": successes,
                                      "success_rate": successes / count, "average_action_steps": 300},
                          "episodes": [{"episode_seed": seed, "success": index < successes}
                                       for index, seed in enumerate(evaluation_seed_bank(kwargs["master_seed"], count))]}
                atomic_json(output / "evaluation_metrics.json", result)
                return result

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            runtime = FixtureRuntime(REPO.parents[1], output)
            runtime.train_calls = []
            dataset = output / "fixture_dataset"
            atomic_json(dataset / "meta/info.json", {"total_episodes": 10, "total_frames": 100})
            atomic_json(output / "smoke_status.json", {"tasks": {"click_bell": {"status": "passed", "dataset": str(dataset)}}})
            atomic_json(output / "task_inventory.json", [{"task": "click_bell", "checkpoint_available": False}])
            controller = ResearchController(runtime, ResearchConfig(pilot=True, screen_steps=2, final_steps=4,
                development_episodes=3, confirmation_episodes=3, collect_per_round=2, hours_per_task=1))
            state = controller.run_task("click_bell")
            self.assertEqual(state["status"], "completed_development", state.get("error"))
            self.assertEqual(state["completed_rounds"], 2)
            self.assertEqual(len(runtime.train_calls), 7)
            self.assertEqual(sum(call.get("resume", False) for call in runtime.train_calls), 2)
            self.assertEqual(state["improvement"], "not_established")
            self.assertNotEqual(state["rounds"][0]["proposal_signature"], state["rounds"][1]["proposal_signature"])
            self.assertEqual(controller.run_task("click_bell"), state)
            self.assertEqual(len(runtime.train_calls), 7)
            controller.ledger.close()

            fallback_output = output / "fallback"
            fallback = FixtureRuntime(REPO.parents[1], fallback_output)
            fallback.train_calls = []
            fallback.collect = Mock(side_effect=AssertionError("fallback must not call unavailable expert"))
            atomic_json(fallback_output / "smoke_status.json", {"tasks": {"click_bell": {"status": "failed"}}})
            atomic_json(fallback_output / "official_policy_smoke_status.json", {"tasks": {"click_bell": {"status": "passed"}}})
            atomic_json(fallback_output / "task_inventory.json", [{"task": "click_bell", "checkpoint_available": False,
                        "dataset_available": True, "official_dataset": str(dataset)}])
            fallback_controller = ResearchController(fallback, controller.config)
            result = fallback_controller.run_task("click_bell")
            self.assertEqual(result["status"], "completed_development", result.get("error"))
            self.assertEqual(result["research_mode"], "official_data_only_fallback")
            self.assertFalse(result["automatic_collection_validated"])
            fallback.collect.assert_not_called()
            self.assertTrue(all(not row["collection_executed"] for row in result["rounds"]))
            fallback_controller.ledger.close()


if __name__ == "__main__":
    unittest.main()
