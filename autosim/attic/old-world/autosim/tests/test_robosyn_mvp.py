import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import yaml

from autosim.robosyn_mvp import (
    FrozenEvaluatorGuard,
    MVPConfig,
    RoboSynMVPRunner,
    _validate_lerobot_dataset,
    acceptance_decision,
    generate_recipes,
    metric_order,
    paired_success_comparison,
    route_diagnostic_proposals,
    route_feedback_sweep,
)
from autosim.robosyn_data import (
    audit_collection_manifests,
    evaluation_seed_bank,
    write_mixture_manifest,
)
from autosim.robosyn_selection import select_confirmed_candidate
from autosim.robosyn_finalize import evaluate_frozen_once


class RoboSynMVPTest(unittest.TestCase):
    def _config(self, root: Path) -> MVPConfig:
        repo = root / "repo"
        checkpoint = repo / "checkpoint"
        python = root / "python"
        for path in [
            repo / "scripts/eval_policy.py",
            repo / "policy/act/deploy_policy.yml",
            repo / "policy/act/deploy_policy.py",
            repo / "policy/inference_timing.py",
            repo / "robosynchallenge/tasks/click_bell/click_bell.py",
            repo / "configs/click_bell/random/gym_config.json",
            repo / "configs/click_bell/action_config.json",
            checkpoint / "config.json",
            checkpoint / "model.safetensors",
            python,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test")
        payload = {
            "name": "test",
            "repo": str(repo),
            "python": str(python),
            "task": "click_bell",
            "setting": "random",
            "baseline_checkpoint": str(checkpoint),
            "output_root": str(root / "output"),
            "train": {"chunk_size": 50, "n_action_steps": 50, "use_amp": False},
            "evaluation": {"max_trials": 2},
            "search_space": {
                "chunk_size": [32, 50],
                "n_action_steps": [32, 50],
                "use_amp": [False],
            },
        }
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(payload))
        return MVPConfig.load(config_path)

    def test_recipe_generation_is_bounded_and_skips_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            recipes = generate_recipes(config)
            self.assertEqual(len(recipes), 2)
            self.assertTrue(
                all(
                    recipe.params["chunk_size"] != 50
                    or recipe.params["n_action_steps"] != 50
                    for recipe in recipes
                )
            )

    def test_collection_seed_bank_is_reproducible_and_separate(self):
        evaluation = set(evaluation_seed_bank(0, 100))
        self.assertEqual(evaluation_seed_bank(0, 100), evaluation_seed_bank(0, 100))
        import numpy as np

        collection = np.random.RandomState(310001).randint(0, 2**31 - 1, 500)
        self.assertFalse(evaluation & set(int(value) for value in collection))

    def test_seed_audit_checks_development_and_frozen_banks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_seed = evaluation_seed_bank(910001, 40)[0]
            manifest = root / "collection.json"
            manifest.write_text(
                json.dumps(
                    {
                        "profile": "policy_correction",
                        "successful_episode_seeds": [collection_seed],
                        "failed_attempt_seeds": [],
                    }
                )
            )
            result = audit_collection_manifests(
                [manifest],
                excluded_seed_banks={"development_main": (910001, 40)},
            )
            self.assertFalse(result["passed"])
            self.assertEqual(
                result["evaluation_overlaps"][0]["bank"], "development_main"
            )

    def test_seed_audit_does_not_treat_frozen_alias_as_bank_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "collection.json"
            manifest.write_text(
                json.dumps(
                    {
                        "profile": "policy_correction",
                        "successful_episode_seeds": [123456789],
                        "failed_attempt_seeds": [],
                    }
                )
            )
            result = audit_collection_manifests(
                [manifest],
                eval_master_seed=1_039_001,
                eval_episodes=200,
                excluded_seed_banks={
                    "internal_frozen_v3": (1_039_001, 200),
                },
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["seed_bank_overlaps"], [])
            self.assertNotIn("frozen_final", result["excluded_seed_banks"])

    def test_mixture_manifest_preserves_official_dataset_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "official"
            targeted = root / "targeted"
            for path, episodes in ((official, 1000), (targeted, 100)):
                (path / "meta").mkdir(parents=True)
                (path / "meta/info.json").write_text(
                    json.dumps(
                        {"total_episodes": episodes, "total_frames": episodes * 74}
                    )
                )
            payload = write_mixture_manifest(
                root / "mixture.json",
                official,
                [f"targeted_clutter={targeted}"],
            )
            self.assertEqual(payload["datasets"][0]["role"], "official_full_random")
            self.assertEqual(payload["datasets"][1]["profile"], "targeted_clutter")

    def test_frozen_guard_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            guard = FrozenEvaluatorGuard(config.repo, config.frozen_files)
            (config.repo / "scripts/eval_policy.py").write_text("mutated")
            with self.assertRaises(RuntimeError):
                guard.assert_unchanged()

    def test_frozen_finalizer_refuses_a_second_query_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            manifest = root / "final_training_v2.json"
            state = root / "frozen_evaluation_state_v2.json"
            state.write_text(json.dumps({"status": "completed", "query_count": 1}))
            with self.assertRaisesRegex(RuntimeError, "already started or completed"):
                evaluate_frozen_once(
                    config,
                    manifest,
                    root / "missing_official.json",
                    root / "missing_targeted.json",
                )

    def test_runner_can_append_to_an_existing_audit_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            run_dir = root / "existing-run"
            run_dir.mkdir()
            runner = RoboSynMVPRunner(config, run_dir=run_dir)
            self.assertEqual(runner.run_dir, run_dir.resolve())
            self.assertEqual(runner.events.path, run_dir.resolve() / "events.jsonl")

    def test_evaluator_retries_startup_sigsegv_and_preserves_attempt_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            config.evaluation.update(
                {
                    "startup_retries": 1,
                    "startup_retry_delay_seconds": 0,
                    "startup_retry_exit_codes": [-11],
                }
            )
            run_dir = root / "run"
            run_dir.mkdir()
            runner = RoboSynMVPRunner(config, run_dir=run_dir)
            calls = []

            def fake_run(command, log_path, timeout):
                calls.append(log_path)
                if len(calls) == 1:
                    raise RuntimeError(f"command failed with exit code -11; see {log_path}")

            metric = root / "metric.json"
            metric.write_text(
                json.dumps(
                    {
                        "summary": {
                            "success_rate": 0.5,
                            "episode_count": 2,
                            "success_count": 1,
                        },
                        "episodes": [],
                        "config": {"seed": 7},
                    }
                )
            )
            runner._run_command = fake_run
            runner._new_metric_file = lambda before: metric
            with patch("autosim.robosyn_mvp.time.sleep"):
                result = runner.evaluate(config.baseline_checkpoint, 2, "candidate", seed=7)
            self.assertEqual(result["success_rate"], 0.5)
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[1].name.endswith("_attempt2.log"))

    def test_development_selector_uses_confirmation_not_earlier_rung(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, early, confirmation in (
                ("early-winner", 0.9, 0.5),
                ("confirmed-winner", 0.6, 0.7),
            ):
                path = root / name / "summary.json"
                path.parent.mkdir()
                path.write_text(
                    json.dumps(
                        {
                            "name": name,
                            "status": "completed",
                            "evaluation_protocol": "development",
                            "records": [
                                {
                                    "status": "completed",
                                    "rung": 1,
                                    "training_steps": 20_000,
                                    "checkpoint": f"/{name}/checkpoint",
                                    "metrics": {
                                        "episode_count": 40,
                                        "success_count": int(early * 40),
                                        "success_rate": early,
                                        "average_action_steps": 100,
                                        "evaluation_config": {"seed": 910001},
                                    },
                                },
                                {
                                    "status": "completed",
                                    "rung": 2,
                                    "training_steps": 20_000,
                                    "checkpoint": f"/{name}/checkpoint",
                                    "metrics": {
                                        "episode_count": 40,
                                        "success_count": int(confirmation * 40),
                                        "success_rate": confirmation,
                                        "average_action_steps": 100,
                                        "evaluation_config": {"seed": 920001},
                                    },
                                },
                            ],
                        }
                    )
                )
                paths.append(path)
            result = select_confirmed_candidate(paths)
            self.assertEqual(result["selected"]["experiment"], "confirmed-winner")
            self.assertEqual(result["confirmation_seed"], 920001)
            self.assertFalse(result["frozen_evaluation_performed"])

    def test_development_selector_rejects_frozen_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "evaluation_protocol": "frozen_final",
                        "records": [],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "development summaries only"):
                select_confirmed_candidate([path])

    def test_partial_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "videos/chunk-000/cam_high.color").mkdir(parents=True)
            (root / "meta/info.json").write_text(
                json.dumps(
                    {
                        "total_episodes": 2,
                        "features": {"cam_high.color": {"dtype": "video"}},
                    }
                )
            )
            (root / "data/chunk-000/episode_000000.parquet").touch()
            problems = _validate_lerobot_dataset(root)
            self.assertIn("incomplete dataset parquet files: 1/2", problems)
            self.assertIn(
                "incomplete dataset video cam_high.color: 0/2", problems
            )

    def test_metric_order_never_trades_success_for_speed(self):
        reliable = {
            "success_rate": 0.6,
            "average_action_steps": 350,
            "average_inference_time_per_episode_seconds": 1.0,
        }
        fast_but_worse = {
            "success_rate": 0.5,
            "average_action_steps": 1,
            "average_inference_time_per_episode_seconds": 0.001,
        }
        self.assertGreater(metric_order(reliable), metric_order(fast_but_worse))

    def test_paired_gate_rejects_tiny_or_underpowered_run(self):
        seeds = list(range(20))
        baseline = {
            "episode_count": 20,
            "success_rate": 0.5,
            "episodes": [
                {"episode_seed": seed, "success": seed < 10} for seed in seeds
            ],
        }
        candidate = {
            "episode_count": 20,
            "success_rate": 0.55,
            "episodes": [
                {"episode_seed": seed, "success": seed < 11} for seed in seeds
            ],
        }
        comparison = paired_success_comparison(candidate, baseline)
        self.assertEqual(comparison["candidate_only_successes"], 1)
        decision = acceptance_decision(
            candidate,
            baseline,
            {
                "acceptance_min_episodes": 100,
                "acceptance_min_success_delta": 0.03,
                "acceptance_max_p_value": 0.05,
            },
        )
        self.assertFalse(decision["accepted"])
        self.assertTrue(any("100 episodes" in item for item in decision["reasons"]))

    def test_diagnostic_result_can_never_be_accepted(self):
        episodes = [
            {"episode_seed": seed, "success": seed < 90} for seed in range(100)
        ]
        baseline = {
            "episode_count": 100,
            "success_rate": 0.5,
            "episodes": [
                {"episode_seed": seed, "success": seed < 50}
                for seed in range(100)
            ],
        }
        diagnostic = {
            "episode_count": 100,
            "success_rate": 0.9,
            "episodes": episodes,
            "evaluation_config": {"diagnostic_profile": "appearance"},
        }
        decision = acceptance_decision(
            diagnostic,
            baseline,
            {
                "acceptance_min_episodes": 100,
                "acceptance_min_success_delta": 0.03,
                "acceptance_max_p_value": 0.05,
            },
        )
        self.assertFalse(decision["accepted"])
        self.assertIn(
            "diagnostic slices are not ranking eligible", decision["reasons"]
        )

    def test_development_seed_result_can_promote_but_never_accept(self):
        episodes = [
            {"episode_seed": seed, "success": seed < 90} for seed in range(100)
        ]
        candidate = {"episode_count": 100, "success_rate": 0.9, "episodes": episodes}
        baseline = {
            "episode_count": 100,
            "success_rate": 0.5,
            "episodes": [
                {"episode_seed": seed, "success": seed < 50} for seed in range(100)
            ],
        }
        decision = acceptance_decision(
            candidate,
            baseline,
            {
                "protocol": "development",
                "acceptance_min_episodes": 100,
                "acceptance_min_success_delta": 0.03,
                "acceptance_max_p_value": 0.05,
            },
        )
        self.assertFalse(decision["accepted"])
        self.assertIn("development seed banks", decision["reasons"][0])

    def test_diagnostic_router_prioritizes_hard_factor(self):
        results = {
            "appearance": {"success_rate": 0.55, "average_action_steps": 197},
            "camera": {"success_rate": 0.70, "average_action_steps": 141},
            "robot_pose": {"success_rate": 0.55, "average_action_steps": 199},
            "clutter": {"success_rate": 0.50, "average_action_steps": 209},
        }
        routed = route_diagnostic_proposals(results)
        self.assertEqual(routed["factor_priority"][0], "clutter")
        self.assertIn("clutter_occlusion_mild", routed["implemented_next"])

    def test_feedback_router_keeps_successful_long_queue(self):
        results = {
            "8": {"success_rate": 0.2, "average_action_steps": 300},
            "50": {"success_rate": 0.55, "average_action_steps": 189},
        }
        routed = route_feedback_sweep(results)
        self.assertEqual(routed["selected_n_action_steps"], 50)
        self.assertFalse(routed["retrain_required"])


if __name__ == "__main__":
    unittest.main()
