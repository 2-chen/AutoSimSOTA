import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from autosim.experiment_validation.robosyn_milestone import (
    integration_ready,
    integration_terminal,
    reserve_development_banks,
    validate_baseline_sidecars,
    validate_baseline_metrics,
    validate_nested_evaluation_process,
)
from autosim.research.common import atomic_json, digest
from autosim.robosyn_data import evaluation_seed_bank
from autosim.experiment_validation.paired_initialization_audit import evaluation_artifact_directory


class RoboSynMilestoneTest(unittest.TestCase):
    def test_integration_gate_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            self.assertEqual(integration_terminal(path), (False, "missing"))
            atomic_json(path, {"status": "running"})
            self.assertEqual(integration_terminal(path), (False, "running"))
            for status in ("integration_gate_reached", "requires_capability_review", "queue_wait_budget_exhausted"):
                atomic_json(path, {"status": status})
                self.assertEqual(integration_terminal(path), (True, status))

    def test_only_successful_integration_opens_baseline_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_json(path, {"status": "running"})
            self.assertEqual(integration_ready(path), (False, False, "running"))
            atomic_json(path, {"status": "requires_capability_review"})
            self.assertEqual(integration_ready(path), (True, False, "requires_capability_review"))
            atomic_json(path, {"status": "integration_gate_reached"})
            self.assertEqual(integration_ready(path), (True, True, "integration_gate_reached"))

    def baseline_metrics(self, checkpoint: Path, *, task="click_bell", episodes=3, master=17):
        seeds = evaluation_seed_bank(master, episodes)
        rows = [{"episode_index": index, "episode_seed": seed, "success": index == 0,
                 "action_steps": 20 if index == 0 else 361} for index, seed in enumerate(seeds)]
        return {"execution_mode": "real_simulation", "purpose": "development",
                "harness": "official_control_loop_observation_only_rpc_v1",
                "config": {"task": task, "setting": "random", "episode_count": episodes,
                           "timeout_action_steps": 361, "seed": master,
                           "checkpoint_path": str(checkpoint)},
                "episodes": rows,
                "summary": {"episode_count": episodes, "success_count": 1,
                            "success_rate": 1 / episodes}}

    def test_baseline_metrics_require_exact_reserved_ordered_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            metrics = self.baseline_metrics(checkpoint)
            result = validate_baseline_metrics(metrics, task="click_bell", episodes=3,
                                               master_seed=17, max_episode_steps=361,
                                               checkpoint=checkpoint)
            self.assertTrue(result["ordered_seed_bank_verified"])
            metrics["episodes"].reverse()
            with self.assertRaisesRegex(ValueError, "pre-reserved ordered bank"):
                validate_baseline_metrics(metrics, task="click_bell", episodes=3,
                                          master_seed=17, max_episode_steps=361,
                                          checkpoint=checkpoint)

    def test_baseline_metrics_reject_summary_or_purpose_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            metrics = self.baseline_metrics(checkpoint)
            metrics["summary"]["success_count"] = 2
            with self.assertRaisesRegex(ValueError, "summary"):
                validate_baseline_metrics(metrics, task="click_bell", episodes=3,
                                          master_seed=17, max_episode_steps=361,
                                          checkpoint=checkpoint)
            metrics = self.baseline_metrics(checkpoint)
            metrics["purpose"] = "final"
            with self.assertRaisesRegex(ValueError, "non-development"):
                validate_baseline_metrics(metrics, task="click_bell", episodes=3,
                                          master_seed=17, max_episode_steps=361,
                                          checkpoint=checkpoint)

    def test_baseline_sidecars_corroborate_every_reset_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, checkpoint = Path(directory), Path(directory) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"test-fixture-model")
            seeds = evaluation_seed_bank(17, 3)
            atomic_json(root / "protocol.json", {"seed": 17, "episodes": 3,
                "purpose": "development", "policy": "act", "task": {"name": "click_bell"},
                "checkpoint_sha256": digest(checkpoint / "model.safetensors")})
            with (root / "initializations.jsonl").open("w") as stream:
                for seed in seeds:
                    stream.write(json.dumps({"event": "reset", "seed": seed,
                                             "allowed_observation_sha256": f"fixture-{seed}"}) + "\n")
            with (root / "telemetry.jsonl").open("w") as stream:
                for seed in seeds:
                    stream.write(json.dumps({"seed": seed, "step": 0, "task": "click_bell",
                                             "missing": []}) + "\n")
            result = validate_baseline_sidecars(root, task="click_bell", episodes=3,
                                                master_seed=17, checkpoint=checkpoint)
            self.assertEqual(result["reset_records_verified"], 3)
            rows = (root / "telemetry.jsonl").read_text().splitlines()
            (root / "telemetry.jsonl").write_text("\n".join(rows[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "step-0 telemetry"):
                validate_baseline_sidecars(root, task="click_bell", episodes=3,
                                           master_seed=17, checkpoint=checkpoint)

    def test_baseline_artifact_lineage_allows_only_startup_retry_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation"
            retry = root / "startup_attempt_3"
            retry.mkdir(parents=True)
            self.assertEqual(evaluation_artifact_directory(
                root, {"artifact_directory": str(retry)}), retry.resolve())
            with self.assertRaisesRegex(ValueError, "outside the allowed"):
                evaluation_artifact_directory(
                    root, {"artifact_directory": str(Path(directory) / "different_run")})

    def test_nested_evaluation_process_must_be_real_completed_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "process.json"
            atomic_json(path, {"status": "completed", "returncode": 0,
                               "elapsed_seconds": 12.5,
                               "command": ["python", "-m", "autosim.research.evaluation"]})
            result = validate_nested_evaluation_process(path)
            self.assertTrue(result["nested_evaluation_process_verified"])
            atomic_json(path, {"status": "completed", "returncode": 1,
                               "command": ["python", "-m", "autosim.research.evaluation"]})
            with self.assertRaisesRegex(ValueError, "did not complete"):
                validate_nested_evaluation_process(path)

    def test_all_ten_development_banks_are_global_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = reserve_development_banks(root, 100)
            second = reserve_development_banks(root, 100)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 10)
            db = sqlite3.connect(root / "seeds.sqlite")
            self.assertEqual(db.execute("select count(*) from banks where purpose='development'").fetchone()[0], 10)
            self.assertEqual(db.execute("select count(*) from seeds").fetchone()[0], 1000)
            self.assertEqual(db.execute("select count(*) from banks where purpose='final'").fetchone()[0], 0)
            db.close()

    def test_baseline_bank_overlap_fails_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reserve_development_banks(root, 100)
            with self.assertRaises(ValueError):
                reserve_development_banks(root, 101)


if __name__ == "__main__":
    unittest.main()
