import json
import tempfile
import unittest
from pathlib import Path

from autosim.experiment_validation.research_comparison import gate_status, one_sided_exact, source_manifest, write_report
from autosim.research.common import atomic_json
from autosim.robosyn_data import evaluation_seed_bank


class ResearchComparisonTest(unittest.TestCase):
    def test_exact_one_sided_sign_test(self):
        self.assertEqual(one_sided_exact(0, 0), 1.0)
        self.assertEqual(one_sided_exact(3, 0), 0.125)
        self.assertEqual(one_sided_exact(0, 3), 1.0)

    def test_gate_requires_both_completed_upstreams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            click, gap = root / "click.json", root / "gap.json"
            self.assertEqual(gate_status(click, gap)[0], "waiting_click_bell")
            atomic_json(click, {"status": "completed_development"})
            self.assertEqual(gate_status(click, gap)[0], "waiting_collection_gap")
            atomic_json(gap, {"status": "collection_gap_handled"})
            self.assertEqual(gate_status(click, gap)[0], "ready")

    def test_failed_upstream_never_starts_research(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            click, gap = root / "click.json", root / "gap.json"
            atomic_json(click, {"status": "failed"})
            atomic_json(gap, {"status": "collection_gap_handled"})
            self.assertEqual(gate_status(click, gap)[0], "requires_review")

    def test_frozen_comparison_sources_include_numeric_initialization_audit(self):
        paths = source_manifest()
        self.assertTrue(any(path.endswith("paired_initialization_audit.py") for path in paths))

    def test_effect_gate_requires_and_publishes_confirmation_initialization_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            research, report = root / "research", root / "report"
            atomic_json(research / "research_config.json", {
                "confirmation_episodes": 3, "train_seed": 1000, "pilot": False})
            for task_index, task in enumerate(("click_bell", "water_pouring", "drawer_open_place")):
                evaluations = research / task / "evaluations"
                paths = {}
                checkpoints = {}
                master_seed = 80_000_000 + task_index * 1_000_000 + 2
                for label, successes, offset in (
                    ("controlled_baseline", (False, False, False), 0.0),
                    ("auto_selected", (True, True, True), 0.0001),
                    ("random_selected", (False, False, False), -0.0001),
                ):
                    path = evaluations / f"{label}_confirmation"
                    path.mkdir(parents=True)
                    checkpoint = research / task / "checkpoints" / label
                    checkpoint.mkdir(parents=True)
                    (checkpoint / "model.safetensors").write_bytes(f"fixture-{task}-{label}".encode())
                    atomic_json(checkpoint / "config.json", {"fixture": True})
                    checkpoints[label] = checkpoint
                    seeds = evaluation_seed_bank(master_seed, 3)
                    episodes = [{"episode_index": index, "episode_seed": seed, "success": success,
                                 "action_steps": 10 if success else 361}
                                for index, (seed, success) in enumerate(zip(seeds, successes))]
                    metrics = {"test_fixture_only": True, "execution_mode": "real_simulation",
                               "purpose": "confirmation", "harness": "test_fixture_only",
                               "config": {"task": task, "setting": "random", "timeout_action_steps": 361,
                                          "episode_count": 3, "seed": master_seed,
                                          "checkpoint_path": str(checkpoint)},
                               "episodes": episodes,
                               "summary": {"episode_count": 3, "success_count": sum(successes),
                                           "success_rate": sum(successes) / 3, "average_action_steps": 100}}
                    metrics["harness"] = "official_control_loop_observation_only_rpc_v1"
                    metrics["artifact_directory"] = str(path)
                    metrics["startup_attempt_count"] = 1
                    atomic_json(path / "evaluation_metrics.json", metrics)
                    from autosim.research.common import digest
                    atomic_json(path / "evaluation_request.json", {
                        "checkpoint": str(checkpoint.resolve()),
                        "weight_sha256": digest(checkpoint / "model.safetensors"),
                        "model_config_sha256": digest(checkpoint / "config.json"),
                        "episodes": 3, "master_seed": master_seed,
                        "purpose": "confirmation", "policy": "act"})
                    atomic_json(path / "protocol.json", {
                        "task": {"name": task}, "purpose": "confirmation",
                        "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
                        "seed": master_seed, "episodes": 3, "policy": "act",
                        "frozen_files": {str((checkpoint / "config.json").resolve()):
                                         digest(checkpoint / "config.json")}})
                    atomic_json(path / "process/process.json", {
                        "status": "completed", "returncode": 0,
                        "command": ["python", "-m", "autosim.research.evaluation",
                                    "--task", task, "--checkpoint", str(checkpoint.resolve()),
                                    "--output", str(path.resolve()), "--episodes", "3",
                                    "--seed", str(master_seed), "--purpose", "confirmation",
                                    "--policy", "act"]})
                    with (path / "initializations.jsonl").open("w") as stream:
                        for seed in seeds:
                            stream.write(json.dumps({"seed": seed,
                                "allowed_observation_sha256": f"fixture-{seed}-{offset}"}) + "\n")
                    with (path / "telemetry.jsonl").open("w") as stream:
                        for seed in seeds:
                            stream.write(json.dumps({"seed": seed, "step": 0,
                                "robot_qpos": [[offset, 0.0]], "entities": {}}) + "\n")
                    paths[label] = path
                state = {"task": task, "status": "completed_development", "automatic_collection_validated": True,
                         "completed_rounds": 2, "research_mode": "test_fixture_only",
                         "baseline_checkpoint": str(checkpoints["controlled_baseline"]),
                         "selected": {
                             "auto": {"evaluation": str(paths["auto_selected"]),
                                      "checkpoint": str(checkpoints["auto_selected"]),
                                      "summary": {"success_count": 0, "success_rate": 0.0,
                                                  "test_fixture_stale_cache": True},
                                      "vs_controlled_baseline": {"success_delta": 1.0}},
                             "random": {"evaluation": str(paths["random_selected"]),
                                        "checkpoint": str(checkpoints["random_selected"]),
                                        "summary": {"success_count": 3, "success_rate": 1.0,
                                                    "test_fixture_stale_cache": True}},
                         }}
                atomic_json(research / task / "state.json", state)
            result = write_report(report, research)
            self.assertTrue(result["study_execution_complete"])
            self.assertTrue(result["reset_pairing_evidence_complete"])
            self.assertTrue(result["automatic_decision_beats_random_control_established"])
            self.assertEqual(len(list((report / "initialization_audits").glob("*_confirmation.json"))), 3)
            for row in result["tasks"]:
                self.assertEqual(row["auto_success_count"], 3)
                self.assertEqual(row["auto_vs_fixed_training"]["candidate_only_successes"], 3)
                self.assertTrue(row["confirmation_identity_verified"])
            protocol_path = research / "click_bell/evaluations/auto_selected_confirmation/protocol.json"
            protocol = json.loads(protocol_path.read_text())
            protocol["checkpoint_sha256"] = "wrong-checkpoint"
            atomic_json(protocol_path, protocol)
            with self.assertRaisesRegex(ValueError, "expected checkpoint/seed bank"):
                write_report(report / "tampered", research)


if __name__ == "__main__":
    unittest.main()
