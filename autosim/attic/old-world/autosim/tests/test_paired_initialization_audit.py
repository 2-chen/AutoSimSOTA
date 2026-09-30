import json
from pathlib import Path
import tempfile
import unittest

from autosim.experiment_validation.paired_initialization_audit import audit_bank
from autosim.research.common import atomic_json


class PairedInitializationAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def evaluation(self, label, robot_offset=0.0, button_offset=0.0, seeds=(11, 12), successes=(False, True)):
        directory = self.root / label
        directory.mkdir()
        episodes = [{"episode_seed": seed, "success": success, "action_steps": 361}
                    for seed, success in zip(seeds, successes)]
        metrics = {
            "config": {"task": "click_bell", "setting": "random", "timeout_action_steps": 361},
            "purpose": "development", "episodes": episodes,
        }
        atomic_json(directory / "evaluation_metrics.json", metrics)
        atomic_json(directory / "protocol.json", {"fixture": True})
        with (directory / "initializations.jsonl").open("w") as stream:
            for seed in seeds:
                stream.write(json.dumps({"seed": seed, "allowed_observation_sha256": f"hash-{seed}-{robot_offset}"}) + "\n")
        pose = [[[1, 0, 0, button_offset], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
        with (directory / "telemetry.jsonl").open("w") as stream:
            for seed in seeds:
                stream.write(json.dumps({"seed": seed, "step": 0,
                    "robot_qpos": [[0.0, robot_offset]],
                    "entities": {"button": {"pose": pose, "qpos": [[button_offset]]}}}) + "\n")
        return directory

    def test_quantifies_multi_policy_reset_variation_without_pass_tolerance(self):
        result = audit_bank({
            "a": self.evaluation("a"),
            "b": self.evaluation("b", robot_offset=0.001, button_offset=0.002,
                                 successes=(True, True)),
            "c": self.evaluation("c", robot_offset=-0.001, button_offset=-0.001),
        })
        summary = result["summary"]
        self.assertEqual(summary["evaluation_count"], 3)
        self.assertEqual(summary["episode_count"], 2)
        self.assertAlmostEqual(summary["robot_qpos_max_component_range_rad"]["maximum"], 0.002)
        self.assertAlmostEqual(summary["entities"]["button"]["translation_max_pairwise_difference_m"]["maximum"], 0.003)
        self.assertEqual(summary["outcome_discordant_episode_count"], 1)
        self.assertEqual(result["interpretation"]["pass_tolerance"], "not_predeclared_no_pass_claim")

    def test_rejects_reordered_seed_bank(self):
        first = self.evaluation("a")
        second = self.evaluation("b", seeds=(12, 11))
        with self.assertRaisesRegex(ValueError, "ordered seed bank"):
            audit_bank({"a": first, "b": second})

    def test_requires_complete_step_zero_telemetry(self):
        first = self.evaluation("a")
        second = self.evaluation("b")
        rows = (second / "telemetry.jsonl").read_text().splitlines()
        (second / "telemetry.jsonl").write_text(rows[0] + "\n")
        with self.assertRaisesRegex(ValueError, "step-0 telemetry"):
            audit_bank({"a": first, "b": second})

    def test_follows_only_allowed_startup_retry_sidecars(self):
        first = self.evaluation("a")
        second = self.evaluation("b")
        retry = second / "startup_attempt_2"
        retry.mkdir()
        for name in ("protocol.json", "initializations.jsonl", "telemetry.jsonl"):
            (second / name).replace(retry / name)
        metrics = json.loads((second / "evaluation_metrics.json").read_text())
        metrics["artifact_directory"] = str(retry)
        atomic_json(second / "evaluation_metrics.json", metrics)
        result = audit_bank({"a": first, "b": second})
        self.assertEqual(result["evaluations"][1]["artifact_directory"], str(retry.resolve()))
        metrics["artifact_directory"] = str(self.root / "unrelated")
        atomic_json(second / "evaluation_metrics.json", metrics)
        with self.assertRaisesRegex(ValueError, "outside the allowed"):
            audit_bank({"a": first, "b": second})


if __name__ == "__main__":
    unittest.main()
