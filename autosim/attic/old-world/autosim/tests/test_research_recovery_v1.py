import json
import tempfile
import unittest
from pathlib import Path

from autosim.experiment_validation.research_recovery_v1 import verify_reused_evaluation
from autosim.research.common import atomic_json, digest
from autosim.robosyn_data import evaluation_seed_bank


class ResearchRecoveryV1Test(unittest.TestCase):
    def fixture(self, root: Path):
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        evaluation = root / "evaluation"
        seed, count = 82000001, 3
        atomic_json(evaluation / "evaluation_request.json", {
            "checkpoint": str(checkpoint),
            "weight_sha256": digest(checkpoint / "model.safetensors"),
        })
        atomic_json(evaluation / "evaluation_metrics.json", {
            "execution_mode": "real_simulation", "purpose": "development",
            "config": {"task": "water_pouring"},
            "episodes": [{"episode_seed": item, "success": False}
                         for item in evaluation_seed_bank(seed, count)],
        })
        return evaluation, checkpoint, seed, count

    def test_exact_reuse_contract_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation, checkpoint, seed, count = self.fixture(Path(directory))
            result = verify_reused_evaluation(evaluation, checkpoint, task="water_pouring",
                                              episodes=count, master_seed=seed,
                                              purpose="development")
            self.assertTrue(all(result["checks"].values()))

    def test_changed_seed_bank_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation, checkpoint, seed, count = self.fixture(Path(directory))
            with self.assertRaises(RuntimeError):
                verify_reused_evaluation(evaluation, checkpoint, task="water_pouring",
                                         episodes=count, master_seed=seed + 1,
                                         purpose="development")


if __name__ == "__main__":
    unittest.main()
