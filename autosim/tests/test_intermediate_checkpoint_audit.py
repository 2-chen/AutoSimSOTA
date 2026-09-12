import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from autosim.experiment_validation.intermediate_checkpoint_audit import audit_intermediate_checkpoint
from autosim.research.common import atomic_json


class IntermediateCheckpointAuditTest(unittest.TestCase):
    def fixture(self, root: Path, *, process_status="running"):
        checkpoint = root / "round_1/random/train/checkpoints/060000/pretrained_model"
        state = checkpoint.parent / "training_state"
        atomic_json(checkpoint / "config.json", {"input_features": {
            "observation.state": {"type": "STATE", "shape": [14]},
            **{f"observation.images.camera_{index}": {"type": "VISUAL", "shape": [3, 480, 640]}
               for index in range(3)}}, "output_features": {"action": {"type": "ACTION", "shape": [14]}}})
        atomic_json(checkpoint / "train_config.json", {"steps": 80000, "seed": 1000})
        atomic_json(state / "training_step.json", {"step": 60000})
        atomic_json(state / "optimizer_param_groups.json", [{"lr": 1e-5}])
        checkpoint.mkdir(parents=True, exist_ok=True)
        save_file({"weight": np.ones((2, 3), dtype=np.float32)}, checkpoint / "model.safetensors")
        save_file({"moment": np.ones((2, 3), dtype=np.float32)}, state / "optimizer_state.safetensors")
        save_file({"torch_rng_state": np.ones((8,), dtype=np.uint8)}, state / "rng_state.safetensors")
        process = root / "round_1/random/process_train_80000/process.json"
        atomic_json(process, {"status": process_status, "returncode": None, "pid": os.getpid(),
                              "command": ["train.py", "--steps", "80000", "--seed", "1000", "--resume"]})
        (checkpoint.parents[1] / "last").symlink_to(checkpoint.parent.name)
        return checkpoint, process

    def test_running_intermediate_checkpoint_passes_without_completion_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, process = self.fixture(Path(directory))
            result = audit_intermediate_checkpoint(
                checkpoint, process, expected_serialized_step=60000,
                expected_target_step=80000, expected_seed=1000, require_resume=True)
            self.assertTrue(result["passed"], result)
            self.assertFalse(result["training_complete_claim"])
            self.assertEqual(result["status"], "intermediate_serialization_passed")

    def test_completed_process_is_not_an_in_progress_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, process = self.fixture(Path(directory), process_status="completed")
            result = audit_intermediate_checkpoint(
                checkpoint, process, expected_serialized_step=60000,
                expected_target_step=80000, expected_seed=1000, require_resume=True)
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
