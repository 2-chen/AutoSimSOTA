import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from autosim.experiment_validation.checkpoint_audit import audit_checkpoint
from autosim.research.common import atomic_json


class CheckpointAuditTest(unittest.TestCase):
    def fixture(self, root: Path, step=80000, seed=1000):
        arm = root / "round_1/auto"
        checkpoint = arm / f"train/checkpoints/{step:06d}/pretrained_model"
        atomic_json(checkpoint / "config.json", {"input_features": {
            "observation.state": {"type": "STATE", "shape": [14]},
            **{f"observation.images.camera_{index}": {"type": "VISUAL", "shape": [3, 480, 640]}
               for index in range(3)}}, "output_features": {"action": {"type": "ACTION", "shape": [14]}}})
        atomic_json(checkpoint / "train_config.json", {"steps": step, "seed": seed})
        state = checkpoint.parent / "training_state"
        atomic_json(state / "training_step.json", {"step": step})
        atomic_json(state / "optimizer_param_groups.json", [{"lr": 1e-5}])
        checkpoint.mkdir(parents=True, exist_ok=True)
        save_file({"weight": np.ones((2, 3), dtype=np.float32)}, checkpoint / "model.safetensors")
        save_file({"moment": np.ones((2, 3), dtype=np.float32)}, state / "optimizer_state.safetensors")
        save_file({"torch_rng_state": np.ones((8,), dtype=np.uint8)}, state / "rng_state.safetensors")
        process = arm / f"process_train_{step}/process.json"
        atomic_json(process, {"returncode": 0, "command": ["train.py", "--steps", str(step), "--seed", str(seed)]})
        last = checkpoint.parents[1] / "last"
        last.symlink_to(checkpoint.parent.name)
        return checkpoint

    def test_complete_checkpoint_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_checkpoint(self.fixture(Path(directory)), 80000, 1000)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["tensor_count"], 1)
            self.assertEqual(result["parameter_elements"], 6)
            self.assertEqual(result["training_state"]["optimizer_state_header"]["tensor_count"], 1)
            self.assertEqual(result["training_state"]["rng_state_header"]["elements"], 8)

    def test_wrong_step_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_checkpoint(self.fixture(Path(directory), step=80000), 60000, 1000)
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
