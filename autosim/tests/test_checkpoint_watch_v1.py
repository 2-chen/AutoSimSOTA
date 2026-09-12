import tempfile
import unittest
from pathlib import Path

from autosim.experiment_validation.checkpoint_watch_v1 import checkpoint_path, intermediate_ready


class CheckpointWatchV1Test(unittest.TestCase):
    def fixture(self, root: Path):
        checkpoint = checkpoint_path(root / "arm", 20000)
        step = checkpoint.parent
        for path in (
            checkpoint / "config.json", checkpoint / "train_config.json",
            checkpoint / "model.safetensors", step / "training_state/training_step.json",
            step / "training_state/optimizer_param_groups.json",
            step / "training_state/optimizer_state.safetensors",
            step / "training_state/rng_state.safetensors",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        return checkpoint

    def test_ready_requires_complete_files_and_matching_last_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.fixture(Path(directory))
            self.assertFalse(intermediate_ready(checkpoint))
            (checkpoint.parents[1] / "last").symlink_to(checkpoint.parent.name)
            self.assertTrue(intermediate_ready(checkpoint))

    def test_missing_state_file_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.fixture(Path(directory))
            (checkpoint.parents[1] / "last").symlink_to(checkpoint.parent.name)
            (checkpoint.parent / "training_state/rng_state.safetensors").unlink()
            self.assertFalse(intermediate_ready(checkpoint))


if __name__ == "__main__":
    unittest.main()
