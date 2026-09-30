import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from autosim.experiment_validation.dataset_video_integrity_v1 import audit_dataset
from autosim.research.common import atomic_json


def write_video(path: Path, frames: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.full((16, 16, 3), index * 20, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class DatasetVideoIntegrityV1Test(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        dataset = root / "dataset"
        atomic_json(dataset / "meta/info.json", {
            "total_episodes": 2,
            "total_videos": 4,
            "chunks_size": 1000,
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.a": {"dtype": "video", "shape": [16, 16, 3]},
                "observation.images.b": {"dtype": "video", "shape": [16, 16, 3]},
            },
        })
        episodes = [
            {"episode_index": 0, "length": 3, "tasks": ["test"]},
            {"episode_index": 1, "length": 3, "tasks": ["test"]},
        ]
        path = dataset / "meta/episodes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8")
        for episode in range(2):
            for camera in ("a", "b"):
                write_video(dataset / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4")
        return dataset

    def test_full_decode_passes_complete_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_dataset(self.fixture(Path(directory)), expected_episodes=2,
                                   expected_video_keys=["observation.images.a", "observation.images.b"], workers=2)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["decoded_video_count"], 4)
            self.assertEqual(result["decoded_frame_count"], 12)

    def test_corrupt_video_rejects_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.fixture(Path(directory))
            broken = dataset / "videos/chunk-000/observation.images.a/episode_000001.mp4"
            broken.write_bytes(broken.read_bytes()[:64])
            result = audit_dataset(dataset, expected_episodes=2, workers=2)
            self.assertFalse(result["passed"])
            self.assertEqual(result["training_admission"], "rejected")
            self.assertEqual(len(result["failed_videos"]), 1)

    def test_missing_video_rejects_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.fixture(Path(directory))
            (dataset / "videos/chunk-000/observation.images.b/episode_000000.mp4").unlink()
            result = audit_dataset(dataset, expected_episodes=2, workers=2)
            self.assertFalse(result["passed"])
            file_check = next(row for row in result["checks"] if row["name"] == "video_file_set")
            self.assertEqual(len(file_check["missing"]), 1)


if __name__ == "__main__":
    unittest.main()
