import importlib.util
import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


TRAIN_SCRIPT = (
    Path(__file__).parents[2]
    / "RoboSynChallenge/policy/act/scripts/train.py"
)
if os.environ.get("AUTOSIM_ROBOSYN_REPO"):
    TRAIN_SCRIPT = Path(os.environ["AUTOSIM_ROBOSYN_REPO"]) / "policy/act/scripts/train.py"
if not TRAIN_SCRIPT.is_file():
    raise unittest.SkipTest("External RoboSyn integration: set AUTOSIM_ROBOSYN_REPO")
SPEC = importlib.util.spec_from_file_location("robosyn_act_train", TRAIN_SCRIPT)
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def _stats(mean, std, count, minimum=None, maximum=None):
    mean = np.asarray(mean, dtype=np.float32)
    return {
        "mean": mean,
        "std": np.asarray(std, dtype=np.float32),
        "count": np.asarray([count], dtype=np.int64),
        "min": np.asarray(minimum if minimum is not None else mean, dtype=np.float32),
        "max": np.asarray(maximum if maximum is not None else mean, dtype=np.float32),
    }


class _FakeDataset:
    def __init__(self, episode_lengths):
        starts = np.cumsum([0] + list(episode_lengths[:-1]))
        stops = np.cumsum(episode_lengths)
        self.episode_data_index = {
            "from": torch.as_tensor(starts),
            "to": torch.as_tensor(stops),
        }
        self._length = int(sum(episode_lengths))

    def __len__(self):
        return self._length


class RoboSynDataPipelineTest(unittest.TestCase):
    def test_population_statistics_are_merged_by_count(self):
        first = SimpleNamespace(
            stats={"action": _stats([0.0], [1.0], 3, [-1.0], [1.0])}
        )
        second = SimpleNamespace(
            stats={"action": _stats([4.0], [2.0], 1, [2.0], [7.0])}
        )
        merged = TRAIN._merge_feature_statistics([first, second])["action"]
        self.assertAlmostEqual(float(merged["mean"][0]), 1.0)
        # E[x^2] = (3*(1+0) + 1*(4+16))/4 = 5.75; Var = 4.75.
        self.assertAlmostEqual(float(merged["std"][0]), np.sqrt(4.75), places=6)
        self.assertEqual(int(merged["count"][0]), 4)
        self.assertEqual(float(merged["min"][0]), -1.0)
        self.assertEqual(float(merged["max"][0]), 7.0)

    def test_stratified_sampler_preserves_requested_profile_mass(self):
        datasets = [_FakeDataset([4, 4]), _FakeDataset([4])]
        entries = [
            {"profile": "full_random", "root": "/official"},
            {"profile": "correction", "root": "/correction"},
        ]
        sampling = {
            "strategy": "stratified_phase",
            "profile_masses": {"full_random": 0.7, "correction": 0.3},
            "phase_bins": [
                {"name": "early", "start": 0.0, "end": 0.5, "weight": 1.0},
                {"name": "late", "start": 0.5, "end": 1.01, "weight": 2.0},
            ],
        }
        weights, audit = TRAIN._build_stratified_sample_weights(
            datasets, entries, sampling
        )
        self.assertAlmostEqual(float(weights[:8].sum()), 0.7)
        self.assertAlmostEqual(float(weights[8:].sum()), 0.3)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(float(weights[2]), float(weights[0]))
        self.assertEqual(audit[1]["phase_frame_counts"], {"early": 2, "late": 2})

    def test_sampling_bins_must_cover_full_episode(self):
        with self.assertRaisesRegex(ValueError, "do not cover"):
            TRAIN._build_stratified_sample_weights(
                [_FakeDataset([4])],
                [{"profile": "x", "root": "/x"}],
                {
                    "strategy": "stratified_phase",
                    "phase_bins": [
                        {"name": "early", "start": 0.0, "end": 0.5, "weight": 1.0}
                    ],
                },
            )

    def test_duplicate_profile_shards_share_aggregate_mass(self):
        datasets = [_FakeDataset([10]), _FakeDataset([4]), _FakeDataset([6])]
        entries = [
            {"profile": "official", "root": "/official"},
            {"profile": "correction", "root": "/early"},
            {"profile": "correction", "root": "/late"},
        ]
        weights, _ = TRAIN._build_stratified_sample_weights(
            datasets,
            entries,
            {
                "profile_masses": {"official": 0.7, "correction": 0.3},
                "phase_bins": [
                    {"name": "all", "start": 0.0, "end": 1.01, "weight": 1.0}
                ],
            },
        )
        self.assertAlmostEqual(float(weights[:10].sum()), 0.7)
        self.assertAlmostEqual(float(weights[10:14].sum()), 0.12)
        self.assertAlmostEqual(float(weights[14:].sum()), 0.18)

    def test_exposure_audit_counts_batches_and_removes_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exposure.json"
            audit = TRAIN._ExposureAudit(
                path,
                [
                    {"root": "/official", "profile": "official", "source_kind": "official"},
                    {"root": "/new", "profile": "camera", "source_kind": "self_collected"},
                ],
                [8, 2],
                [0.8, 0.2],
                flush_batches=1,
            )
            batches = [
                {"x": torch.ones(3), "__autosim_source_id__": torch.tensor([0, 0, 1])},
                {"x": torch.ones(2), "__autosim_source_id__": torch.tensor([1, 0])},
            ]
            observed = list(TRAIN._ExposureTrackingLoader(batches, audit))
            self.assertTrue(all("__autosim_source_id__" not in row for row in observed))
            payload = json.loads(path.read_text())
            self.assertEqual(payload["batches_yielded"], 2)
            self.assertEqual(payload["samples_yielded"], 5)
            self.assertEqual([row["yielded_samples"] for row in payload["parts"]], [3, 2])
            self.assertAlmostEqual(payload["parts"][1]["realized_sampling_mass"], 0.4)

    def test_concat_source_tag_is_opt_in(self):
        class Items(_FakeDataset):
            def __init__(self, episode_lengths):
                super().__init__(episode_lengths)
                features = {
                    key: {"dtype": "video" if "images" in key else "float32"}
                    for key in TRAIN.ACT_REQUIRED_FEATURES
                }
                self.meta = SimpleNamespace(features=features, stats={})
                self.num_frames = len(self)
                self.num_episodes = len(episode_lengths)

            def __getitem__(self, index):
                return {"x": index}

        dataset = TRAIN._CompatibleConcatDataset(
            [Items([2]), Items([1])],
            entries=[{"root": "/a"}, {"root": "/b"}],
            track_sources=True,
        )
        self.assertEqual(dataset[0]["__autosim_source_id__"], 0)
        self.assertEqual(dataset[2]["__autosim_source_id__"], 1)


if __name__ == "__main__":
    unittest.main()
