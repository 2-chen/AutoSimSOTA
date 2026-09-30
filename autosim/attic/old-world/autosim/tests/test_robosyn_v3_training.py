import importlib.util
import os
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from autosim.robosyn_v3 import (
    BANKS,
    aggregate_replicates,
    correction_allocation,
    evaluate_internal_frozen_once,
    failure_stage_report,
)
from autosim.robosyn_v3_mixtures import build_mixture_experiments


TRAIN_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "RoboSynChallenge/policy/act/scripts/train.py"
)
if os.environ.get("AUTOSIM_ROBOSYN_REPO"):
    TRAIN_SCRIPT = Path(os.environ["AUTOSIM_ROBOSYN_REPO"]) / "policy/act/scripts/train.py"
if not TRAIN_SCRIPT.is_file():
    raise unittest.SkipTest("External RoboSyn integration: set AUTOSIM_ROBOSYN_REPO")
SPEC = importlib.util.spec_from_file_location("robosyn_act_train_v3", TRAIN_SCRIPT)
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class _Dataset:
    def __init__(self, length=4):
        self.episode_data_index = {"from": [0], "to": [length]}
        self.num_frames = length
        self.num_episodes = 1

    def __len__(self):
        return self.num_frames


class RoboSynV3TrainingTest(unittest.TestCase):
    def test_mixture_experiments_preserve_processing_winner_and_mass_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def dataset(name, episodes):
                path = root / name
                (path / "meta").mkdir(parents=True)
                (path / "meta/info.json").write_text(
                    json.dumps(
                        {
                            "total_episodes": episodes,
                            "total_frames": episodes * 74,
                        }
                    )
                )
                return path

            old_profiles = [
                ("official_full_random", "full_random", 1000),
                ("targeted_training_only", "targeted_clutter", 100),
                ("targeted_training_only", "targeted_appearance", 100),
                ("targeted_training_only", "targeted_recovery", 100),
                ("targeted_training_only", "policy_correction", 100),
                ("targeted_training_only", "composite_hard", 100),
            ]
            old_datasets = []
            for index, (role, profile, episodes) in enumerate(old_profiles):
                old_datasets.append(
                    {
                        "role": role,
                        "profile": profile,
                        "root": str(dataset(f"old-{index}", episodes)),
                    }
                )
            old_mixture = root / "old.json"
            old_mixture.write_text(json.dumps({"datasets": old_datasets}))

            correction_roots = {
                "targeted_camera": dataset("new-camera", 100),
                "targeted_recovery": dataset("new-recovery", 121),
                "targeted_clutter": dataset("new-clutter", 79),
                "composite_contact": dataset("new-contact", 100),
            }
            corrections = root / "corrections.json"
            corrections.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "allocation": {
                            "targeted_camera": 100,
                            "targeted_recovery": 121,
                            "targeted_clutter": 79,
                            "composite_contact": 100,
                        },
                        "shards": {
                            name: {"dataset_root": str(path)}
                            for name, path in correction_roots.items()
                        },
                    }
                )
            )
            selected_mixture = root / "selected.json"
            selected_mixture.write_text(
                json.dumps(
                    {
                        "sampling": {
                            "strategy": "stratified_phase",
                            "phase_bins": [{"name": "contact", "weight": 3.0}],
                        }
                    }
                )
            )
            processing = root / "processing.json"
            processing.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "selected": {"name": "P5_illumination"},
                        "candidates": {
                            "P5_illumination": {
                                "dataset_mixture_manifest": str(selected_mixture),
                                "train_params": {
                                    "action_loss_profile": "valid_mean",
                                    "image_augmentation_profile": "illumination_step_mild",
                                },
                            }
                        },
                    }
                )
            )

            result = build_mixture_experiments(
                old_mixture, corrections, processing, root / "mixtures"
            )
            specification = json.loads(Path(result["specification"]).read_text())
            self.assertEqual(specification["processing_winner"], "P5_illumination")
            self.assertEqual(
                [item["name"] for item in specification["candidates"]],
                [
                    "M0_current",
                    "M1_append",
                    "M2_replace_old_targeted",
                    "M3_failure_balanced",
                ],
            )
            for candidate in specification["candidates"]:
                self.assertEqual(
                    candidate["train_overrides"]["image_augmentation_profile"],
                    "illumination_step_mild",
                )
            for manifest_path in result["manifests"].values():
                manifest = json.loads(Path(manifest_path).read_text())
                self.assertLessEqual(manifest["targeted_episode_fraction"], 0.60)
                self.assertAlmostEqual(
                    sum(manifest["sampling"]["profile_masses"].values()), 1.0
                )
                self.assertEqual(
                    manifest["sampling"]["phase_bins"],
                    [{"name": "contact", "weight": 3.0}],
                )

    def test_valid_mean_does_not_dilute_short_horizon_loss(self):
        actions = torch.zeros((1, 4, 1))
        predictions = torch.ones_like(actions)
        is_pad = torch.tensor([[False, False, True, True]])
        legacy = TRAIN._masked_action_l1_loss(
            actions, predictions, is_pad, "legacy_mask_mean"
        )
        valid = TRAIN._masked_action_l1_loss(
            actions, predictions, is_pad, "valid_mean"
        )
        self.assertAlmostEqual(legacy.item(), 0.5)
        self.assertAlmostEqual(valid.item(), 1.0)

    def test_horizon_floor_retains_late_contact_anchors(self):
        dataset = _Dataset(length=4)
        weights, audit = TRAIN._build_stratified_sample_weights(
            [dataset],
            [{"profile": "full_random", "root": "/fake"}],
            {
                "strategy": "stratified_phase",
                "profile_masses": {"full_random": 1.0},
                "phase_bins": [
                    {"name": "all", "start": 0.0, "end": 1.01, "weight": 1.0}
                ],
                "horizon_weighting": {
                    "mode": "linear_floor",
                    "chunk_size": 4,
                    "floor": 0.25,
                },
            },
        )
        np.testing.assert_allclose(weights.numpy(), [0.4, 0.3, 0.2, 0.1])
        self.assertEqual(audit[0]["horizon_weighting"]["mode"], "linear_floor")

    def test_v3_seed_masters_are_unique_and_retire_seed_zero(self):
        masters = [spec["master_seed"] for spec in BANKS.values()]
        self.assertEqual(len(masters), len(set(masters)))
        self.assertNotIn(0, masters)

    def test_replicate_aggregation_requires_identical_seed_coverage(self):
        common = [
            {"episode_seed": 1, "success": True},
            {"episode_seed": 2, "success": False},
        ]
        result = aggregate_replicates(
            [
                {
                    "episode_count": 2,
                    "success_rate": 0.5,
                    "average_action_steps": 200,
                    "episodes": common,
                },
                {
                    "episode_count": 2,
                    "success_rate": 1.0,
                    "average_action_steps": 150,
                    "episodes": common,
                },
            ]
        )
        self.assertEqual(result["replicate_count"], 2)
        self.assertAlmostEqual(result["mean_success_rate"], 0.75)
        self.assertEqual(result["minimum_success_rate"], 0.5)

    def test_failure_report_and_allocation_are_bounded(self):
        report = failure_stage_report(
            [
                {
                    "episodes": [
                        {
                            "failure_stage": "no_button_contact",
                            "max_button_press_depth_m": 0.0,
                        },
                        {
                            "failure_stage": "contact_insufficient_press",
                            "max_button_press_depth_m": 0.002,
                        },
                    ]
                }
            ]
        )
        self.assertEqual(report["stage_counts"]["no_button_contact"], 1)
        allocation = correction_allocation(
            {
                "appearance": 0.6,
                "camera": 0.4,
                "robot_pose": 0.5,
                "clutter": 0.3,
                "contact": 0.2,
            }
        )
        self.assertEqual(sum(allocation["episode_allocation"].values()), 400)
        self.assertEqual(allocation["episode_allocation"]["composite_contact"], 126)

    def test_internal_frozen_state_refuses_repeat_before_checkpoint_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "repo": str(root / "repo"),
                        "python": str(root / "python"),
                        "baseline_checkpoint": str(root / "baseline"),
                        "output_root": str(root / "output"),
                    }
                )
            )
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            output = root / "frozen"
            output.mkdir()
            (output / "internal_frozen_v3_state.json").write_text(
                json.dumps({"status": "completed"})
            )
            with self.assertRaisesRegex(RuntimeError, "already started or completed"):
                evaluate_internal_frozen_once(
                    config,
                    protocol,
                    root / "candidate",
                    root / "incumbent",
                    output,
                )


if __name__ == "__main__":
    unittest.main()
