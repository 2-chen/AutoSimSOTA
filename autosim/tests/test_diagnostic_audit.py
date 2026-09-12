import json
import tempfile
import unittest
from pathlib import Path

from autosim.research.diagnostic_audit import (
    audit_collection_scene_evidence,
    audit_history,
    render_markdown,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class DiagnosticAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, *, with_scene=False, with_scene_sidecar=False,
                with_exposure=False):
        collection = self.root / "click_bell/round_1/auto/collection"
        attempts = [{"seed": 1, "saved": True, "reason": "saved"},
                    {"seed": 2, "saved": False, "reason": "invalid"}]
        if with_scene:
            attempts[0]["scene_parameters"] = {"button_x": 0.1}
            attempts[1]["scene_parameters"] = {"button_x": 0.2}
        write(collection / "collection.json", {
            "task": "click_bell", "profile": "full_random", "status": "completed",
            "attempts": attempts,
        })
        if with_scene_sidecar:
            rows = [
                {"event": "collection_scene_reset", "seed": 1,
                 "entities": {"button": {"pose": [[1.0]]}},
                 "robot_qpos": [0.0],
                 "unavailable_realized_parameters": ["camera_intrinsics"]},
                {"event": "collection_scene_reset", "seed": 2,
                 "entities": {"button": {"pose": [[2.0]]}},
                 "robot_qpos": [0.1],
                 "unavailable_realized_parameters": ["camera_intrinsics"]},
            ]
            (collection / "scene_resets.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n")
        write(collection / "data_audit.json", {
            "passed": True, "errors": [], "dataset": {"total_episodes": 1},
        })
        round_dir = collection.parents[1]
        write(round_dir / "auto_mixture.json", {
            "sampling": "proportional_to_frames", "datasets": [
                {"root": "/official", "profile": "full_random", "source_kind": "official_full_random",
                 "episode_count": 10, "frame_count": 90},
                {"root": "/new", "profile": "camera", "source_kind": "self_collected",
                 "episode_count": 1, "frame_count": 10},
            ],
        })
        write(round_dir / "auto/recipe_10.json", {"training_data_content_ids": {"/new": "id"}})
        if with_exposure:
            write(round_dir / "auto/training_exposure_audit.json", {
                "samples_yielded": 20, "parts": [{"yielded_samples": 18}, {"yielded_samples": 2}],
            })

    def test_missing_scene_and_exposure_remains_unidentifiable(self):
        self.fixture()
        result = audit_history(self.root, ["click_bell"])
        self.assertFalse(result["conclusion"]["historical_natural_bias_proven"])
        self.assertTrue(result["conclusion"]["prospective_instrumentation_required"])
        run = result["runs"][0]
        self.assertAlmostEqual(run["training"]["sources"][1]["expected_sampling_mass"], 0.1)
        self.assertFalse(run["identifiability"]["accepted_source_to_training_exposure"])
        self.assertIn("不可识别", render_markdown(result))

    def test_scene_and_exposure_make_flow_observable_but_not_causal(self):
        self.fixture(with_scene=True, with_exposure=True)
        result = audit_history(self.root, ["click_bell"])
        self.assertTrue(result["conclusion"]["historical_natural_bias_proven"])
        self.assertFalse(result["runs"][0]["identifiability"]["failure_slice_to_learning_effect"])

    def test_seed_joined_scene_sidecar_is_realized_evidence(self):
        self.fixture(with_scene_sidecar=True, with_exposure=True)
        result = audit_history(self.root, ["click_bell"])
        scene = result["runs"][0]["scene_evidence"]
        self.assertEqual(scene["attempts_joined_by_seed"], 2)
        self.assertEqual(scene["attempts_with_measurable_realized_state"], 2)
        self.assertTrue(scene["complete_for_attempt_outcomes"])
        self.assertEqual(scene["unsupported_realized_parameter_fields"],
                         ["camera_intrinsics"])
        self.assertTrue(result["conclusion"]["historical_natural_bias_proven"])

    def test_collection_scene_gate_joins_by_seed_and_ignores_extra_reset(self):
        self.fixture(with_scene_sidecar=True)
        collection = self.root / "click_bell/round_1/auto/collection"
        with (collection / "scene_resets.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "seed": 999, "task": "click_bell",
                "requested_profile": "full_random", "entities": {},
                "robot_qpos": [0.0],
                "privileged_training_diagnostics_only": True,
            }) + "\n")
        # Fill the metadata required by the prospective gate; the history audit
        # intentionally accepts older sidecars without these fields.
        rows = [json.loads(line) for line in
                (collection / "scene_resets.jsonl").read_text().splitlines()]
        for row in rows[:2]:
            row.update(task="click_bell", requested_profile="full_random",
                       privileged_training_diagnostics_only=True)
        (collection / "scene_resets.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n")
        result = audit_collection_scene_evidence(
            collection / "collection.json", collection / "scene_resets.jsonl")
        self.assertTrue(result["passed"])
        self.assertEqual(result["joined_attempt_count"], 2)
        self.assertEqual(result["extra_reset_seeds_not_attempts"], [999])


if __name__ == "__main__":
    unittest.main()
