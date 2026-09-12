import sqlite3
import json
import tempfile
import unittest
from pathlib import Path

from autosim.experiment_validation.collection_gap import (
    collection_master_seed,
    derived_config,
    find_node,
    reserve_collection_banks,
    validate_collection_seed_trace,
)
from autosim.robosyn_data import evaluation_seed_bank


class CollectionGapTest(unittest.TestCase):
    def source(self, root, task):
        source = Path(__file__).resolve().parents[2] / f"RoboSynChallenge/configs/{task}/action_config.json"
        self.assertTrue(source.is_file())
        return source

    def test_official_graph_is_exact_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("sample_loading", "item_assembly"):
                source, target = self.source(root, task), root / f"{task}.json"
                provenance = derived_config(task, "official_graph", source, target)
                self.assertEqual(json.loads(source.read_text()), json.loads(target.read_text()))
                self.assertIsNone(provenance["mutation"])

    def test_sample_mutation_only_prepends_explicit_pose_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = self.source(root, "sample_loading")
            target = root / "derived.json"
            derived_config("sample_loading", "final_place_raise_001m", source, target)
            before, after = json.loads(source.read_text()), json.loads(target.read_text())
            a = find_node(after, "left_arm_cube_place_qpos")["kwargs"]["affordance_infos"][0]["valid_funcs_name_kwargs_proc"]
            b = find_node(before, "left_arm_cube_place_qpos")["kwargs"]["affordance_infos"][0]["valid_funcs_name_kwargs_proc"]
            self.assertEqual(a[1:], b)
            self.assertEqual(a[0]["pass_processes"][0]["kwargs"], {"offset_value": .01, "direction": "z", "mode": "extrinsic"})
            a.pop(0)
            self.assertEqual(after, before)

    def test_item_mutation_changes_one_declared_scalar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = self.source(root, "item_assembly")
            target = root / "derived.json"
            derived_config("item_assembly", "left_align_offset_minus_020m", source, target)
            before, after = json.loads(source.read_text()), json.loads(target.read_text())
            process = find_node(after, "left_align_qpos")["kwargs"]["affordance_infos"][0]["valid_funcs_name_kwargs_proc"][0]["pass_processes"][0]
            self.assertEqual(process["kwargs"]["offset_value"], -.20)
            process["kwargs"]["offset_value"] = -.24
            self.assertEqual(after, before)

    def test_probe_and_production_seed_streams_are_globally_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = reserve_collection_banks(root)
            self.assertEqual(first, reserve_collection_banks(root))
            for task, variants in first.items():
                streams = []
                for phases in variants.values():
                    for phase in phases.values():
                        streams.append(set(evaluation_seed_bank(
                            phase["master_seed"], phase["reserved_reset_count"])))
                for index, left in enumerate(streams):
                    for right in streams[index + 1:]:
                        self.assertFalse(left & right)
            db = sqlite3.connect(root / "seeds.sqlite")
            self.assertEqual(db.execute("select count(*) from banks").fetchone()[0], 12)
            self.assertEqual(db.execute("select count(*) from seeds").fetchone()[0], 1284)
            db.close()

    def test_collection_trace_must_match_reserved_stream_and_saved_lineage(self):
        master, target, maximum = collection_master_seed("sample_loading", "official_graph", False), 1, 12
        seeds = evaluation_seed_bank(master, maximum + 1)
        collection = {"master_seed": master, "target_successful_episodes": target,
            "collection_mode": "expert", "profile": "full_random", "status": "completed",
            "expert_attempt_count": 2,
            "attempts": [{"seed": seeds[0], "saved": False}, {"seed": seeds[1], "saved": True}],
            "resets": [{"seed": seed} for seed in seeds[:3]],
            "successful_episode_seeds": [seeds[1]], "failed_attempt_seeds": [seeds[0]]}
        result = validate_collection_seed_trace(collection, master_seed=master,
                                                target=target, max_attempts=maximum)
        self.assertTrue(result["ordered_reserved_seed_stream_verified"])
        collection["resets"][1]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "reserved ordered seed stream"):
            validate_collection_seed_trace(collection, master_seed=master,
                                           target=target, max_attempts=maximum)


if __name__ == "__main__":
    unittest.main()
