import unittest

from autosim.experiment_validation.fairness_audit import (
    collection_seed_set,
    command_flag,
    command_value,
    selected_round,
)


class FairnessAuditTest(unittest.TestCase):
    def test_command_value_requires_one_explicit_value(self):
        process = {"command": ["train.py", "--steps", "20000", "--seed", "1000"]}
        self.assertEqual(command_value(process, "--steps"), "20000")
        with self.assertRaises(ValueError):
            command_value(process, "--missing")
        with self.assertRaises(ValueError):
            command_value({"command": ["x", "--steps", "1", "--steps", "2"]}, "--steps")

    def test_command_flag_requires_at_most_one_occurrence(self):
        self.assertTrue(command_flag({"command": ["train.py", "--resume"]}, "--resume"))
        self.assertFalse(command_flag({"command": ["train.py"]}, "--resume"))
        with self.assertRaises(ValueError):
            command_flag({"command": ["train.py", "--resume", "--resume"]}, "--resume")

    def test_collection_seed_set_reports_duplicates(self):
        seeds, unique = collection_seed_set({"successful_episode_seeds": [3, 4, 3]})
        self.assertEqual(seeds, {3, 4})
        self.assertFalse(unique)
        seeds, unique = collection_seed_set({"successful_episode_seeds": [3, 4]})
        self.assertEqual(seeds, {3, 4})
        self.assertTrue(unique)

    def test_selected_round_uses_score_then_shorter_trajectory(self):
        rows = [
            {"index": 0, "auto_selection_score": .6, "random_selection_score": .7,
             "summary": {"average_action_steps": 100}, "random_summary": {"average_action_steps": 200}},
            {"index": 1, "auto_selection_score": .6, "random_selection_score": .6,
             "summary": {"average_action_steps": 90}, "random_summary": {"average_action_steps": 80}},
        ]
        self.assertEqual(selected_round(rows, "auto")["index"], 1)
        self.assertEqual(selected_round(rows, "random")["index"], 0)
        with self.assertRaises(ValueError):
            selected_round(rows, "unknown")


if __name__ == "__main__":
    unittest.main()
