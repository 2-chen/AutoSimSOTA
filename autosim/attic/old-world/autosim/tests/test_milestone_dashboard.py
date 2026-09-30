import unittest

from autosim.experiment_validation.milestone_dashboard import decide


class MilestoneDashboardTest(unittest.TestCase):
    def complete(self):
        return {"integration": {"status": "integration_gate_reached"},
                "baseline": {"status": "baseline_table_complete", "completed": 10},
                "gap": {"status": "collection_gap_handled"},
                "comparison_state": {"status": "three_task_comparison_complete"},
                "comparison": {"study_execution_complete": True,
                    "automatic_decision_beats_random_control_established": True,
                    "final_test_used": False, "sota_established": False},
                "fairness": {"all_three_tasks_equal_budget": True},
                "checkpoints": {"all_selected_checkpoints_passed": True}}

    def test_only_all_eight_real_gates_complete_milestone(self):
        gates, complete = decide(self.complete())
        self.assertTrue(complete)
        self.assertEqual(len(gates), 8)
        for name in self.complete():
            data = self.complete()
            if name == "comparison":
                data[name]["automatic_decision_beats_random_control_established"] = False
            elif name == "baseline":
                data[name]["completed"] = 9
            elif name == "integration":
                data[name]["status"] = "requires_capability_review"
            elif name == "gap":
                data[name]["status"] = "requires_review"
            elif name == "comparison_state":
                data[name]["status"] = "running"
            elif name == "fairness":
                data[name]["all_three_tasks_equal_budget"] = False
            elif name == "checkpoints":
                data[name]["all_selected_checkpoints_passed"] = False
            self.assertFalse(decide(data)[1], name)


if __name__ == "__main__":
    unittest.main()
