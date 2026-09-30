import tempfile
import unittest
from pathlib import Path

from autosim.experiment_validation.fault_suite import run_case
from autosim.research.common import digest


class FaultSuiteTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = {str(Path(__file__).resolve()): digest(Path(__file__))}

    def tearDown(self):
        self.temp.cleanup()

    def case(self, scenario, variant):
        return run_case(self.root / (scenario + variant), scenario, variant, self.sources)

    def test_post_receipt_recovery_avoids_real_duplicate_work(self):
        fixed = self.case("controller_crash_after_receipt", "fixed_with_retry")
        durable = self.case("controller_crash_after_receipt", "durable_with_retry")
        self.assertTrue(fixed["committed"] and durable["committed"])
        self.assertEqual((fixed["fixture_launches"], durable["fixture_launches"]), (2, 1))
        self.assertEqual((fixed["commit_count"], durable["commit_count"]), (1, 1))
        self.assertGreater(fixed["successful_receipt_work_discarded_seconds"], 0)

    def test_transient_retry_is_independent_of_receipt_reuse(self):
        for variant in ("fixed_with_retry", "durable_with_retry"):
            row = self.case("transient_exit_137", variant)
            self.assertTrue(row["committed"])
            self.assertEqual(row["fixture_launches"], 2)
        row = self.case("transient_exit_137", "durable_no_retry")
        self.assertFalse(row["committed"])
        self.assertEqual(row["fixture_launches"], 1)

    def test_shared_safe_stops_and_zero_score_semantics(self):
        for scenario in ("invalid_artifact", "evaluation_exit_137", "unknown_receipt", "zero_score_fixture"):
            row = self.case(scenario, "durable_with_retry")
            self.assertTrue(row["expected_outcome_reached"])
            self.assertFalse(row["wrong_accept"])
            self.assertEqual(row["evaluation_reexecutions"], 0)


if __name__ == "__main__":
    unittest.main()
