import unittest

from autosim.research.correction_probe_audit import _flat


class CorrectionProbeAuditTest(unittest.TestCase):
    def test_vectorized_reset_state_is_flattened_without_changing_values(self):
        self.assertEqual(_flat([[1, 2, 3]]), [1.0, 2.0, 3.0])
        self.assertEqual(_flat([1, 2, 3]), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
