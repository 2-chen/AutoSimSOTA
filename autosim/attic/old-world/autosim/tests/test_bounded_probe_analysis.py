import unittest

from autosim.research.bounded_probe_analysis import _features, _wilson


class BoundedProbeAnalysisTest(unittest.TestCase):
    def test_extracts_realized_features_without_using_profile_label(self):
        row = {
            "requested_profile": "misleading_name_is_ignored",
            "cameras": {"cam_high": {
                "intrinsics": [[[600, 0, 320], [0, 590, 240], [0, 0, 1]]],
                "local_pose": [[[1, 0, 0, .1], [0, 1, 0, .2],
                                [0, 0, 1, 1.5], [0, 0, 0, 1]]],
            }},
            "entities": {"button": {"pose": [
                [[1, 0, 0, .7], [0, 1, 0, -.1], [0, 0, 1, .83], [0, 0, 0, 1]]]}},
            "robot_qpos": [[3.0, 4.0]],
            "active_distractors": [{"uid": "a"}, {"uid": "b"}],
        }
        result = _features(row)
        self.assertEqual(result["camera_fx"], 600)
        self.assertEqual(result["camera_z"], 1.5)
        self.assertEqual(result["button_y"], -.1)
        self.assertEqual(result["robot_qpos_l2"], 5.0)
        self.assertEqual(result["active_distractor_count"], 2.0)

    def test_wilson_interval_handles_small_probe(self):
        low, high = _wilson(10, 20)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        self.assertIsNone(_wilson(0, 0))


if __name__ == "__main__":
    unittest.main()
