import unittest

from autosim.research.failure_slice_analysis import scene_slice


class FailureSliceAnalysisTest(unittest.TestCase):
    def test_frozen_camera_and_clutter_slices_use_realized_state(self):
        row = {
            "cameras": {"cam_high": {
                "intrinsics": [[[631.315186, 0, 320], [0, 606.100952, 240], [0, 0, 1]]],
                "local_pose": [[[1, 0, 0, .257046], [0, 1, 0, .049382],
                                [0, 0, 1, 1.459689], [0, 0, 0, 1]]],
            }},
            "entities": {"button": {"pose": [
                [[1, 0, 0, .5], [0, 1, 0, 0], [0, 0, 1, .83], [0, 0, 0, 1]]]}},
            "active_distractors": [{"pose": [
                [[1, 0, 0, .6], [0, 1, 0, 0], [0, 0, 1, .9], [0, 0, 0, 1]]]}],
        }
        result = scene_slice(row)
        self.assertTrue(result["camera_high"])
        self.assertAlmostEqual(result["camera_intrinsic_severity"], .5)
        self.assertTrue(result["clutter_near"])
        self.assertAlmostEqual(result["clutter_distance"], .1)


if __name__ == "__main__":
    unittest.main()
