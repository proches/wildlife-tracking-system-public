"""19/10: looking down next to the water-hole shows the lodge pool and guests' loungers."""

import unittest

import numpy as np

from utils.privacy import in_privacy_zone, path_crosses_privacy, privacy_filter


class PrivacyTests(unittest.TestCase):
    def test_pool_view_is_private_plain_and_water_are_not(self):
        self.assertTrue(in_privacy_zone((0.70, -0.90, 0.03)))      # the pool (measured view)
        self.assertFalse(in_privacy_zone((0.7249, -1.0, 0.2193)))  # water-hole preset
        self.assertTrue(in_privacy_zone((0.6244, -0.93, 0.03)))    # pool side: loungers at the edge
        self.assertFalse(in_privacy_zone((0.4189, -0.80, 0.03)))   # low_5: clean
        self.assertFalse(in_privacy_zone((0.70, -1.0, 0.03)))      # same pan, plain view

    def test_glide_back_from_the_low_row_must_lift_first(self):
        self.assertTrue(path_crosses_privacy(0.6244, 0.7249, -0.93))
        self.assertFalse(path_crosses_privacy(0.6244, 0.7249, -1.0))
        self.assertFalse(path_crosses_privacy(0.4189, 0.3450, -0.80))
        self.assertTrue(path_crosses_privacy(0.4189, 0.7249, -0.80))   # low_5 -> water-hole: lift first

    def test_private_frames_are_blurred_for_stream_and_preview(self):
        rng = np.random.default_rng(0)
        f = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
        out = privacy_filter(f, (0.70, -0.90, 0.03))
        self.assertLess(float(np.abs(np.diff(out[100:200, 100:400].astype(int), axis=1)).mean()), 5.0)
        self.assertIs(privacy_filter(f, (0.70, -1.0, 0.03)), f)

    def test_patrol_lifts_before_crossing(self):
        from tracking.observation_patrol import ObservationPatrol

        class Cam:
            def stop(self):
                pass
        p = ObservationPatrol(Cam(), clock=lambda: 0.0)
        p.presets = {"x": {"token": "1", "pan": 0.7}}
        water = next(i for i, s in enumerate(p.stations) if s["name"] == p.water_name) if p.water_name else None
        if water is None:
            self.skipTest("no water-hole configured")
        p.route = [water]
        p.route_pos = 0
        p._last_pan, p._last_tilt = 0.6244, -0.93
        p._begin_travel()
        self.assertEqual(p.phase, "lift")


if __name__ == "__main__":
    unittest.main()
