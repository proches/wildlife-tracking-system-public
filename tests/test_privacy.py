"""19/10: looking down next to the water-hole shows the lodge pool and guests' loungers."""

import unittest

import numpy as np

from utils.privacy import in_privacy_zone, min_zoom_for_pan, pan_high_for_zoom, path_crosses_privacy, privacy_filter


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


class ZoomedArcEndTests(unittest.TestCase):
    """27/10 (user: "ndio ongeza ila pima"): measured with the camera at pans 0.735-0.785 - the
    lodge deck starts at pan ~0.79 at the bottom-left of the picture.  Past the old arc end
    (0.735) the camera may turn only while zoomed in so far that it stays out of the picture."""

    def test_wide_views_keep_the_old_arc_end(self):
        for zoom in (None, 0.0, 0.011, 0.025, 0.03):
            self.assertAlmostEqual(pan_high_for_zoom(zoom), 0.735, places=6, msg=zoom)

    def test_zoomed_in_it_may_turn_further_but_never_past_the_tree(self):
        self.assertTrue(0.762 < pan_high_for_zoom(0.10) < 0.770)       # 10.5x
        self.assertTrue(0.766 < pan_high_for_zoom(0.144) < 0.775)      # 14.7x (zebras 26/10)
        self.assertAlmostEqual(pan_high_for_zoom(0.49), 0.775, places=6)
        for zoom in (0.05, 0.1, 0.144, 0.3):
            pan = pan_high_for_zoom(zoom)
            self.assertFalse(in_privacy_zone((pan, -1.0, zoom)), zoom)

    def test_a_view_whose_edge_reaches_the_deck_is_private(self):
        self.assertTrue(in_privacy_zone((0.77, -1.0, 0.03)))           # 3.85x: the deck shows (probe 0.775)
        self.assertTrue(in_privacy_zone((0.785, -1.0, 0.10)))          # the railing post at the edge (probe)
        self.assertFalse(in_privacy_zone((0.765, -1.0, 0.144)))        # probe: tree and lanterns only
        self.assertFalse(in_privacy_zone((0.7249, -1.0, 0.2193)))      # the water-hole preset

    def test_tilting_down_stays_private_in_the_extended_arc(self):
        self.assertTrue(in_privacy_zone((0.76, -0.90, 0.144)))         # looking down toward the pool

    def test_the_widest_zoom_allowed_at_a_pan(self):
        self.assertEqual(min_zoom_for_pan(0.70), 0.0)
        for pan in (0.74, 0.755, 0.77):
            z = min_zoom_for_pan(pan)
            self.assertGreaterEqual(pan_high_for_zoom(z) + 1e-9, pan, pan)
            self.assertFalse(in_privacy_zone((pan, -1.0, z)), pan)


if __name__ == "__main__":
    unittest.main()
