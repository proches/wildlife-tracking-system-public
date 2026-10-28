"""Click-to-zoom (19/10): centre a clicked animal by measured picture shift, then zoom in."""

import unittest

import numpy as np

import config
from tracking.manual_look import ManualLook

W, H = 1920, 1080


class FakeCam:
    """The view is a window on a big textured canvas; pan + moves the picture +x."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.ox, self.oy = 600.0, 400.0          # window origin on the canvas
        self.vel = (0.0, 0.0)
        self.zoom = 0.03
        self.t = 0.0

    def velocity_move(self, pan, tilt, zoom, pose=None, timeout=None, edge_margin=None):
        self.vel = (pan, tilt)
        return pan, tilt, zoom

    def stop(self):
        self.vel = (0.0, 0.0)

    def move(self, pan, tilt, zoom, prevalidated=False, pulse_seconds=0.4):
        # measured on the real camera: pan + moves the picture right
        self.ox -= pan / 0.2 * 90 * pulse_seconds       # window moves left = picture moves right
        self.oy -= tilt / 0.2 * 50 * pulse_seconds

    def zoom_only(self, z):
        self.zoom = z

    def advance(self, dt):
        self.t += dt
        self.ox -= self.vel[0] / 0.2 * 90 * dt     # 90 px/s at speed 0.2; picture moves opposite to the window
        self.oy -= self.vel[1] / 0.2 * 50 * dt

    def frame(self):
        x, y = int(round(self.ox)), int(round(self.oy))
        return self.canvas[y:y + H, x:x + W].copy()


class ManualLookTests(unittest.TestCase):
    def test_click_is_centred_then_zoomed_and_held(self):
        rng = np.random.default_rng(1)
        small = rng.integers(0, 255, (130, 220, 3), dtype=np.uint8)
        canvas = np.kron(small, np.ones((20, 20, 1), np.uint8))          # blocky texture 4400x2600
        cam = FakeCam(canvas)
        t = [0.0]
        look = ManualLook(cam, (1500, 800), cam.frame(), (0.5, -1.0, 0.03), clock=lambda: t[0])
        target_on_canvas = (cam.ox + 1500, cam.oy + 800)
        phases = []
        for _ in range(int(120 / 0.125)):
            t[0] += 0.125
            cam.advance(0.125)
            look.tick(cam.frame(), t[0] - 0.05, (0.5, -1.0, cam.zoom))
            phases.append(look.phase)
            if look.phase == "hold":
                break
        self.assertEqual(look.phase, "hold")
        cx, cy = target_on_canvas[0] - cam.ox, target_on_canvas[1] - cam.oy
        self.assertLess(abs(cx - W / 2), config.MANUAL_CENTER_TOLERANCE * W + 30)
        self.assertLess(abs(cy - H / 2), config.MANUAL_CENTER_TOLERANCE * W + 30)
        mag = 1 + config.PTZ_ZOOM_MAG_PER_UNIT * cam.zoom
        self.assertAlmostEqual(mag, 3.85 * config.MANUAL_ZOOM_FACTOR, delta=0.3)
        self.assertTrue(look.detection_allowed)

    def test_the_reference_picture_survives_the_measurements(self):
        """26/10: with a window OpenCV 5 multiplies phaseCorrelate's inputs in place, so every
        measurement faded the kept reference (on camera frames: response 0.96 -> 0.59 in 7)."""
        rng = np.random.default_rng(2)
        canvas = np.kron(rng.integers(0, 255, (130, 220, 3), dtype=np.uint8), np.ones((20, 20, 1), np.uint8))
        cam = FakeCam(canvas)
        look = ManualLook(cam, (1500, 800), cam.frame(), (0.5, -1.0, 0.03), clock=lambda: 0.0)
        ref = look.ref.copy()
        cam.ox -= 120
        first = look._shift(cam.frame())
        for _ in range(6):
            look._shift(cam.frame())
        np.testing.assert_array_equal(look.ref, ref)
        self.assertAlmostEqual(look._shift(cam.frame())[2], first[2], places=6)


if __name__ == "__main__":
    unittest.main()


class AutoInvestigateTests(unittest.TestCase):
    """19/10: a baboon in deep shade scored 0.46 'unidentified' and was never looked at closer."""

    def test_half_seen_candidate_on_still_camera_is_zoomed_once(self):
        from types import SimpleNamespace
        from main import Runtime
        cam = FakeCam(np.zeros((1200, 2200, 3), np.uint8))
        tracker = SimpleNamespace(locked=False, manual_active=False)
        patrol = SimpleNamespace(moving=False, motion=SimpleNamespace(target=None))
        rt = Runtime()
        cand = {"box": [1068, 513, 1120, 580], "det_confidence": 0.46, "track_confirmed": False,
                "track_hits": 1, "species": "unidentified"}
        S = SimpleNamespace(tracks=[cand])
        frame = np.zeros((H, W, 3), np.uint8)
        rt._maybe_investigate(cam, tracker, patrol, None, frame, (0.345, -0.80, 0.03), 100.0, S)
        self.assertIsInstance(rt.manual, ManualLook)
        self.assertTrue(tracker.manual_active)
        rt.manual = None
        rt._maybe_investigate(cam, tracker, patrol, None, frame, (0.345, -0.80, 0.03), 200.0, S)
        self.assertIsNone(rt.manual)                     # same spot: not again within 10 min

    @staticmethod
    def _vetoed(box, det):
        """An AI result whose only box SpeciesNet vetoed as "blank" (28/10 trace entries)."""
        return {"frame": np.zeros((H, W, 3), np.uint8), "pose": (0.5721, -1.0, 0.1114),
                "metrics": {"dropped": [{"why": "non_wildlife", "box": box, "det": det, "src": "custom_tile",
                                         "label": "blank", "label_conf": 1.0, "cached": False}]}}

    def test_operator_view_spot_vetoed_twice_gets_one_closer_look(self):
        """28/10 10:09-10:14: elephants behind branches in an operator view, every box "blank"."""
        from types import SimpleNamespace
        from main import Runtime
        from tracking.manual_look import OperatorHold
        cam = FakeCam(np.zeros((1200, 2200, 3), np.uint8))
        tracker = SimpleNamespace(locked=False, manual_active=False)
        patrol = SimpleNamespace(moving=False, motion=SimpleNamespace(target=None))
        rt = Runtime()
        hold = OperatorHold((0.5721, -1.0, 0.1114))
        rt.manual = hold
        S = SimpleNamespace(tracks=[])
        pose = (0.5721, -1.0, 0.1114)
        res = self._vetoed([627, 599, 715, 675], 0.277)
        rt._maybe_investigate(cam, tracker, patrol, None, res["frame"], pose, 100.0, S, res)
        self.assertIs(rt.manual, hold)                   # once is not enough
        res = self._vetoed([200, 200, 260, 250], 0.30)
        rt._maybe_investigate(cam, tracker, patrol, None, res["frame"], pose, 120.0, S, res)
        self.assertIs(rt.manual, hold)                   # another spot is not a repeat
        res = self._vetoed([630, 601, 705, 675], 0.249)  # the same spot 56 s later
        rt._maybe_investigate(cam, tracker, patrol, None, res["frame"], pose, 156.0, S, res)
        self.assertIsInstance(rt.manual, ManualLook)
        self.assertTrue(rt.manual.auto)
        self.assertIs(rt.manual.back_to, hold)
        reset = []
        patrol.reset = lambda: reset.append(1)
        rt._look_done(rt.manual, tracker, patrol)        # nothing confirmed: the person's watch goes on
        self.assertIs(rt.manual, hold)
        self.assertEqual(reset, [])

    def test_vetoed_spots_in_patrol_views_or_at_night_are_not_looked_at(self):
        """65 h of trace: 100 such repeats in patrol views, mostly the same bushes every hour."""
        from types import SimpleNamespace
        from main import Runtime
        from tracking.manual_look import OperatorHold
        cam = FakeCam(np.zeros((1200, 2200, 3), np.uint8))
        tracker = SimpleNamespace(locked=False, manual_active=False)
        patrol = SimpleNamespace(moving=False, motion=SimpleNamespace(target=None))
        pose = (0.5721, -1.0, 0.1114)
        rt, S = Runtime(), SimpleNamespace(tracks=[])
        for t in (100.0, 130.0, 160.0):
            res = self._vetoed([627, 599, 715, 675], 0.4)
            rt._maybe_investigate(cam, tracker, patrol, None, res["frame"], pose, t, S, res)
        self.assertIsNone(rt.manual)
        rt, S = Runtime(), SimpleNamespace(tracks=[])
        rt.manual = hold = OperatorHold(pose)
        for t in (100.0, 130.0, 160.0):
            res = self._vetoed([627, 599, 715, 675], 0.4)
            res["metrics"]["frame_quality"] = {"low_light": True}
            rt._maybe_investigate(cam, tracker, patrol, None, res["frame"], pose, t, S, res)
        self.assertIs(rt.manual, hold)

    def test_a_finished_look_without_an_operator_view_resumes_the_patrol(self):
        from types import SimpleNamespace
        from main import Runtime
        rng = np.random.default_rng(3)
        cam = FakeCam(np.kron(rng.integers(0, 255, (130, 220, 3), dtype=np.uint8), np.ones((20, 20, 1), np.uint8)))
        rt = Runtime()
        rt.manual = ManualLook(cam, (900, 500), cam.frame(), (0.5, -1.0, 0.03))
        reset = []
        tracker = SimpleNamespace(locked=False, manual_active=True)
        rt._look_done(rt.manual, tracker, SimpleNamespace(reset=lambda: reset.append(1)))
        self.assertIsNone(rt.manual)
        self.assertFalse(tracker.manual_active)
        self.assertEqual(reset, [1])

    def test_moving_patrol_or_weak_blob_is_not_investigated(self):
        from types import SimpleNamespace
        from main import Runtime
        cam = FakeCam(np.zeros((1200, 2200, 3), np.uint8))
        rt = Runtime()
        frame = np.zeros((H, W, 3), np.uint8)
        weak = SimpleNamespace(tracks=[{"box": [100, 100, 150, 150], "det_confidence": 0.25,
                                        "track_confirmed": False, "track_hits": 3}])
        tracker = SimpleNamespace(locked=False, manual_active=False)
        rt._maybe_investigate(cam, tracker, SimpleNamespace(moving=False, motion=SimpleNamespace(target=None)),
                              None, frame, (0.5, -1, 0.03), 10.0, weak)
        self.assertIsNone(rt.manual)
        strong = SimpleNamespace(tracks=[{"box": [100, 100, 150, 150], "det_confidence": 0.6,
                                          "track_confirmed": False, "track_hits": 3}])
        rt._maybe_investigate(cam, tracker, SimpleNamespace(moving=True, motion=SimpleNamespace(target=None)),
                              None, frame, (0.5, -1, 0.03), 10.0, strong)
        self.assertIsNone(rt.manual)


class ApproachZoomTests(unittest.TestCase):
    def test_zoom_factor_is_used(self):
        rng = np.random.default_rng(2)
        small = rng.integers(0, 255, (130, 220, 3), dtype=np.uint8)
        cam = FakeCam(np.kron(small, np.ones((20, 20, 1), np.uint8)))
        t = [0.0]
        look = ManualLook(cam, (W / 2, H / 2), cam.frame(), (0.5, -1.0, 0.03), clock=lambda: t[0], zoom_factor=4.0)
        for _ in range(400):
            t[0] += 0.125
            look.tick(cam.frame(), t[0] - 0.05, (0.5, -1.0, cam.zoom))
            if look.phase == "hold":
                break
        self.assertAlmostEqual(1 + config.PTZ_ZOOM_MAG_PER_UNIT * cam.zoom, 3.85 * 4.0, delta=0.4)
