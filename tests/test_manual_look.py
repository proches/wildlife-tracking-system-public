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
