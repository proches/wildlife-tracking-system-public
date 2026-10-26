import unittest
from unittest.mock import Mock, patch

import config
from tests.sim_camera import SimCamera
from tracking.ptz_tracker import PTZTracker

W, H = 1920, 1080


def animal(box, tid=1, confirmed=True):
    return {"box": box, "class_id": 0, "det_confidence": 0.8, "species": "zebra", "species_confidence": 0.9,
            "track_id": tid, "track_confirmed": confirmed, "priority_score": 0.7}


class FreshnessTests(unittest.TestCase):
    def setUp(self):
        self.cam = SimCamera(pan=0.5)
        self.tracker = PTZTracker(self.cam)

    def test_stale_ticks_never_re_steer_from_old_boxes(self):
        self.tracker.update([animal([1500, 480, 1620, 580])], W, H, fresh=True, now=10.0)
        pulses = len([e for e in self.cam.log if e[0] == "pulse"])
        self.assertEqual(pulses, 1)
        for i in range(20):
            self.tracker.update([animal([1500, 480, 1620, 580])], W, H, fresh=False, now=10.1 + i * 0.1)
        self.assertEqual(len([e for e in self.cam.log if e[0] == "pulse"]), pulses)
        self.assertTrue(self.tracker.locked)

    def test_lock_is_kept_between_slow_ai_results_then_released_after_timeout(self):
        self.tracker.update([animal([900, 500, 1020, 580])], W, H, fresh=True, now=10.0)
        self.assertTrue(self.tracker.locked)
        self.tracker.update([], W, H, fresh=False, now=10.0 + config.LOST_TARGET_SECONDS - 1)
        self.assertTrue(self.tracker.locked)
        # One slow AI pass (12 s on the site PC) must not drop the animal...
        self.tracker.update([], W, H, fresh=False, now=10.0 + config.LOST_TARGET_SECONDS + 1)
        self.assertTrue(self.tracker.locked)
        # ...only fresh results that no longer contain it do.
        t = 10.0 + config.LOST_TARGET_SECONDS + 2
        for i in range(config.LOST_TARGET_MIN_RESULTS):
            self.tracker.update([], W, H, fresh=True, now=t + i)
        self.tracker.update([], W, H, fresh=False, now=t + config.LOST_TARGET_MIN_RESULTS)
        self.assertFalse(self.tracker.locked)
        # 19/10: first HOLD the view (the animal may be behind a bush), patrol only afterwards
        t_lost = t + config.LOST_TARGET_MIN_RESULTS
        self.assertTrue(self.tracker.holding(t_lost + 1))
        self.tracker.update([], W, H, fresh=False, now=t_lost + config.LOST_HOLD_SECONDS / 2)
        self.assertFalse(self.tracker.searching)
        self.tracker.update([], W, H, fresh=False, now=t_lost + config.LOST_HOLD_SECONDS + 1)
        self.assertTrue(self.tracker.searching)

    def test_lock_released_when_the_ai_stalls(self):
        self.tracker.update([animal([900, 500, 1020, 580])], W, H, fresh=True, now=10.0)
        self.tracker.update([], W, H, fresh=False, now=10.0 + config.LOST_TARGET_MAX_SECONDS + 1)
        self.assertFalse(self.tracker.locked)

    def test_unconfirmed_tracks_are_ignored(self):
        self.tracker.update([animal([1500, 480, 1620, 580], confirmed=False)], W, H, fresh=True, now=5.0)
        self.assertFalse(self.tracker.locked)
        self.assertFalse([e for e in self.cam.log if e[0] == "pulse"])

    def test_patrol_resumes_when_nothing_is_there(self):
        for i in range(30):
            self.tracker.update([], W, H, fresh=False, now=1.0 + i * 0.125)
            self.cam.advance(0.125)
        self.assertTrue(self.tracker.searching)
        # the patrol is under way (commands are paced by the real clock, which a
        # 30-tick unit test barely advances)
        self.assertNotEqual(self.tracker.observation_patrol.phase, "bootstrap")
        self.assertIsNotNone(self.tracker.observation_patrol.motion.target)

    def test_never_leaves_wildlife_arc_even_when_target_is_at_the_edge(self):
        cam = SimCamera(pan=config.ALLOWED_PAN_RANGE[1] - 0.001)
        tracker = PTZTracker(cam)
        for i in range(20):
            tracker.update([animal([1800, 480, 1900, 580])], W, H, fresh=True, now=float(i))
            cam.advance(0.1)
            self.assertLessEqual(cam.pan, config.ALLOWED_PAN_RANGE[1] + 0.003)


class EdgeGuardTests(unittest.TestCase):
    def test_patrol_margin_reaches_water_hole_but_not_beyond_the_arc(self):
        cam = SimCamera()
        high = config.ALLOWED_PAN_RANGE[1]
        water = config.WATER_HOLE_POSITION["pan"]
        pose = (water, -1.0, 0.01)
        # live-tracking margin (0.03) blocks pushing further out here ...
        self.assertEqual(cam.guard_pan_speed(0.1, status=_st(pose)), 0)
        # ... patrol margin lets it settle on the water-hole, but never past the hard limit
        self.assertGreater(cam.guard_pan_speed(0.1, status=_st((high - 0.01, -1, 0.01)), margin=config.PATROL_EDGE_MARGIN), 0)
        self.assertEqual(cam.guard_pan_speed(0.1, status=_st((high - 0.001, -1, 0.01)), margin=config.PATROL_EDGE_MARGIN), 0)
        self.assertEqual(cam.guard_pan_speed(0.1, status=_st((high + 0.05, -1, 0.01)), margin=config.PATROL_EDGE_MARGIN), 0)


def _st(pose):
    from camera.onvif_camera import SimpleNamespace_pose
    return SimpleNamespace_pose(pose)


if __name__ == "__main__":
    unittest.main()


class InconclusiveResultTests(unittest.TestCase):
    """19/10: a walking baboon was dropped by site-model-only results."""

    def test_site_model_only_results_do_not_lose_the_lock(self):
        cam = SimCamera(pan=0.5)
        tracker = PTZTracker(cam)
        tracker.update([animal([900, 500, 1020, 580])], W, H, fresh=True, now=10.0)
        t = 10.0
        for _ in range(3):      # MegaDetector ran on none of these
            t += 4.0
            tracker.update([], W, H, fresh=True, now=t, conclusive=False)
        self.assertTrue(tracker.locked)
        for _ in range(config.LOST_TARGET_MIN_RESULTS + 1):
            t += 4.0
            tracker.update([], W, H, fresh=True, now=t, conclusive=True)
        tracker.update([], W, H, fresh=False, now=t + 1)
        self.assertFalse(tracker.locked)

    def test_endless_inconclusive_results_are_capped(self):
        cam = SimCamera(pan=0.5)
        tracker = PTZTracker(cam)
        tracker.update([animal([900, 500, 1020, 580])], W, H, fresh=True, now=10.0)
        t = 10.0
        for _ in range(20):
            t += 4.0
            tracker.update([], W, H, fresh=True, now=t, conclusive=False)
        self.assertFalse(tracker.locked)

    def test_camera_frames_the_whole_group(self):
        cam = SimCamera(pan=0.5)
        tracker = PTZTracker(cam)
        a = animal([1500, 500, 1560, 560]); a["track_id"] = 1
        b = animal([1700, 520, 1760, 580]); b["track_id"] = 2
        box = tracker._group_box(a, [a, b], W, H)
        self.assertEqual(box, [1500, 500, 1760, 580])


class WeakTargetTests(unittest.TestCase):
    """20/10: a clicked vervet monkey (box 0.21) was confirmed but never locked."""

    def test_operator_target_locks_below_the_normal_score(self):
        tracker = PTZTracker(SimCamera(pan=0.5))
        d = dict(animal([900, 500, 960, 560]), det_confidence=0.21)
        d["operator_target"] = True
        tracker.update([d], W, H, fresh=True, now=10.0)
        self.assertTrue(tracker.locked)

    def test_locked_target_is_kept_when_its_score_drops(self):
        tracker = PTZTracker(SimCamera(pan=0.5))
        tracker.update([dict(animal([900, 500, 960, 560]), det_confidence=0.6)], W, H, fresh=True, now=10.0)
        for i in range(5):
            tracker.update([dict(animal([905, 502, 965, 562]), det_confidence=0.17)], W, H, fresh=True, now=14.0 + 4 * i)
        self.assertTrue(tracker.locked)
