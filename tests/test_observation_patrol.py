import os
import tempfile
import unittest
from unittest.mock import patch

import config
from tests.sim_camera import SimCamera
from tracking.observation_patrol import ObservationPatrol, OBSERVE_PHASES


def run_patrol(camera, seconds, ai_every=3.0, dt=0.125, hold_candidates=0):
    """Drive the patrol like main.py does: 8 Hz ticks, a fake AI result every few seconds."""
    now = [0.0]
    patrol = ObservationPatrol(camera, clock=lambda: now[0])
    patrol.candidates_pending = hold_candidates
    visited, observed_zoom_at, last_ai = [], {}, 0.0
    low, high = config.ALLOWED_PAN_RANGE
    max_zoom_seen = 0.0
    while now[0] < seconds:
        now[0] += dt
        camera.advance(dt)
        assert low - 0.003 <= camera.pan <= high + 0.003, f"left the wildlife arc: {camera.pan}"
        patrol.tick(pose=camera.get_pose())
        max_zoom_seen = max(max_zoom_seen, camera.zoom)
        if patrol.detection_allowed and now[0] - last_ai >= ai_every:
            patrol.note_result(patrol.observe_epoch)
            last_ai = now[0]
        if patrol.phase in OBSERVE_PHASES:
            name = patrol.station["name"]
            if not visited or visited[-1] != (name, patrol.phase):
                visited.append((name, patrol.phase))
            if patrol.phase == "observe_zoom":
                observed_zoom_at[name] = camera.zoom
    return patrol, visited, observed_zoom_at, max_zoom_seen


class PatrolWithWorkingAbsoluteMove(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        p = patch.object(config, "PTZ_CAPABILITIES_FILE", os.path.join(self.tmp, "caps.json"))
        p.start()
        self.addCleanup(p.stop)

    def test_visits_every_station_including_water_hole_and_zooms_in(self):
        cam = SimCamera(pan=0.476, absolute_works=True)
        patrol, visited, zooms, max_zoom = run_patrol(cam, 900)
        names = {n for n, _ in visited}
        self.assertEqual(names, {s["name"] for s in patrol.stations})
        water = next(s for s in patrol.stations if s["name"] == "water_hole")
        self.assertAlmostEqual(zooms["water_hole"], water["zoom"], delta=config.PTZ_ZOOM_TOLERANCE * 2)
        self.assertGreater(max_zoom, config.WATER_HOLE_POSITION["zoom"] - 0.01)  # really zoomed in at the water-hole

    def test_zoom_returns_wide_before_travelling(self):
        cam = SimCamera(pan=0.476, absolute_works=True)
        patrol, *_ = run_patrol(cam, 400)
        # whenever the patrol is in 'travel' the lens must be wide
        cam2 = SimCamera(pan=0.5, zoom=0.30, absolute_works=True)
        now = [0.0]
        p = ObservationPatrol(cam2, clock=lambda: now[0])
        saw_travel_zoom = []
        for _ in range(4000):
            now[0] += 0.125
            cam2.advance(0.125)
            p.tick(pose=cam2.get_pose())
            if p.phase == "travel":
                saw_travel_zoom.append(cam2.zoom)
            if p.detection_allowed:
                p.note_result(p.observe_epoch)
        self.assertTrue(saw_travel_zoom)
        self.assertLess(max(saw_travel_zoom), config.ZOOM_SEARCH_POSITION + config.PTZ_ZOOM_TOLERANCE * 2)


class PatrolWhenCameraIgnoresAbsoluteMove(unittest.TestCase):
    """The real-world failure: AbsoluteMove accepted but nothing moves."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.caps = os.path.join(self.tmp, "caps.json")
        p = patch.object(config, "PTZ_CAPABILITIES_FILE", self.caps)
        p.start()
        self.addCleanup(p.stop)

    def test_falls_back_to_continuous_and_still_covers_the_arc(self):
        cam = SimCamera(pan=0.7265, zoom=0.0058, absolute_works=False)   # the stuck pose from the field logs
        patrol, visited, zooms, max_zoom = run_patrol(cam, 1500)
        self.assertIs(patrol.motion.absolute_ok, False)
        names = {n for n, _ in visited}
        self.assertEqual(names, {s["name"] for s in patrol.stations})
        self.assertGreater(max_zoom, config.WATER_HOLE_POSITION["zoom"] - 0.01)
        with open(self.caps) as f:
            self.assertIn('"absolute_move_works": false', f.read())

    def test_pan_actually_sweeps_to_both_ends(self):
        cam = SimCamera(pan=0.7265, zoom=0.0058, absolute_works=False)
        pans = []
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        for _ in range(12000):
            now[0] += 0.125
            cam.advance(0.125)
            patrol.tick(pose=cam.get_pose())
            if patrol.detection_allowed:
                patrol.note_result(patrol.observe_epoch)
            pans.append(cam.pan)
        self.assertLess(min(pans), config.ALLOWED_PAN_RANGE[0] + 0.07)
        self.assertGreater(max(pans), config.ALLOWED_PAN_RANGE[1] - 0.02)


class PatrolRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        p = patch.object(config, "PTZ_CAPABILITIES_FILE", os.path.join(self.tmp, "caps.json"))
        p.start()
        self.addCleanup(p.stop)

    @patch.object(config, "WATER_HOLE_EVERY_N_STATIONS", 2)
    def test_water_hole_revisited_every_n_stations(self):
        cam = SimCamera()
        patrol = ObservationPatrol(cam)
        names = [patrol.stations[i]["name"] for i in patrol.route]
        every = config.WATER_HOLE_EVERY_N_STATIONS
        gaps, since = [], 0
        for n in names * 2:
            if n == "water_hole":
                gaps.append(since)
                since = 0
            else:
                since += 1
        self.assertTrue(gaps)
        self.assertLessEqual(max(gaps[1:]), every + 1)
        self.assertEqual({s["name"] for s in patrol.stations}, set(names))

    def test_tilt_mismatch_does_not_block_arrival(self):
        """Regression: station tilt -0.98 vs camera floor -1.0 used to prevent 'arrived' forever."""
        cam = SimCamera(pan=0.5, tilt=-1.0, absolute_works=True)
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        for _ in range(3000):
            now[0] += 0.125
            cam.advance(0.125)
            patrol.tick(pose=cam.get_pose())
            if patrol.phase in OBSERVE_PHASES:
                break
        self.assertIn(patrol.phase, OBSERVE_PHASES)

    def test_ai_gated_until_camera_is_still_and_settled(self):
        cam = SimCamera(pan=0.7265, zoom=0.30)
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        for _ in range(200):
            now[0] += 0.125
            cam.advance(0.125)
            patrol.tick(pose=cam.get_pose())
            if patrol.moving:
                self.assertFalse(patrol.detection_allowed)

    def test_patrol_waits_for_ai_result_before_leaving_station(self):
        cam = SimCamera(pan=0.5, absolute_works=True)
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        station_at_first_observe = None
        for _ in range(4000):
            now[0] += 0.125
            cam.advance(0.125)
            patrol.tick(pose=cam.get_pose())
            if patrol.phase == "observe_wide":
                station_at_first_observe = patrol.station["name"]
                break
        # no AI result is ever delivered: patrol must stay until the max wait expires
        start = now[0]
        while patrol.phase == "observe_wide" and now[0] - start < config.PATROL_OBSERVE_MAX_WAIT - 1:
            now[0] += 0.125
            patrol.tick(pose=cam.get_pose())
        self.assertEqual(patrol.phase, "observe_wide")
        self.assertEqual(patrol.station["name"], station_at_first_observe)

    def test_candidates_extend_the_observation(self):
        cam = SimCamera(pan=0.5, absolute_works=True)
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        while patrol.phase != "observe_wide":
            now[0] += 0.125
            cam.advance(0.125)
            patrol.tick(pose=cam.get_pose())
        patrol.candidates_pending = 1
        start = now[0]
        while now[0] - start < config.PATROL_OBSERVE_MAX_WAIT - 2:
            now[0] += 0.125
            patrol.note_result(patrol.observe_epoch)
            patrol.tick(pose=cam.get_pose())
        self.assertEqual(patrol.phase, "observe_wide")

    def test_water_hole_outside_limits_rejected(self):
        with patch.object(config, "WATER_HOLE_POSITION", dict(pan=-0.5, tilt=-0.9, zoom=0.1)):
            with self.assertRaises(ValueError):
                ObservationPatrol(SimCamera())

    def test_missing_position_stops_instead_of_moving(self):
        cam = SimCamera()
        cam.get_pose = lambda: None
        cam.get_status = lambda: None
        patrol = ObservationPatrol(cam)
        self.assertFalse(patrol.tick())
        self.assertEqual(cam.log, [])

    def test_far_outside_arc_never_moves(self):
        cam = SimCamera(pan=-0.8)
        patrol = ObservationPatrol(cam)
        self.assertFalse(patrol.tick())
        self.assertEqual(cam.log, [])


if __name__ == "__main__":
    unittest.main()
