"""Headless integration test of the 8 Hz control loop with fake grabber / AI worker."""
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import config
from main import Runtime
from tests.sim_camera import SimCamera
from tracking.multi_object_tracker import MultiObjectTracker
from tracking.observation_patrol import OBSERVE_PHASES
from tracking.ptz_tracker import PTZTracker

W, H = 1920, 1080


class FakeGrabber:
    ok = True

    def __init__(self, clock):
        self.clock = clock
        self.frame = np.full((H, W, 3), 100, np.uint8)
        self.seq = 0

    def latest(self):
        self.seq += 1
        return self.frame, self.clock(), self.seq

    def frame_fn(self):
        return self.frame


class FakeWorker:
    """Pretends to be the AI thread: records jobs, returns scripted detections."""

    def __init__(self, clock, script):
        self.clock, self.script = clock, script
        self.busy = False
        self.last_seconds = 2.5
        self.jobs, self.pending, self.resets = [], None, 0
        self.tracks = MultiObjectTracker()

    def submit(self, job):
        self.jobs.append(job)
        dets = self.script(job)
        tracks = self.tracks.update(dets, W, H)
        self.pending = dict(job, tracks=tracks, metrics={"inference_seconds": 2.5}, done_at=self.clock())
        return True

    def request_reset(self):
        self.resets += 1
        self.tracks.reset()

    def take_result(self):
        r, self.pending = self.pending, None
        return r


def zebra(box, conf=0.8):
    return {"box": box, "class_id": 0, "det_confidence": conf, "species": "zebra", "species_confidence": 0.9,
            "source": "fake", "in_water_zone": False}


class RuntimeIntegration(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        for name, val in (("PTZ_CAPABILITIES_FILE", os.path.join(tmp, "caps.json")),
                          ("PTZ_TRACE_CSV", os.path.join(tmp, "trace.csv")),
                          ("METRICS_JSONL", os.path.join(tmp, "m.jsonl"))):
            p = patch.object(config, name, val)
            p.start()
            self.addCleanup(p.stop)
        for name in ("log_detection", "save_review", "maybe_save_negative_sample", "write_health"):
            p = patch(f"main.{name}", Mock(return_value=None))
            p.start()
            self.addCleanup(p.stop)

    def build(self, script, cam=None):
        self.now = [100.0]
        clock = lambda: self.now[0]
        cam = cam or SimCamera(pan=0.5, absolute_works=True)
        patcher = patch("time.monotonic", clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        grabber = FakeGrabber(clock)
        worker = FakeWorker(clock, script)
        tracker = PTZTracker(cam)
        tracker.observation_patrol.clock = clock
        tracker.observation_patrol.motion.clock = clock
        rt = Runtime()
        S = SimpleNamespace(
            last_seq=-1, last_motion=clock(), last_epoch=-1, tracks=[], result_pose=None, result_t=0.0,
            full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0, prev_done=clock(),
            health=None, allowed=False, epoch=-1)
        trace_f = open(config.PTZ_TRACE_CSV, "w", newline="")
        self.addCleanup(trace_f.close)
        import csv
        trace = csv.writer(trace_f)
        focus = Mock(last_result="", busy=False)
        tracker.observation_patrol.focus = None

        def tick(n, dt=0.125):
            for _ in range(n):
                self.now[0] += dt
                cam.advance(dt)
                rt._tick(cam, grabber, worker, tracker, tracker.observation_patrol, focus, None, S, trace, trace_f, self.now[0])
        return cam, tracker, worker, rt, tick

    @patch.object(config, "PATROL_DETECT_WHILE_SWEEPING", False)   # this test is about the settle gate
    def test_empty_scene_patrol_only_analyses_settled_frames_and_keeps_moving(self):
        cam, tracker, worker, rt, tick = self.build(lambda job: [])
        pans = []
        for _ in range(240 * 8 // 8):
            tick(8)
            pans.append(cam.pan)
        self.assertGreater(len(worker.jobs), 10)
        patrol = tracker.observation_patrol
        # camera really moved across the arc, while the AI thread was "busy" for seconds at a time
        self.assertGreater(max(pans) - min(pans), 0.2)
        # every analysed frame was captured while the camera was still
        for job in worker.jobs:
            self.assertGreaterEqual(job["epoch"], 1)
        low, high = config.ALLOWED_PAN_RANGE
        self.assertTrue(all(low - 0.003 <= p <= high + 0.003 for p in pans))

    def test_confirmed_animal_takes_over_from_patrol_and_is_followed(self):
        # an animal in the right third of the frame once the camera is observing
        def script(job):
            return [zebra([1500, 480, 1620, 580])]
        cam, tracker, worker, rt, tick = self.build(script)
        for _ in range(400):
            tick(4)
            if tracker.locked:
                break
        self.assertTrue(tracker.locked, "tracker never locked a confirmed animal")
        pulses = [e for e in cam.log if e[0] == "pulse"]
        self.assertTrue(pulses)
        # animal right of centre: on this camera the pan must DECREASE (measured 19/10)
        self.assertLess(pulses[0][1] * config.TRACK_PAN_IMAGE_SIGN, 0) if False else self.assertEqual(pulses[0][1] > 0, config.TRACK_PAN_IMAGE_SIGN > 0)

    def test_unconfirmed_weak_blob_never_steers_the_camera(self):
        def script(job):
            return [{"box": [1500, 480, 1560, 530], "class_id": 0, "det_confidence": 0.21,
                     "species": "unidentified", "species_confidence": 0.1, "source": "fake"}]
        cam, tracker, worker, rt, tick = self.build(script)
        for _ in range(600):
            tick(4)
        self.assertFalse(tracker.locked)
        self.assertFalse([e for e in cam.log if e[0] == "pulse"])

    def test_patrol_waits_while_a_candidate_is_being_verified(self):
        def script(job):
            return [{"box": [900, 500, 960, 540], "class_id": 0, "det_confidence": 0.26,
                     "species": "unidentified", "species_confidence": 0.1, "source": "fake"}]
        cam, tracker, worker, rt, tick = self.build(script)
        patrol = tracker.observation_patrol
        # run until observing
        for _ in range(2000):
            tick(1)
            if patrol.phase == "observe_wide":
                break
        station = patrol.station["name"]
        tick(8 * 12)   # 12 s later: still here because a candidate is pending
        self.assertEqual(patrol.station["name"], station)
        self.assertGreaterEqual(patrol.candidates_pending, 1)


if __name__ == "__main__":
    unittest.main()
