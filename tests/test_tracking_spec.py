"""The 15 acceptance scenarios of the 20/10/2026 tracking review.

Every test drives the REAL code - PTZTracker, ObservationPatrol, the Runtime
state machine and control tick, DetectionWorker, FrameGrabber and the
calibration loaders - with fakes only for the hardware (SimCamera / WorldCam
with the picture rates measured on the SUNBA) and for the AI.  Nothing here
talks to the camera or the network, and every state file is redirected to a
temporary directory.

 1 valid detection while patrolling        9 loss and reacquire
 2 off-centre correction                  10 persistent loss -> recovery -> patrol
 3 inside the deadband                    11 PTZ command failure / timeout
 4 moving target                          12 detection exception while patrolling
 5 temporary occlusion                    13 patrol command while tracking
 6 weak small target                      14 missing / invalid calibration data
 7 multiple targets                       15 stream disconnect / reconnect
 8 false-positive blip
"""

import json
import logging
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import config
from main import DetectionWorker, FrameGrabber, Runtime
from tracking.manual_look import ManualLook, OperatorHold
from tests.sim_camera import SimCamera
import tests.test_world_scenarios as tw    # module import: its TestCase must not be collected twice
from tests.test_world_scenarios import KX, World, WorldCam, mag
from tracking.multi_object_tracker import MultiObjectTracker
from tracking.ptz_tracker import PTZTracker, plan_pulse

W, H = 1920, 1080


def animal(box, tid=1, conf=0.8, confirmed=True, **extra):
    d = {"box": list(box), "class_id": 0, "det_confidence": conf, "species": "african buffalo",
         "species_confidence": 0.9, "track_id": tid, "track_confirmed": confirmed, "priority_score": conf}
    d.update(extra)
    return d


def pulses(cam):
    return [e for e in cam.log if e[0] == "pulse"]


def patrol_commands(cam):
    """Commands only the patrol sends (absolute moves / continuous sweeps)."""
    return [e for e in cam.log if e[0] in ("abs", "vel")]


class _Isolated(unittest.TestCase):
    """Every state file the code under test may write goes to a temporary directory."""

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.tmp = tmp
        for name, val in (("PTZ_CAPABILITIES_FILE", os.path.join(tmp, "caps.json")),
                          ("PTZ_TRACE_CSV", os.path.join(tmp, "trace.csv")),
                          ("METRICS_JSONL", os.path.join(tmp, "m.jsonl")),
                          ("PIPELINE_TRACE_FILE", os.path.join(tmp, "pipeline_trace.jsonl")),
                          ("STATIC_FP_FILE", os.path.join(tmp, "static.json")),
                          ("REVIEW_DIR", os.path.join(tmp, "review"))):
            p = patch.object(config, name, val)
            p.start()
            self.addCleanup(p.stop)

    def machine(self, cam=None):
        """Tracker + patrol + the Runtime state machine around them."""
        cam = cam or SimCamera(pan=0.5)
        tracker = PTZTracker(cam)
        rt = Runtime()
        S = SimpleNamespace()
        return cam, tracker, tracker.observation_patrol, rt, S

    @staticmethod
    def step(tracker, patrol, rt, S, dets, now, fresh=True):
        """What main._tick does around the tracker: ownership, tracker, ownership."""
        rt._update_state(tracker, patrol, now, S)
        tracker.update(dets, W, H, fresh=fresh, now=now)
        rt._update_state(tracker, patrol, now, S)
        return rt.state


# ============================================================ 1-3: acquisition + centring
class AcquireAndCentreTests(_Isolated):
    def test_01_valid_detection_while_patrolling_locks_and_suspends_the_patrol(self):
        cam, tracker, patrol, rt, S = self.machine()
        for i in range(30):                                     # patrol under way
            self.step(tracker, patrol, rt, S, [], 1.0 + i * 0.125, fresh=False)
            cam.advance(0.125)
        self.assertTrue(tracker.searching)
        self.assertEqual(rt.state, "PATROL")
        # (230 px above centre: 26/10 the vertical deadband became 0.20 of the half height)
        state = self.step(tracker, patrol, rt, S, [animal([1300, 260, 1420, 360])], 6.0)
        self.assertEqual(state, "TRACKING")
        self.assertTrue(tracker.locked)
        self.assertFalse(tracker.searching)
        self.assertEqual(patrol.suspended_by, "TRACKING")
        self.assertEqual(tracker.last_selection, (1, "new target: highest priority confirmed animal"))
        # animal right of centre -> the picture must move left -> pan DEcreases (mirrored EFlip handled
        # in the camera layer; TRACK_PAN_IMAGE_SIGN = -1 measured live 20/10)
        self.assertLess(pulses(cam)[-1][1], 0)
        self.assertGreater(pulses(cam)[-1][2], 0)               # above centre -> tilt up

    def test_02_off_centre_target_is_brought_to_the_middle_without_overshoot(self):
        """Closed loop on the simulated camera (measured speed levels, start-up delays,
        30 % slower at high zoom).  Target starts 540 px right / 150 px above."""
        cam = WorldCam(pan=0.45, tilt=-1.0, zoom=0.2)
        m = mag(0.2)
        world = World(cam, 0.45 - 540.0 / (KX * m), -1.0 + 150.0 / (2600.0 * m), size_1x=10.0)
        tracker = PTZTracker(cam)
        tracker.observation_patrol = None                       # tracker alone
        errs = []
        for k in range(8):                                      # one AI result every 4 s
            pose = cam.get_pose()
            tracker.pose_hint = pose
            x, y, s = world.screen(pose)
            errs.append((x - W / 2, y - H / 2))
            box = [x - s / 2, y - s * 0.4, x + s / 2, y + s * 0.4]
            tracker.update([animal(box)], W, H, fresh=True, now=10.0 + 4 * k)
            cam.advance(0.1)
        dz_x = config.FRAME_CENTER_DEADZONE * W / 2
        dist = [abs(ex) for ex, _ in errs]
        self.assertLess(dist[1], dist[0] * 0.5, "the first correction did not halve the offset")
        self.assertLess(dist[-1], 2 * dz_x, f"not centred: {[round(v) for v in dist]}")
        # no overshoot: never more than a deadband to the OTHER side
        self.assertTrue(all(ex > -dz_x for ex, _ in errs), f"overshoot: {[round(e) for e, _ in errs]}")
        self.assertLess(abs(errs[-1][1]), abs(errs[0][1]), "the vertical offset was not reduced")

    def test_03_inside_the_deadband_nothing_moves(self):
        cam, tracker, patrol, rt, S = self.machine()
        tracker.pose_hint = cam.get_pose()
        off_x, off_y = 30.0, 18.0          # px: well inside the 6 % deadband (57 / 32 px), fixed on purpose
        self.assertEqual(plan_pulse(off_x, off_y, W, H)[:2], (0.0, 0.0))
        half = config.TRACK_TARGET_PX / 2                        # right size: no zoom either
        box = [W / 2 + off_x - half, H / 2 + off_y - half * 0.8, W / 2 + off_x + half, H / 2 + off_y + half * 0.8]
        self.step(tracker, patrol, rt, S, [animal(box)], 5.0)
        self.assertTrue(tracker.locked)
        self.assertEqual(pulses(cam), [], "a centred, right-sized target must not move the camera")
        self.assertFalse(tracker.follow_box(box, W, H, now=6.0), "the visual follow moved inside the deadband")
        self.assertEqual(pulses(cam), [])


# ============================================================ 4-5, 9-10: the integrated loop
class WorldLoopTests(_Isolated):
    """The real control tick (main.Runtime._tick, visual follow included) in the simulated world."""

    def setUp(self):
        super().setUp()
        tw.WorldScenarioTests.setUp(self)

    def run_world(self, **kw):
        return tw.WorldScenarioTests.run_world(self, **kw)

    def test_04_moving_target_is_followed_by_the_visual_tracker_between_ai_results(self):
        sent = []
        orig = PTZTracker._command

        def spy(tr, source, *a, **k):
            sent.append(source)
            return orig(tr, source, *a, **k)
        with patch.object(PTZTracker, "_command", spy):
            log = self.run_world(seconds=260, events=[(150.0, lambda w: setattr(w, "walk", 0.0015))])
        walking = [r for r in log if r["t"] > 160]
        in_frame = sum(1 for r in walking if 0 <= r["x"] < W and 0 <= r["y"] < H) / len(walking)
        self.assertGreater(in_frame, 0.85, "lost the walking animal")
        self.assertIn("flow", sent, "the frame-to-frame tracker never steered the camera")

    def test_05_temporary_occlusion_keeps_the_lock_and_the_view(self):
        """What the viewer sees: the camera stays on the spot and follows the animal again
        when it steps out.  (The lock itself may lapse into the RECOVERY hold: this sim's
        3-s AI gives the 3 "missed" results in 9 s and its smooth 25x texture has no
        features for the frame-to-frame tracker - 21/10, after the 42-min-lock fix.)"""
        def hide(w):
            w.present = False

        def show(w):
            w.present = True
        log = self.run_world(seconds=230, events=[(170.0, hide), (179.0, show)])
        first_lock = next(r["t"] for r in log if r["locked"])
        self.assertLess(first_lock, 160)
        around = [r for r in log if 170.0 <= r["t"] <= 200.0]
        self.assertFalse(any(r["searching"] for r in around), "the patrol took the camera during the occlusion")
        self.assertTrue(all(0 <= r["x"] < W for r in around), "the camera left the animal's spot")
        back = next((r["t"] for r in log if r["t"] >= 179.0 and r["locked"]), None)
        self.assertIsNotNone(back, "the animal stepped out again and was not followed")
        self.assertLess(back - 179.0, 10.0, "too slow to follow the animal again")

    def test_09_lost_target_is_reacquired_during_the_recovery_hold(self):
        def leave(w):
            w.present = False

        def back(w):
            w.present = True
        log = self.run_world(seconds=300, events=[(160.0, leave), (215.0, back)])
        gone = [r for r in log if 160.0 <= r["t"] < 215.0]
        t_unlock = next((r["t"] for r in gone if not r["locked"] and not r["manual"]), None)
        self.assertIsNotNone(t_unlock, "the lock was never released while the animal was away")
        self.assertFalse(any(r["searching"] for r in gone), "patrol resumed during the recovery hold")
        relock = next((r["t"] for r in log if r["t"] > 215.0 and r["locked"]), None)
        self.assertIsNotNone(relock, "the animal came back and was not reacquired")
        self.assertLess(relock - 215.0, 30.0)

    def test_10_persistent_loss_goes_recovery_then_patrol(self):
        states = []
        orig = Runtime._update_state

        def spy(rt, *a, **k):
            orig(rt, *a, **k)
            if not states or states[-1] != rt.state:
                states.append(rt.state)
        with patch.object(Runtime, "_update_state", spy):
            log = self.run_world(seconds=330, events=[(180.0, lambda w: setattr(w, "present", False))])
        # the recorded order after the animal left: ... TRACKING -> TARGET_LOST -> RECOVERY -> PATROL
        i = max(k for k, s in enumerate(states) if s == "TRACKING")
        tail = states[i:]
        self.assertEqual(tail[:2], ["TRACKING", "TARGET_LOST"], states)
        self.assertIn("RECOVERY", tail, states)
        self.assertIn(tail[-1], ("PATROL", "CANDIDATE"), states)
        self.assertLess(tail.index("RECOVERY"), len(tail) - 1)
        gone = [r for r in log if r["t"] > 180.0]
        t_unlock = next(r["t"] for r in gone if not r["locked"] and not r["manual"])
        hold = [r for r in gone if t_unlock < r["t"] < t_unlock + config.LOST_HOLD_SECONDS - 2]
        self.assertFalse(any(r["searching"] for r in hold), "the patrol resumed before the hold expired")
        self.assertTrue(any(r["searching"] for r in gone if r["t"] > t_unlock + config.LOST_HOLD_SECONDS + 3))


# ============================================================ 5-10 at tracker level (fast, exact)
class TrackerDecisionTests(_Isolated):
    def lock(self, now=5.0, box=(900, 480, 1020, 580)):
        cam, tracker, patrol, rt, S = self.machine()
        tracker.pose_hint = cam.get_pose()
        self.step(tracker, patrol, rt, S, [animal(box)], now)
        self.assertEqual(rt.state, "TRACKING")
        return cam, tracker, patrol, rt, S

    def test_05_occlusion_shorter_than_the_loss_rule_keeps_the_same_target(self):
        cam, tracker, patrol, rt, S = self.lock()
        n_patrol = len(patrol_commands(cam))
        for k in range(config.LOST_TARGET_MIN_RESULTS - 1):            # hidden for 2 AI results
            state = self.step(tracker, patrol, rt, S, [], 9.0 + 4 * k)
            self.assertEqual(state, "TARGET_LOST")
            self.assertTrue(tracker.locked)
        state = self.step(tracker, patrol, rt, S, [animal([930, 480, 1050, 580])], 17.5)
        self.assertEqual(state, "TRACKING")
        self.assertEqual(tracker.locked_track_id, 1)
        self.assertEqual(tracker.missed_frames, 0)
        self.assertEqual(len(patrol_commands(cam)), n_patrol, "the patrol moved during the occlusion")

    def test_06_weak_small_target_is_verified_not_ignored_and_not_locked_on_one_blip(self):
        cam, tracker, patrol, rt, S = self.machine()
        weak = animal([1400, 300, 1430, 324], conf=0.30, confirmed=False, track_hits=1)
        patrol.candidates_pending, patrol.candidate_best = 1, 0.30    # what main sets from the result
        state = self.step(tracker, patrol, rt, S, [weak], 5.0)
        self.assertFalse(tracker.locked, "one weak unconfirmed box must not take the camera")
        self.assertEqual(state, "CANDIDATE")
        self.assertIsNone(patrol.suspended_by, "the patrol (verify pause) owns the camera while it checks")
        self.assertIsNone(tracker.last_ptz, "the tracker steered toward an unconfirmed box")
        # confirmed later by the multi-frame rule: locked even though the box is small and weak
        state = self.step(tracker, patrol, rt, S, [dict(weak, track_confirmed=True)], 9.0)
        self.assertEqual(state, "TRACKING")
        # ...and kept while its score drops below the lock threshold (20/10: a monkey at 0.21 was lost)
        state = self.step(tracker, patrol, rt, S, [dict(weak, det_confidence=0.18, priority_score=0.18)], 13.0)
        self.assertEqual(state, "TRACKING")

    def test_06b_weak_candidate_on_a_still_camera_is_zoomed_in_on(self):
        """main._maybe_investigate: a candidate seen twice >= AUTO_INVESTIGATE_MIN_SCORE -> ACQUIRING."""
        cam, tracker, patrol, rt, S = self.machine(SimCamera(pan=0.5, zoom=0.03))
        frame = np.full((H, W, 3), 90, np.uint8)
        S.tracks = [animal([1400, 300, 1440, 330], conf=config.AUTO_INVESTIGATE_MIN_SCORE + 0.02,
                           confirmed=False, track_hits=2)]
        with patch.object(type(patrol), "moving", new=property(lambda p: False)), \
                patch("main.save_review", Mock()):
            rt._maybe_investigate(cam, tracker, patrol, None, frame, cam.get_pose(), 100.0, S)
        self.assertIsNotNone(rt.manual, "a repeated weak candidate was not looked at")
        self.assertTrue(rt.manual.auto)
        rt._update_state(tracker, patrol, 100.0, S)
        self.assertEqual(rt.state, "ACQUIRING")
        self.assertEqual(patrol.suspended_by, "ACQUIRING")

    def test_07_multiple_targets_group_is_framed_and_the_lock_does_not_jump(self):
        cam, tracker, patrol, rt, S = self.lock(box=(800, 480, 900, 560))
        herd = [animal([800, 480, 900, 560], tid=1, conf=0.6),
                animal([1050, 470, 1150, 560], tid=2, conf=0.9)]       # a stronger second animal
        self.step(tracker, patrol, rt, S, herd, 9.0)
        self.assertEqual(tracker.locked_track_id, 1, "the lock jumped to the stronger animal")
        self.assertEqual(tracker.group_count, 2, "the second animal was not framed with the first")
        # the followed animal disappears behind a bush while the other stays: switch, no patrol
        self.step(tracker, patrol, rt, S, [animal([1050, 470, 1150, 560], tid=2, conf=0.9)], 13.0)
        self.assertEqual(tracker.locked_track_id, 2)
        self.assertEqual(rt.state, "TRACKING")

    def test_08_false_positive_blip_never_takes_the_camera(self):
        """Real MultiObjectTracker: a box seen once (bush glint) is never confirmed."""
        cam, tracker, patrol, rt, S = self.machine()
        mot = MultiObjectTracker()
        blip = {"box": [1500, 200, 1540, 230], "class_id": 0, "det_confidence": 0.45, "species": "unidentified",
                "species_confidence": 0.2, "species_trusted": True, "source": "megadetector",
                "site_model_active": True, "in_water_zone": False}
        seq = [[blip], [], [], []]
        for k, dets in enumerate(seq):
            tracks = mot.update(dets, W, H, pose=cam.get_pose())
            self.step(tracker, patrol, rt, S, tracks, 5.0 + 4 * k)
            self.assertFalse(tracker.locked)
            self.assertNotEqual(rt.state, "TRACKING")
        self.assertIsNone(tracker.last_ptz, "the tracker steered toward a one-off box")

    def test_09_reacquire_during_recovery_without_patrol(self):
        cam, tracker, patrol, rt, S = self.lock()
        t = 5.0
        for k in range(config.LOST_TARGET_MIN_RESULTS):
            t = 9.0 + 4.0 * k
            self.step(tracker, patrol, rt, S, [], t)
        self.step(tracker, patrol, rt, S, [], t + config.LOST_TARGET_SECONDS, fresh=False)
        self.assertEqual(rt.state, "RECOVERY")
        self.assertEqual(patrol.suspended_by, "RECOVERY")
        n_patrol = len(patrol_commands(cam))
        state = self.step(tracker, patrol, rt, S, [animal([1000, 480, 1120, 580], tid=7)],
                          t + config.LOST_TARGET_SECONDS + 20.0)
        self.assertEqual(state, "TRACKING")
        self.assertEqual(tracker.locked_track_id, 7)
        self.assertFalse(tracker.searching)
        self.assertEqual(len(patrol_commands(cam)), n_patrol)

    def test_09b_the_ai_keeps_looking_during_the_recovery_hold(self):
        """20/10 field bug (ptz_trace.csv: ai_busy=0 for the whole 60-s hold, 4 of 4 holds):
        the parked patrol ("bootstrap") counted as camera motion and closed the AI gate."""
        import csv
        cam, tracker, patrol, rt, S0 = self.machine()
        patrol.reset()                                   # what a lock taken from the patrol does
        self.step(tracker, patrol, rt, S0, [animal([900, 480, 1020, 580])], 5.0)
        for k in range(config.LOST_TARGET_MIN_RESULTS):
            self.step(tracker, patrol, rt, S0, [], 9.0 + 4.0 * k)
        self.assertEqual(rt.state, "RECOVERY")
        self.assertTrue(patrol.moving, "precondition: the parked patrol reports 'moving'")
        worker = Mock(busy=False, take_result=Mock(return_value=None), last_seconds=0.0, last_stages=None)
        worker.submit.return_value = True
        S = SimpleNamespace(last_seq=-1, last_motion=0.0, last_epoch=-1, tracks=[], result_pose=None, result_t=0.0,
                            full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0, prev_done=0.0,
                            health=None, allowed=False, epoch=-1, soft_since=0.0)
        trace_f = open(os.path.join(self.tmp, "t.csv"), "w", newline="")
        self.addCleanup(trace_f.close)
        frame = np.full((H, W, 3), 80, np.uint8)
        t_hold = 21.0 + config.LOST_TARGET_SECONDS
        with patch("main.write_health", Mock()), patch.object(config, "OPERATOR_DETECT", False), \
                patch.object(config, "MOTION_CUE_ENABLED", False):
            for k in range(40):                          # 5 s of control ticks inside the hold
                t0 = t_hold + k * 0.125
                grabber = SimpleNamespace(latest=lambda t0=t0, k=k: (frame, t0 - 0.05, 100 + k), ok=True)
                rt._tick(cam, grabber, worker, tracker, patrol, None, None, S, csv.writer(trace_f), trace_f, t0)
                self.assertEqual(rt.state, "RECOVERY")
        self.assertGreaterEqual(worker.submit.call_count, 10, "the AI was starved during the recovery hold")
        self.assertEqual(patrol_commands(cam), [], "the camera moved during the recovery hold")

    def test_10_states_and_reasons_are_logged_in_order(self):
        cam, tracker, patrol, rt, S = self.lock()
        with self.assertLogs("wildlife_tracker.main", level="INFO") as logs:
            t = 9.0
            for k in range(config.LOST_TARGET_MIN_RESULTS):
                t = 9.0 + 4.0 * k
                self.step(tracker, patrol, rt, S, [], t)
            t_lost = t + config.LOST_TARGET_SECONDS
            self.step(tracker, patrol, rt, S, [], t_lost, fresh=False)
            self.step(tracker, patrol, rt, S, [], t_lost + config.LOST_HOLD_SECONDS / 2, fresh=False)
            self.assertEqual(rt.state, "RECOVERY")
            self.step(tracker, patrol, rt, S, [], t_lost + config.LOST_HOLD_SECONDS + 1, fresh=False)
        lines = [r.getMessage() for r in logs.records if r.getMessage().startswith("STATE")]
        self.assertEqual([ln.split(":")[0] for ln in lines],
                         ["STATE TRACKING -> TARGET_LOST", "STATE TARGET_LOST -> RECOVERY", "STATE RECOVERY -> PATROL"])
        self.assertIn("recovery hold expired", lines[-1])
        self.assertTrue(tracker.searching, "the patrol did not resume in the tick the hold expired")
        self.assertIsNone(patrol.suspended_by)


# ============================================================ 11-15: faults
class FaultTests(_Isolated):
    def test_11_ptz_command_timeout_is_logged_stopped_and_survived(self):
        cam = SimCamera(pan=0.5)
        tracker = PTZTracker(cam)
        tracker.pose_hint = cam.get_pose()
        calls = {"n": 0}
        real_move, real_stop = cam.move, cam.stop
        stops = []

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("ONVIF ContinuousMove timed out")
            return real_move(*a, **k)

        def stop():
            stops.append(1)
            real_stop()
        cam.move, cam.stop = flaky, stop
        trace = Mock()
        tracker.trace = trace
        with self.assertLogs("wildlife_tracker.tracking", level="WARNING") as logs:
            tracker.update([animal([1500, 480, 1620, 580])], W, H, fresh=True, now=10.0)
        self.assertTrue(any("Tracking PTZ command failed" in r.getMessage() for r in logs.records))
        self.assertTrue(stops, "the camera was not stopped after the failed command")
        self.assertTrue(tracker.locked, "a camera timeout must not drop the target")
        self.assertFalse(tracker.last_ptz["ok"])
        self.assertIn("TimeoutError", tracker.last_ptz["error"])
        self.assertTrue(trace.log.call_args.kwargs.get("force"), "a failed command must always be traced")
        tracker.update([animal([1450, 480, 1570, 580])], W, H, fresh=True, now=14.0)
        self.assertTrue(tracker.last_ptz["ok"], "the next command did not go through")

    def test_11b_control_loop_survives_a_failing_tick(self):
        rt = Runtime()
        cam = SimCamera(pan=0.5)
        cam.stop = Mock()
        n = {"ticks": 0}

        def tick(*a, **k):
            n["ticks"] += 1
            if n["ticks"] <= 2:
                raise TimeoutError("camera did not answer")
            rt.stop.set()
        with patch.object(rt, "_tick", tick), patch("main.write_health", Mock(return_value=None)), \
                patch.object(config, "OPERATOR_DETECT", False), \
                self.assertLogs("wildlife_tracker.main", level="ERROR") as logs:
            tracker = PTZTracker(cam)
            rt.control_loop(cam, Mock(), Mock(), tracker, None, None)
        self.assertEqual(n["ticks"], 3, "the loop stopped after an exception")
        self.assertEqual(sum("Control tick failed" in r.getMessage() for r in logs.records), 2)
        self.assertGreaterEqual(cam.stop.call_count, 2, "the camera was not stopped after a failed tick")

    def test_12_detection_exception_is_logged_and_the_worker_keeps_going(self):
        pipeline = Mock()
        pipeline.process_frame.side_effect = [RuntimeError("CUDA/ONNX blew up"), []]
        pipeline.last_metrics = {"inference_seconds": 0.1}
        stop = threading.Event()
        worker = DetectionWorker(pipeline, stop)
        job = dict(frame=np.zeros((H, W, 3), np.uint8), pose=(0.5, -1.0, 0.03), epoch=0, scan_mode=True,
                   roi=None, deep=True, seq=1, captured=0.0)
        with self.assertLogs("wildlife_tracker.main", level="ERROR") as logs:
            worker.start()
            self.addCleanup(stop.set)
            worker.submit(dict(job))
            deadline = time.monotonic() + 5
            while pipeline.process_frame.call_count < 1 or worker.busy:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
        self.assertTrue(any("Detection job failed" in r.getMessage() for r in logs.records))
        self.assertIsNone(worker.take_result(), "a failed job produced a result")
        self.assertTrue(worker.submit(dict(job, seq=2)))
        deadline = time.monotonic() + 5
        res = None
        while res is None and time.monotonic() < deadline:
            res = worker.take_result()
            time.sleep(0.01)
        self.assertIsNotNone(res, "the worker died after the exception")
        self.assertEqual(res["seq"], 2)

    def test_12b_patrol_continues_while_the_ai_fails(self):
        cam, tracker, patrol, rt, S = self.machine()
        for i in range(40):                       # no AI result ever arrives
            self.step(tracker, patrol, rt, S, [], 1.0 + i * 0.125, fresh=False)
            cam.advance(0.125)
        self.assertTrue(tracker.searching)
        self.assertEqual(rt.state, "PATROL")
        self.assertIsNotNone(patrol.motion.target, "the patrol stopped without AI results")

    def test_13_patrol_cannot_move_the_camera_while_tracking(self):
        cam, tracker, patrol, rt, S = self.machine()
        for i in range(30):
            self.step(tracker, patrol, rt, S, [], 1.0 + i * 0.125, fresh=False)
            cam.advance(0.125)
        self.assertIsNotNone(patrol.motion.target)       # a patrol move is in flight
        cam.stop = Mock(wraps=cam.stop)
        self.step(tracker, patrol, rt, S, [animal([1300, 400, 1420, 500])], 6.0)
        self.assertTrue(cam.stop.called, "the patrol's move was not stopped at the lock")
        before = list(cam.log)
        with self.assertLogs("wildlife_tracker.tracking", level="INFO") as logs:
            for i in range(10):                          # whatever calls it, the patrol stays quiet
                self.assertFalse(patrol.tick(pose=cam.get_pose()))
        self.assertEqual(patrol_commands(SimpleNamespace(log=cam.log[len(before):])), [])
        self.assertEqual(sum("Patrol suspended" in r.getMessage() for r in logs.records), 1)

    def test_14_missing_or_corrupt_calibration_files_are_reported_and_safe(self):
        from tracking.motion import MotionController
        from tracking.observation_patrol import ObservationPatrol
        corrupt = os.path.join(self.tmp, "corrupt.json")
        with open(corrupt, "w") as f:
            f.write("{ not json")
        # presets: missing -> quietly none; corrupt -> warned, none
        with patch.object(config, "PTZ_PRESETS_FILE", os.path.join(self.tmp, "missing.json")):
            self.assertEqual(ObservationPatrol._load_presets(), {})
        with patch.object(config, "PTZ_PRESETS_FILE", corrupt), \
                self.assertLogs("wildlife_tracker.tracking", level="WARNING"):
            self.assertEqual(ObservationPatrol._load_presets(), {})
        # capabilities: corrupt -> warned; AbsoluteMove NOT trusted (re-tested), never "verified"
        with patch.object(config, "PTZ_CAPABILITIES_FILE", corrupt), \
                self.assertLogs("wildlife_tracker.motion", level="WARNING"):
            mc = MotionController(SimCamera(pan=0.5))
        self.assertIsNone(mc.absolute_ok)
        self.assertFalse(mc._selftest_verified)
        # pan sign: the camera's live EFlip setting wins over a corrupt / contradicting file
        from camera.onvif_camera import OnvifPTZCamera
        with patch.object(config, "PTZ_CAPABILITIES_FILE", corrupt), patch.object(config, "PTZ_PAN_COMMAND_SIGN", "auto"),                 self.assertLogs("wildlife_tracker.camera", level="WARNING"):
            self.assertEqual(OnvifPTZCamera._resolve_pan_sign({"EFlip": "ON"}), -1)
        wrong = os.path.join(self.tmp, "wrong.json")
        with open(wrong, "w") as f:
            json.dump({"pan_command_sign": 1}, f)
        with patch.object(config, "PTZ_CAPABILITIES_FILE", wrong), patch.object(config, "PTZ_PAN_COMMAND_SIGN", "auto"), \
                self.assertLogs("wildlife_tracker.camera", level="ERROR"):
            self.assertEqual(OnvifPTZCamera._resolve_pan_sign({"EFlip": "ON"}), -1)
        # and the patrol still runs without presets, inside the wildlife arc
        with patch.object(config, "PTZ_PRESETS_FILE", os.path.join(self.tmp, "missing.json")):
            cam, tracker, patrol, rt, S = self.machine()
            for i in range(60):
                self.step(tracker, patrol, rt, S, [], 1.0 + i * 0.125, fresh=False)
                cam.advance(0.125)
        self.assertTrue(tracker.searching)
        low, high = config.ALLOWED_PAN_RANGE
        self.assertTrue(low - 0.003 <= cam.pan <= high + 0.003)

    def test_15_stream_disconnect_reconnects_and_the_loop_waits_for_new_frames(self):
        frames = {"n": 0}
        opened = []

        class FakeCap:
            def __init__(self, uri):
                opened.append(uri)
                self.dead = len(opened) == 1          # the first connection dies after 3 frames

            def set(self, *a):
                return True

            def isOpened(self):
                return True

            def read(self):
                frames["n"] += 1
                if self.dead and frames["n"] > 3:
                    return False, None
                return True, np.zeros((4, 4, 3), np.uint8)

            def release(self):
                pass
        metrics = Mock()
        stop = threading.Event()
        with patch("main.cv2.VideoCapture", FakeCap), patch.object(config, "MAX_CONSECUTIVE_DROPPED_FRAMES", 3), \
                patch.object(config, "STREAM_RECONNECT_DELAY_SECONDS", 0.05), \
                self.assertLogs("wildlife_tracker.main", level="WARNING") as logs:
            g = FrameGrabber("rtsp://camera/stream", stop, metrics)
            g.start()
            deadline = time.monotonic() + 5
            while (len(opened) < 2 or g.latest()[2] < 6) and time.monotonic() < deadline:
                time.sleep(0.01)
            stop.set()
            g.join(2)
        self.assertGreaterEqual(len(opened), 2, "never reconnected")
        self.assertTrue(metrics.note_reconnect.called)
        self.assertTrue(any("reconnecting" in r.getMessage() for r in logs.records))
        self.assertGreaterEqual(g.latest()[2], 6, "frames did not resume after the reconnect")
        # the control tick on a frozen stream (same seq) submits nothing twice, and on no frame at all nothing
        cam, tracker, patrol, rt, S0 = self.machine()
        worker = Mock(busy=False, take_result=Mock(return_value=None), last_seconds=0.0, last_stages=None)
        worker.submit.return_value = True
        fresh_t = time.monotonic() + 1e6          # newer than any gate time
        frozen = np.full((H, W, 3), 80, np.uint8)
        S = SimpleNamespace(last_seq=-1, last_motion=0.0, last_epoch=-1, tracks=[], result_pose=None, result_t=0.0,
                            full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0, prev_done=0.0,
                            health=None, allowed=False, epoch=-1, soft_since=0.0)
        trace_f = open(os.path.join(self.tmp, "t.csv"), "w", newline="")
        self.addCleanup(trace_f.close)
        import csv
        with patch("main.write_health", Mock()), patch.object(config, "OPERATOR_DETECT", False), \
                patch.object(config, "MOTION_CUE_ENABLED", False), patch.object(config, "DETECT_ONLY_WHEN_STILL", False):
            for k in range(5):
                grabber = SimpleNamespace(latest=lambda: (frozen, fresh_t, 1), ok=False)
                rt._tick(cam, grabber, worker, tracker, patrol, None, None, S, csv.writer(trace_f), trace_f, 10.0 + k)
            self.assertEqual(worker.submit.call_count, 1, "the frozen frame must be analysed once, not again")
            grabber = SimpleNamespace(latest=lambda: (None, 0.0, 0), ok=False)
            rt._tick(cam, grabber, worker, tracker, patrol, None, None, S, csv.writer(trace_f), trace_f, 20.0)
        self.assertEqual(worker.submit.call_count, 1, "a missing frame was submitted")


# ============================================================ field regressions (real logs)
class ScriptedAI:
    """AI stand-in for the real control tick: each job's result arrives ``secs`` later,
    ``script(n)`` gives the tracks of the n-th result."""

    def __init__(self, clock, script, secs=4.0):
        self.clock, self.script, self.secs = clock, script, secs
        self.job, self.n = None, 0
        self.last_seconds, self.last_stages = secs, None

    @property
    def busy(self):
        return self.job is not None

    def submit(self, job):
        if self.job is not None:
            return False
        self.job, self.due = job, self.clock() + self.secs
        return True

    def request_reset(self):
        pass

    def take_result(self):
        if self.job is None or self.clock() < self.due:
            return None
        job, self.job = self.job, None
        tracks = self.script(self.n)
        self.n += 1
        return dict(job, tracks=tracks, metrics={"conclusive": True, "inference_seconds": self.secs},
                    done_at=self.clock())


class StuckCam(SimCamera):
    """Pulses reach the camera but the picture does not move (21/10: tilt at its limit, the
    reading stays at the -1.0 clamp)."""

    def move(self, pan_speed=0.0, tilt_speed=0.0, zoom_speed=0.0, prevalidated=False,
             travel_mode=False, pulse_seconds=None):
        self.log.append(("pulse", pan_speed, tilt_speed, zoom_speed, pulse_seconds))


class FieldRegressionTests(_Isolated):
    def run_tick_loop(self, frame, script, seconds, cam):
        """The real Runtime._tick at 8 Hz on a fixed picture with a scripted AI."""
        import csv
        now = [1000.0]
        clock = lambda: now[0]
        p = patch("time.monotonic", clock)
        p.start()
        self.addCleanup(p.stop)
        tracker = PTZTracker(cam)
        patrol = tracker.observation_patrol
        patrol.clock = patrol.motion.clock = clock
        patrol.focus = None
        rt = Runtime()
        ai = ScriptedAI(clock, script)
        S = SimpleNamespace(last_seq=-1, last_motion=clock(), last_epoch=-1, tracks=[], result_pose=None,
                            result_t=0.0, full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0,
                            prev_done=clock(), health=None, allowed=False, epoch=-1, soft_since=0.0)
        seq = [0]

        def latest():
            seq[0] += 1
            return frame, now[0] - 0.05, seq[0]
        grabber = SimpleNamespace(latest=latest, ok=True)
        trace_f = open(os.path.join(self.tmp, "t.csv"), "w", newline="")
        self.addCleanup(trace_f.close)
        sent, states = [], []
        orig = PTZTracker._command

        def spy(tr, source, pan, tilt, *a, **k):
            sent.append((now[0], source, pan, tilt))
            return orig(tr, source, pan, tilt, *a, **k)
        with patch.object(PTZTracker, "_command", spy), patch("main.write_health", Mock()), \
                patch("main.log_detection", Mock()), patch("main.save_review", Mock()), \
                patch("main.maybe_save_negative_sample", Mock()), patch.object(config, "OPERATOR_DETECT", False), \
                patch.object(config, "MOTION_CUE_ENABLED", False), patch.object(config, "DETECT_ONLY_WHEN_STILL", False):
            for _ in range(int(seconds / 0.125)):
                now[0] += 0.125
                rt._tick(cam, grabber, ai, tracker, patrol, None, None, S, csv.writer(trace_f), trace_f, now[0])
                if not states or states[-1][1] != rt.state:
                    states.append((now[0], rt.state))
        return sent, states

    def test_16_a_vanished_target_is_not_kept_alive_by_the_visual_tracker(self):
        """21/10 06:59-07:41 (pipeline_trace.jsonl): a buffalo locked at 15x vanished, all 221 AI
        results were empty, yet the lock lasted 42 min and the visual tracker sent 2223 tilt pulses
        into a picture that never moved.  Every empty result re-anchored the flow on the stale AI box
        and refreshed its keepalive."""
        import cv2
        rng = np.random.default_rng(3)
        tex = cv2.resize(rng.integers(30, 220, (270, 480), dtype=np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(cv2.GaussianBlur(tex, (0, 0), 2), cv2.COLOR_GRAY2BGR)   # static: flow stays "ok"
        buffalo = animal([633, 417, 776, 508], tid=2, conf=0.72)
        sent, states = self.run_tick_loop(frame, lambda n: [dict(buffalo)] if n == 0 else [], 200.0,
                                          StuckCam(pan=0.5452, tilt=-1.0, zoom=0.1517))
        names = [s for _, s in states]
        t_lock = next(t for t, s in states if s == "TRACKING")
        self.assertIn("RECOVERY", names, f"the vanished target was never released: {states}")
        t_rec = next(t for t, s in states if s == "RECOVERY")
        self.assertLess(t_rec - t_lock, 60.0, f"released only after {t_rec - t_lock:.0f} s")
        self.assertEqual(names[-1], "PATROL", f"the patrol did not resume: {states}")
        late = [x for x in sent if x[1] == "flow" and x[0] > t_lock + config.FLOW_KEEPALIVE_SECONDS + 1.0]
        self.assertEqual(late, [], "the visual tracker kept steering although the AI no longer saw the animal")


# Hand-checked 21/10 (logs/review candidate frames of the 35 sweep stops for an AI box,
# 19/10 16:20 - 21/10 09:38): (label, score, night/IR, an animal one could follow)
SWEEP_STOPS_21_10 = [
    ("unidentified", 0.44, False, False), ("unidentified", 0.38, False, False), ("unidentified", 0.41, False, False),
    ("unidentified", 0.39, False, False), ("unidentified", 0.43, False, False), ("unidentified", 0.47, False, False),
    ("bovidae family", 0.36, False, False), ("bird", 0.58, True, False), ("unidentified", 0.42, True, False),
    ("bird", 0.47, True, False), ("unidentified", 0.54, True, False), ("unidentified", 0.62, True, False),
    ("unidentified", 0.83, True, False), ("unidentified", 0.46, True, False), ("unidentified", 0.78, True, False),
    ("unidentified", 0.46, True, False), ("unidentified", 0.36, True, False), ("cervidae family", 0.59, False, True),
    ("unidentified", 0.49, False, True), ("bird", 0.75, False, False), ("unidentified", 0.43, False, False),
    ("african buffalo", 0.43, False, True), ("african buffalo", 0.46, False, True), ("unidentified", 0.35, False, False),
    ("unidentified", 0.43, False, False), ("bird", 0.26, False, False), ("bird", 0.63, False, False),
    ("african buffalo", 0.53, False, True), ("african buffalo", 0.29, False, True), ("unidentified", 0.35, False, False),
    ("unidentified", 0.33, False, False), ("unidentified", 0.32, False, False), ("unidentified", 0.36, True, False),
    ("african buffalo", 0.52, False, True), ("helmeted guineafowl", 0.46, False, False),
]


class TravelDetectionTests(_Isolated):
    """Detection while the patrol sweeps (21/10: 204 sweep stops, 7 animals)."""

    def test_18_only_named_daylight_boxes_stop_the_sweep__measured(self):
        from main import worth_a_closer_look
        picked = [real for sp, sc, night, real in SWEEP_STOPS_21_10
                  if worth_a_closer_look({"species": sp, "det_confidence": sc, "track_confirmed": False}, night)]
        real_total = sum(real for *_, real in SWEEP_STOPS_21_10)
        precision, recall = sum(picked) / len(picked), sum(picked) / real_total
        # before: every one of the 35 stopped the sweep (7 real = 20 %), plus 169 stops for movement (0 real)
        self.assertGreaterEqual(precision, 0.70, f"{len(picked)} stops, {sum(picked)} real")
        self.assertGreaterEqual(recall, 0.80, f"{sum(picked)} of {real_total} followable animals kept")
        self.assertLessEqual(len(picked), 10)

    def _moving_blob_cue(self, phase):
        """A small dark animal crossing a textured view: does main._watch_motion report it?"""
        import cv2
        cam, tracker, patrol, rt, S = self.machine(SimCamera(pan=0.2818, tilt=-0.80, zoom=0.03))
        patrol.phase = phase
        rng = np.random.default_rng(5)
        base = cv2.cvtColor(cv2.GaussianBlur(cv2.resize(rng.integers(90, 160, (270, 480), dtype=np.uint8), (W, H)),
                                             (0, 0), 2), cv2.COLOR_GRAY2BGR)
        for k in range(12):                                   # 0.33 s apart, 30 px per step
            f = base.copy()
            x = 600 + 30 * k
            f[500:536, x:x + 36] = 20
            t = 100.0 + 0.34 * k
            rt._watch_motion(tracker, patrol, f, t, cam.get_pose(), t, S)
        return patrol.motion_cue

    def test_18b_generic_boxes_and_movement_do_not_stop_the_sweep(self):
        self.assertIsNone(self._moving_blob_cue("travel"), "movement stopped the sweep: 169 times, 0 animals (21/10)")
        self.assertIsNotNone(self._moving_blob_cue("observe_wide"), "an animal crossing a STILL view went unnoticed")
        cam, tracker, patrol, rt, S = self.machine()
        patrol.phase, patrol.since = "travel", patrol.clock() - 10.0
        patrol.candidates_pending, patrol.candidate_best = 3, 0.47    # acacias called "unidentified"
        patrol.travel_candidate_best = 0.0
        self.assertFalse(patrol._should_verify(0.48, patrol.clock()))
        patrol.travel_candidate_best = 0.46                          # an "african buffalo"
        self.assertTrue(patrol._should_verify(0.48, patrol.clock()))

    def _sweep_machine(self):
        cam, tracker, patrol, rt, S = self.machine(SimCamera(pan=0.4893, tilt=-1.0, zoom=0.03))
        patrol.phase = "travel"                                      # the sweep is moving
        self.assertTrue(patrol.moving)
        cam.stop = Mock(wraps=cam.stop)
        analysed = np.full((H, W, 3), 50, np.uint8)                  # what the AI looked at...
        current = np.full((H, W, 3), 200, np.uint8)                  # ...and the view ~5 s later
        res = {"frame": analysed, "pose": (0.4893, -1.0, 0.03), "metrics": {"frame_quality": {"low_light": False}}}
        return cam, tracker, patrol, rt, S, current, res

    def test_18c_a_named_far_animal_is_zoomed_in_on_from_the_moving_sweep(self):
        cam, tracker, patrol, rt, S, current, res = self._sweep_machine()
        S.tracks = [animal([1000, 500, 1018, 514], tid=9, conf=0.43, confirmed=False, track_hits=1)]   # 18 px buffalo
        rt._maybe_investigate(cam, tracker, patrol, None, current, (0.4860, -1.0, 0.03), 100.0, S, res)
        self.assertIsInstance(rt.manual, ManualLook, "the named far animal was passed")
        self.assertTrue(rt.manual.auto)
        self.assertEqual(rt.manual.zoom_factor, config.APPROACH_MAX_FACTOR)     # 18 px -> as close as allowed
        self.assertLess(float(rt.manual.ref.mean()), 100, "centring must start from the ANALYSED frame")
        self.assertTrue(cam.stop.called, "the sweep was not stopped")
        rt._update_state(tracker, patrol, 100.0, S)
        self.assertEqual(rt.state, "ACQUIRING")

    def test_18d_a_generic_box_on_the_moving_sweep_is_not_zoomed_in_on(self):
        cam, tracker, patrol, rt, S, current, res = self._sweep_machine()
        S.tracks = [animal([1000, 500, 1040, 540], tid=9, conf=0.47, confirmed=False, track_hits=2,
                           species="unidentified")]                    # an acacia
        rt._maybe_investigate(cam, tracker, patrol, None, current, (0.4860, -1.0, 0.03), 100.0, S, res)
        self.assertIsNone(rt.manual)
        res["metrics"]["frame_quality"]["low_light"] = True            # night: not even a named one
        S.tracks = [animal([1000, 500, 1040, 540], tid=9, conf=0.6, confirmed=False, track_hits=2)]
        rt._maybe_investigate(cam, tracker, patrol, None, current, (0.4860, -1.0, 0.03), 100.0, S, res)
        self.assertIsNone(rt.manual)

    def test_18e_a_result_from_the_way_there_is_centred_where_the_animal_was_seen(self):
        """23/10 09:28: a bird seen at pan 0.305 while travelling to scan_7; the result came
        13 s later, the camera standing at pan 0.282.  The look took the LIVE picture as the
        reference for the box, centred 460 px beside the bird and zoomed it out of view."""
        cam, tracker, patrol, rt, S = self.machine(SimCamera(pan=0.2818, tilt=-1.0, zoom=0.03))
        patrol.phase = "observe_wide"                                # arrived: the camera is still
        self.assertFalse(patrol.moving)
        rng = np.random.default_rng(3)
        scene = cv2.GaussianBlur(rng.integers(0, 255, (H, W + 600, 3), dtype=np.uint8), (0, 0), 3)
        analysed = scene[:, 500:500 + W].copy()                      # taken on the way, pan 0.305
        live = scene[:, 40:40 + W].copy()                            # now: the picture moved 460 px
        res = {"frame": analysed, "pose": (0.3048, -1.0, 0.03), "metrics": {"frame_quality": {"low_light": False}}}
        S.tracks = [animal([855, 747, 992, 840], tid=3, conf=0.579, confirmed=False, track_hits=1,
                           species="unidentified")]
        rt._maybe_investigate(cam, tracker, patrol, None, live, cam.get_pose(), 100.0, S, res)
        self.assertIsInstance(rt.manual, ManualLook)
        sx, sy, resp = rt.manual._shift(live)
        x, y = rt.manual.point[0] + sx, rt.manual.point[1] + sy
        self.assertGreater(resp, 0.05)
        self.assertAlmostEqual(x, 924 + 460, delta=15, msg="the look aims beside the bird")
        self.assertAlmostEqual(y, 793, delta=15)

    def test_18f_a_movement_box_is_measured_on_the_live_picture(self):
        cam, tracker, patrol, rt, S = self.machine(SimCamera(pan=0.2818, tilt=-0.80, zoom=0.03))
        patrol.phase = "observe_wide"
        analysed = np.full((H, W, 3), 50, np.uint8)
        live = np.full((H, W, 3), 200, np.uint8)
        res = {"frame": analysed, "pose": (0.3048, -1.0, 0.03), "metrics": {"frame_quality": {"low_light": False}}}
        S.tracks = []
        S.still_motion = (100.0, (1700.0, 900.0, 40.0, 30.0))            # moving now, on the live view
        with patch.object(config, "MOTION_CUE_INVESTIGATE", True), patch("main.save_review"):
            rt._maybe_investigate(cam, tracker, patrol, None, live, cam.get_pose(), 100.5, S, res)
        self.assertIsInstance(rt.manual, ManualLook)
        self.assertGreater(float(rt.manual.ref.mean()), 100, "a live-picture box was put on the old frame")


class MotionLookTests(_Isolated):
    def test_19_movement_on_a_still_view_sends_the_next_ai_look_to_that_spot(self):
        """21/10 10:2x: small monkeys ran across the still low-row view and were missed.
        MegaDetector found 3 of 7 such primates on a native-resolution crop, 0 of 7 on the frame."""
        import csv
        cam, tracker, patrol, rt, S0 = self.machine(SimCamera(pan=0.2818, tilt=-0.80, zoom=0.03))
        patrol.phase = "observe_wide"                      # the camera stands at a station
        worker = Mock(busy=False, take_result=Mock(return_value=None), last_seconds=0.0, last_stages=None)
        worker.submit.return_value = True
        S = SimpleNamespace(last_seq=-1, last_motion=0.0, last_epoch=-1, tracks=[], result_pose=None, result_t=0.0,
                            full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0, prev_done=0.0,
                            health=None, allowed=False, epoch=-1, soft_since=0.0)
        S.still_cue = (100.0, (1700.0, 900.0, 40.0, 30.0), cam.get_pose())   # something moving near the corner
        trace_f = open(os.path.join(self.tmp, "t.csv"), "w", newline="")
        self.addCleanup(trace_f.close)
        frame = np.full((H, W, 3), 80, np.uint8)
        with patch("main.write_health", Mock()), patch.object(config, "OPERATOR_DETECT", False):
            for k, t0 in enumerate((100.5, 101.0)):
                grabber = SimpleNamespace(latest=lambda k=k: (frame, 1e9, k + 1), ok=True)
                rt._tick(cam, grabber, worker, tracker, patrol, None, None, S, csv.writer(trace_f), trace_f, t0)
                if k == 0:
                    roi = worker.submit.call_args[0][0]["roi"]
        x1, y1, x2, y2 = roi
        self.assertEqual((x2 - x1, y2 - y1), (640, 640))
        self.assertTrue(x1 <= 1700 <= x2 and y1 <= 900 <= y2, roi)
        self.assertTrue(0 <= x1 and 0 <= y1 and x2 <= W and y2 <= H, roi)
        self.assertIsNone(worker.submit.call_args[0][0]["roi"], "the following look must cover the whole view again")
        # the camera has turned since the movement: that spot is somewhere else now - no crop
        S.still_cue, S.motion_roi_sent = (110.0, (1700.0, 900.0, 40.0, 30.0), (0.30, -0.80, 0.03)), -1e9
        with patch("main.write_health", Mock()), patch.object(config, "OPERATOR_DETECT", False):
            grabber = SimpleNamespace(latest=lambda: (frame, 1e9, 9), ok=True)
            rt._tick(cam, grabber, worker, tracker, patrol, None, None, S, csv.writer(trace_f), trace_f, 112.0)
        self.assertIsNone(worker.submit.call_args[0][0]["roi"], "a crop was sent for a view the camera has left")


class GroupFramingTests(_Isolated):
    """User 21/10 13:46: "a group is detected - zoom in so ALL of them are seen well, both in the
    centre, follow them, do not leave them".  Measured: 5 animals in 22 % x 13 % of the picture."""

    def tracker_at(self, zoom=0.139):
        cam = SimCamera(pan=0.69, tilt=-1.0, zoom=zoom)
        tracker = PTZTracker(cam)
        tracker.observation_patrol = None
        tracker.pose_hint = cam.get_pose()
        return cam, tracker

    def herd(self, xs, y=540, w=80, h=60, first_tid=1):
        return [animal([x - w / 2, y - h / 2, x + w / 2, y + h / 2], tid=first_tid + i, conf=0.8)
                for i, x in enumerate(xs)]

    def zooms(self, tracker):
        return tracker.last_ptz["cmd"][2] if tracker.last_ptz else 0.0

    def test_20_a_small_group_in_the_centre_is_zoomed_in_on(self):
        cam, tracker = self.tracker_at()
        tracker.update(self.herd([780, 900, 1020, 1140]), W, H, fresh=True, now=10.0,
                       capture_pose=cam.get_pose(), capture_time=10.0)
        self.assertEqual(tracker.group_count, 4)
        self.assertGreater(self.zooms(tracker), 0, "a group in 22 % of the picture must be zoomed in on")

    def test_20b_a_group_that_would_no_longer_fit_is_zoomed_out(self):
        cam, tracker = self.tracker_at(zoom=0.25)
        # the locked animal in the middle, members 700 px to both sides (within the group radius)
        tracker.update(self.herd([960, 260, 1660], w=150, h=110), W, H, fresh=True, now=10.0,
                       capture_pose=cam.get_pose(), capture_time=10.0)
        self.assertEqual(tracker.group_count, 3)
        self.assertLess(self.zooms(tracker), 0, "the group spans 81 % of the picture: zoom out so all stay in")

    def test_20c_a_resting_group_whose_member_count_flickers_is_not_zoomed_out(self):
        cam, tracker = self.tracker_at()
        pattern = [[900, 1020], [780, 900, 1020, 1140, 1320], [900, 1020, 1140], [900, 1020], [780, 900, 1020, 1140]]
        zooms = []
        for k, xs in enumerate(pattern):              # the AI sees 2-5 of the same resting animals
            herd = self.herd(xs, first_tid=1 if xs[0] == 900 else 0)
            herd = [dict(d, track_id=int(x)) for d, x in zip(herd, xs)]   # stable ids per animal
            for d in herd:
                if d["track_id"] == 900:
                    d["track_id"] = 1                 # the locked one
            tracker.pose_hint = cam.get_pose()
            tracker.update(herd, W, H, fresh=True, now=10.0 + 3.0 * k, capture_pose=cam.get_pose(),
                           capture_time=10.0 + 3.0 * k)
            zooms.append(self.zooms(tracker))
        self.assertFalse(any(z < 0 for z in zooms), f"zoomed out of a resting group: {zooms}")

    def test_20d_between_ai_results_the_camera_centres_the_group_not_one_member(self):
        cam, tracker = self.tracker_at()
        herd = self.herd([960, 1110, 1260])                   # locked animal at the centre, 2 to its right
        tracker.update(herd, W, H, fresh=True, now=10.0, capture_pose=cam.get_pose(), capture_time=10.0)
        self.assertGreater(tracker.group_offset[0], 100)
        tracker.camera.log.clear()
        sent = tracker.follow_box([920, 510, 1000, 570], W, H, now=12.0)   # the flow sees the locked one centred
        self.assertTrue(sent, "the camera stayed on one member")
        self.assertLess(tracker.last_ptz["cmd"][0], 0, "it must turn toward the group (picture moves left)")

    def test_20f_tracking_zoom_never_runs_past_the_sharp_limit(self):
        """21/10 13:52: one continuous zoom pulse coasted 26x -> 42x; no rule came back."""
        cam, tracker = self.tracker_at(zoom=0.26)
        tracker.update([animal([910, 500, 1010, 580])], W, H, fresh=True, now=10.0,     # 100 px: zoom in
                       capture_pose=cam.get_pose(), capture_time=10.0)
        steps = [e for e in cam.log if e[0] == "zoomto"]
        self.assertTrue(steps, "no zoom-in step for a small centred animal")
        self.assertLessEqual(steps[-1][1], config.TRACK_MAX_ZOOM + 1e-6, "zoomed past the sharp limit")
        self.assertFalse([e for e in cam.log if e[0] == "pulse" and e[3] != 0], "a coasting zoom pulse was used")
        cam2, tracker2 = self.tracker_at(zoom=0.436)                                       # already at 42x
        tracker2.update([animal([900, 480, 1060, 600])], W, H, fresh=True, now=10.0,
                        capture_pose=cam2.get_pose(), capture_time=10.0)
        back = [e for e in cam2.log if e[0] == "zoomto"]
        self.assertTrue(back and abs(back[-1][1] - config.TRACK_MAX_ZOOM) < 1e-3, f"not brought back: {back}")

    def test_20e_a_far_group_is_approached_with_all_of_them_in_the_centre(self):
        from main import Runtime
        cam = SimCamera(pan=0.69, tilt=-1.0, zoom=0.03)
        tracker = PTZTracker(cam)
        rt = Runtime()
        tracker.pose_hint = cam.get_pose()
        herd = self.herd([700, 1000, 1300], w=40, h=30)       # three 40-px animals spread over 640 px
        tracker.update(herd, W, H, fresh=True, now=10.0, capture_pose=cam.get_pose(), capture_time=10.0)
        self.assertEqual(tracker.group_count, 3)
        g = tracker.group_box
        gw = g[2] - g[0]
        factor = max(1.0, min(config.APPROACH_MAX_FACTOR, config.TRACK_GROUP_FILL_TARGET * W / gw))
        self.assertLess(factor, config.APPROACH_MAX_FACTOR, "test premise: the group limits the zoom")
        # main._tick's approach step, reached through a real tick with a fresh result
        import csv
        frame = np.full((H, W, 3), 90, np.uint8)
        res = dict(frame=frame, pose=cam.get_pose(), epoch=-1, seq=1, captured=10.0, done_at=10.5,
                   tracks=herd, metrics={"conclusive": True, "inference_seconds": 4.0})
        worker = Mock(busy=True, take_result=Mock(return_value=res), last_seconds=4.0, last_stages=None)
        S = SimpleNamespace(last_seq=-1, last_motion=0.0, last_epoch=-1, tracks=[], result_pose=None, result_t=0.0,
                            full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0, prev_done=0.0,
                            health=None, allowed=False, epoch=-1, soft_since=0.0)
        trace_f = open(os.path.join(self.tmp, "t.csv"), "w", newline="")
        self.addCleanup(trace_f.close)
        tracker2 = PTZTracker(cam)
        with patch("main.write_health", Mock()), patch("main.log_detection", Mock()), patch("main.save_review", Mock()), \
                patch("main.maybe_save_negative_sample", Mock()), patch.object(config, "OPERATOR_DETECT", False):
            grabber = SimpleNamespace(latest=lambda: (frame, 11.0, 2), ok=True)
            rt._tick(cam, grabber, worker, tracker2, tracker2.observation_patrol, None, None, S,
                     csv.writer(trace_f), trace_f, 11.0)
        self.assertIsInstance(rt.manual, ManualLook, "the far group was not approached")
        self.assertAlmostEqual(rt.manual.zoom_factor, factor, places=2)
        self.assertAlmostEqual(rt.manual.point[0], (g[0] + g[2]) / 2, delta=1.0)   # the GROUP's centre


class Field22OctTests(_Isolated):
    """22/10 17:38-17:50 (pipeline_trace.jsonl): grazing waterbuck at the right end of the low row."""

    def spy_commands(self):
        sent = []
        orig = PTZTracker._command

        def spy(tr, source, pan, tilt, *a, **k):
            sent.append((source, pan, tilt))
            return orig(tr, source, pan, tilt, *a, **k)
        p = patch.object(PTZTracker, "_command", spy)
        p.start()
        self.addCleanup(p.stop)
        return sent

    def tracker_at(self, pan, tilt=-0.8, zoom=0.0421):
        cam = SimCamera(pan=pan, tilt=tilt, zoom=zoom)
        tracker = PTZTracker(cam)
        tracker.observation_patrol = None
        tracker.pose_hint = cam.get_pose()
        return cam, tracker

    def fresh(self, tracker, cam, dets, t):
        tracker.pose_hint = cam.get_pose()
        tracker.update(dets, W, H, fresh=True, now=t, capture_pose=cam.get_pose(), capture_time=t)

    def test_21_a_lock_hop_to_another_group_member_does_not_send_the_camera_running(self):
        """17:49:43: the lock hopped 856 px to another waterbuck; the hop was read as running and
        feedforward pulses took the camera from pan 0.27 to 0.53 with no animal in view."""
        cam, tracker = self.tracker_at(0.32)
        sent = self.spy_commands()
        # as in the trace: 223 locked, a second waterbuck (224) grazing beside it ...
        self.fresh(tracker, cam, [animal([1395, 356, 1483, 487], tid=223), animal([678, 327, 778, 469], tid=224, conf=0.6)], 10.0)
        self.fresh(tracker, cam, [animal([1400, 355, 1488, 489], tid=223), animal([715, 369, 881, 491], tid=224, conf=0.6)], 22.0)
        self.fresh(tracker, cam, [animal([1399, 351, 1494, 489], tid=223)], 28.0)
        # ... then only a box near where 224 was, under a new tracker id: the lock hops to it
        self.fresh(tracker, cam, [animal([525, 417, 642, 509], tid=225)], 43.0)
        self.assertEqual(tracker.locked_track_id, 225)
        sent.clear()
        for k in range(1, 16):                        # 15 s without an AI result
            tracker.update([], W, H, fresh=False, now=37.0 + k)
        self.assertEqual([s for s in sent if s[0] == "feedforward"], [], "the camera ran after a lock hop")

    def test_21b_an_animal_at_the_end_of_the_arc_is_watched_not_left(self):
        """17:45-17:48: 'Target reached wildlife pan boundary; releasing lock and scanning inward' x4."""
        low = config.ALLOWED_PAN_RANGE[0]
        cam, tracker = self.tracker_at(low + 0.01)
        for k in range(5):                            # the waterbuck stays beyond the right end
            self.fresh(tracker, cam, [animal([1490, 420, 1734, 614], tid=220)], 10.0 + 8 * k)
            self.assertTrue(tracker.locked, f"released at result {k + 1}")
        self.assertFalse(tracker.searching, "the patrol took the camera away from the animal")
        self.assertGreaterEqual(cam.pan, low - 0.003, "left the wildlife arc")

    def test_21c_feedforward_pulses_use_the_slow_level(self):
        cam, tracker = self.tracker_at(0.35, zoom=0.03)
        sent = self.spy_commands()
        for k, x in enumerate((900, 960, 1020)):        # the same animal walking, seen 3 times
            self.fresh(tracker, cam, [animal([x - 50, 500, x + 50, 580], tid=5)], 10.0 + 4 * k)
            cam.pan = 0.35                                 # (the sim camera stays put)
        tracker._world_vel = -0.004                       # a steady walk, as measured from those sightings
        sent.clear()
        for k in range(1, 4):
            tracker.update([], W, H, fresh=False, now=18.0 + k)
        ff = [s for s in sent if s[0] == "feedforward"]
        self.assertTrue(ff, "no feedforward for a steadily walking animal")
        self.assertTrue(all(abs(s[1]) <= config.TRACK_PAN_SLOW for s in ff), f"fast-level feedforward: {ff}")

    def test_21d_low_stations_wait_for_two_ai_looks(self):
        real = os.path.join(os.path.dirname(__file__), "..", "ptz_presets.json")
        copy = os.path.join(self.tmp, "presets.json")
        with open(real, encoding="utf-8") as f, open(copy, "w", encoding="utf-8") as g:
            g.write(f.read())
        with patch.object(config, "PTZ_PRESETS_FILE", copy):
            tracker = PTZTracker(SimCamera(pan=0.2818, tilt=-0.8, zoom=0.03))
        low = [s for s in tracker.observation_patrol.stations if s.get("low")]
        self.assertTrue(low, "test premise: low stations in ptz_presets.json")
        self.assertTrue(all(s["min_results"] >= 2 for s in low))

    def test_21e_the_frame_tracker_restarts_on_the_ai_box_not_on_a_guess(self):
        """17:50:15: re-anchored at a predicted x (a bush) instead of the AI box at x=1748;
        the camera then followed a branch in the wind."""
        import cv2
        from tracking.multi_object_tracker import camera_shift, shift_box
        rng = np.random.default_rng(4)
        frame = cv2.cvtColor(cv2.resize(rng.integers(40, 200, (270, 480), dtype=np.uint8), (W, H),
                                        interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        cam, tracker = self.tracker_at(0.4012, tilt=-0.8147, zoom=0.0206)
        self.fresh(tracker, cam, [animal([1719, 507, 1777, 598], tid=227)], 10.0)
        tracker._world_vel = 0.02                         # a made-up speed from lock hops
        rt = Runtime()
        S = SimpleNamespace()
        tracker.pose_hint = (0.3771, -0.8222, 0.0206)    # the camera has turned since that picture
        res = {"frame": frame, "captured": 9.0, "pose": (0.4012, -0.8147, 0.0206)}
        rt._visual_follow(tracker, frame, 10.5, 2, res, W, H, 10.5, S)
        dx, k, dy = camera_shift((0.4012, -0.8147, 0.0206), tracker.pose_hint, W)
        exp = shift_box([1719, 507, 1777, 598], dx, k, W, H, dy)
        got = rt.flow.box
        self.assertIsNotNone(got)
        self.assertAlmostEqual((got[0] + got[2]) / 2, (exp[0] + exp[2]) / 2, delta=3.0,
                               msg="the frame tracker was not put on the animal the AI saw")


class Field23OctTests(_Isolated):
    def test_22_a_large_correction_from_the_visual_tracker_reaches_the_camera(self):
        """23/10 12:02-12:04 (a herd of antelope): the camera jerked instead of settling on the
        animal.  The optical-flow box is numpy.float32, so the pulse was too, and the camera's
        timedelta() refused every pulse >= 0.6 s: 53 of 106 flow moves failed and each failure
        stopped the camera.  Small corrections went through, the large ones never did."""
        from datetime import timedelta
        cam = SimCamera(pan=0.33, tilt=-0.80, zoom=0.1211)
        real_move = cam.move

        def strict_move(pan, tilt, zoom=0.0, prevalidated=False, travel_mode=False, pulse_seconds=None):
            timedelta(seconds=max(1.2, (pulse_seconds or 0.0) + 0.6))      # what the ONVIF camera builds
            return real_move(pan, tilt, zoom, prevalidated=prevalidated, travel_mode=travel_mode,
                             pulse_seconds=pulse_seconds)
        cam.move = strict_move
        tracker = PTZTracker(cam)
        tracker.observation_patrol = None
        tracker.pose_hint = cam.get_pose()
        tracker.update([animal([920, 500, 1000, 560])], W, H, fresh=True, now=10.0,
                       capture_pose=cam.get_pose(), capture_time=10.0)
        box = np.array([1450, 820, 1530, 880], np.float32)                   # far right and low: big pulse
        tracker.follow_box(box, W, H, now=12.0)
        self.assertTrue(tracker.last_ptz.get("ok"), tracker.last_ptz)
        self.assertGreaterEqual(tracker.last_ptz["cmd"][3], 0.6, "the test needs a pulse >= 0.6 s")
        self.assertIs(type(tracker.last_ptz["cmd"][3]), float)


class Field26OctTests(_Isolated):
    def test_23_the_approach_centres_where_the_animal_was_seen_not_the_live_spot(self):
        """26/10 13:09 (pipeline_trace + ptz_trace): a baboon 66 px, confirmed from a frame
        taken at pan 0.329 while the patrol travelled on.  The approach used the live
        picture (pan 0.345, ~330 px further) as the reference for the box, turned the camera
        0.03 past the baboon and zoomed it out of the 11.6x view; 4 empty results, patrol."""
        import csv
        from main import Runtime
        cam = SimCamera(pan=0.3453, tilt=-0.80, zoom=0.03)
        rng = np.random.default_rng(5)
        scene = cv2.GaussianBlur(rng.integers(0, 255, (H, W + 600, 3), dtype=np.uint8), (0, 0), 3)
        analysed = scene[:, 400:400 + W].copy()                 # what the AI looked at (pan 0.329)
        live = scene[:, 70:70 + W].copy()                       # now: the picture moved 330 px
        baboon = animal([469, 491, 522, 557], tid=19, conf=0.74, species="unidentified")
        res = dict(frame=analysed, pose=(0.3288, -0.80, 0.03), epoch=-1, seq=1, captured=10.0, done_at=18.3,
                   tracks=[baboon], metrics={"conclusive": True, "inference_seconds": 8.3})
        worker = Mock(busy=True, take_result=Mock(return_value=res), last_seconds=8.3, last_stages=None)
        S = SimpleNamespace(last_seq=-1, last_motion=0.0, last_epoch=-1, tracks=[], result_pose=None, result_t=0.0,
                            full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0, prev_done=0.0,
                            health=None, allowed=False, epoch=-1, soft_since=0.0)
        trace_f = open(os.path.join(self.tmp, "t.csv"), "w", newline="")
        self.addCleanup(trace_f.close)
        rt, tracker = Runtime(), PTZTracker(cam)
        tracker.pose_hint = cam.get_pose()
        with patch("main.write_health", Mock()), patch("main.log_detection", Mock()), patch("main.save_review", Mock()), \
                patch("main.maybe_save_negative_sample", Mock()), patch.object(config, "OPERATOR_DETECT", False):
            grabber = SimpleNamespace(latest=lambda: (live, 18.4, 2), ok=True)
            rt._tick(cam, grabber, worker, tracker, tracker.observation_patrol, None, None, S,
                     csv.writer(trace_f), trace_f, 18.4)
        self.assertIsInstance(rt.manual, ManualLook, "the confirmed small animal was not approached")
        self.assertTrue(rt.manual.approach)
        sx, sy, resp = rt.manual._shift(live)
        self.assertGreater(resp, 0.05)
        self.assertAlmostEqual(rt.manual.point[0] + sx, 495.5 + 330, delta=15,
                               msg="the approach aims beside the baboon (the live spot of an old box)")


class ArcEndZoomTests(_Isolated):
    """26/10 17:19-17:25: six zebras drinking LEFT of the water-hole, beyond the arc end (0.735,
    the lodge pool behind it).  The camera held at pan 0.7291 - right - but its zoom pumped:
    "group centre off-centre -> out", "group small and centred -> in", and it drifted from
    47x out to 6x.  Boxes here follow the optical zoom about the picture's centre."""

    HERD = [[40, 460, 110, 515], [193, 450, 275, 520], [390, 540, 480, 600], [550, 540, 650, 598],
            [700, 537, 800, 600], [880, 455, 940, 510]]

    def run_herd(self, herd, results=12, zoom=0.144, lock=2):
        cam = SimCamera(pan=0.7291, tilt=-1.0, zoom=zoom)
        tracker = PTZTracker(cam)
        m0 = 1 + config.PTZ_ZOOM_MAG_PER_UNIT * zoom
        zooms = []
        for i in range(results):
            k = (1 + config.PTZ_ZOOM_MAG_PER_UNIT * cam.zoom) / m0
            dets = [animal([W / 2 + k * (b[0] - W / 2), H / 2 + k * (b[1] - H / 2),
                            W / 2 + k * (b[2] - W / 2), H / 2 + k * (b[3] - H / 2)],
                           tid=10 + j, conf=0.85 if j == lock else 0.7, species="plains zebra")
                    for j, b in enumerate(herd)]
            dets = [d for d in dets if d["box"][2] > 0 and d["box"][0] < W]
            tracker.pose_hint = cam.get_pose()
            tracker.update(dets, W, H, fresh=True, now=100.0 + 6.0 * i, capture_pose=cam.get_pose())
            cam.advance(6.0)
            zooms.append(round(cam.zoom, 4))
        return cam, tracker, zooms

    def test_25_resting_zebras_beyond_the_arc_end_are_not_zoomed_away_from(self):
        cam, tracker, zooms = self.run_herd(self.HERD)
        self.assertTrue(tracker.locked)
        self.assertAlmostEqual(cam.pan, 0.7291, delta=0.004, msg="the camera left the arc end")
        steps = [b - a for a, b in zip(zooms, zooms[1:]) if b != a]
        self.assertFalse(any(s < 0 for s in steps), f"zoomed OUT of resting zebras: {zooms}")
        self.assertGreater(zooms[-1], 0.144, f"never came closer: {zooms}")
        k = (1 + config.PTZ_ZOOM_MAG_PER_UNIT * zooms[-1]) / (1 + config.PTZ_ZOOM_MAG_PER_UNIT * 0.144)
        x1 = W / 2 + k * (self.HERD[2][0] - W / 2)
        self.assertGreater(x1, 0.04 * W, "the followed zebra was zoomed out of the picture")

    def test_25b_a_zebra_walking_out_at_the_arc_end_widens_the_view(self):
        herd = [[30, 500, 120, 580]]                       # the followed animal at the picture's edge
        cam, tracker, zooms = self.run_herd(herd, results=3, zoom=0.2, lock=0)
        self.assertLess(zooms[-1], 0.2, f"the camera cannot turn, so it must widen: {zooms}")


class SteadyFollowTests(_Isolated):
    def test_26_grazing_zebras_bobbing_up_and_down_do_not_nod_the_camera(self):
        """26/10 17:25-17:49: 166 tilt pulses in 24 min for vertical offsets of +-0.09 (heads
        down to drink and up, the group's members changing) - the camera nodded up and down
        and every move kept the AI waiting for a still picture."""
        cam = SimCamera(pan=0.50, tilt=-1.0, zoom=0.15)
        tracker = PTZTracker(cam)
        tracker.observation_patrol = None
        for i in range(10):
            off = 50 if i % 2 else -50                       # +-0.09 of the half height
            box = [W / 2 - 90, H / 2 + off - 70, W / 2 + 90, H / 2 + off + 70]
            tracker.pose_hint = cam.get_pose()
            tracker.update([animal(box, tid=5, species="plains zebra")], W, H, fresh=True, now=10.0 + 5 * i)
            cam.advance(0.2)
        tilts = [e for e in pulses(cam) if e[2] != 0.0]
        self.assertEqual(tilts, [], "the camera nodded at a resting animal")
        self.assertTrue(tracker.locked)


class OperatorDetectionTests(_Isolated):
    """main.Runtime._watch_operator: a person steering the camera vs our own moves."""

    def setUp(self):
        super().setUp()
        p = patch.object(config, "OPERATOR_DETECT", True)
        p.start()
        self.addCleanup(p.stop)
        self.cam, self.tracker, self.patrol, self.rt, self.S = self.machine(SimCamera(pan=0.2818, tilt=-1.0, zoom=0.03))

    def watch(self, pose, t0):
        self.rt._watch_operator(self.cam, self.tracker, self.patrol, pose, t0, self.S)

    def test_17_a_stalled_loop_during_our_own_tilt_glide_is_not_a_hand(self):
        """21/10 09:00:16-20 (ptz_trace.csv): the control loop stalled 3 s during the patrol's
        tilt glide to low_7; the camera slid on under OUR last command (-0.944 -> -0.840) and
        the next tick took it for a hand - the patrol stood still for 5 min."""
        self.cam.last_command_time = 100.0 + 1.0           # velocity_move(timeout=1) at t=100
        self.watch((0.2818, -0.9442, 0.03), 100.0)
        # no tick for 3 s; the camera finished OUR glide meanwhile
        for k in range(8):
            self.watch((0.2818, -0.8404, 0.03), 103.0 + k * 0.125)
        self.assertIsNone(self.rt.manual, "our own glide (seen late) was taken for a person")

    def test_17b_a_real_hand_movement_is_still_detected(self):
        self.cam.last_command_time = 50.0                  # nothing sent for a long time
        self.watch((0.4000, -1.0, 0.03), 100.0)
        self.watch((0.4000, -1.0, 0.03), 100.125)
        self.watch((0.4200, -1.0, 0.03), 100.25)           # someone turns the camera
        self.assertIsInstance(self.rt.manual, OperatorHold)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
