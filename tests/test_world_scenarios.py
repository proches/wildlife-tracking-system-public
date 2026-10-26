"""End-to-end scenarios: the REAL control loop (main.Runtime._tick) against a simulated world.

19/10/2026: animals were detected and then lost again and again - the tracker
turned the wrong way, zoomed into blur, dropped targets between slow AI results
and handed the camera back to the patrol.  Each of those was fixed on its own;
these scenarios check the whole chain at once:

    far small animal -> noticed -> confirmed -> centred + zoomed (sharp) ->
    followed while it walks -> view held after it disappears -> patrol again.

The world: a textured panorama (so the image-based centring works), picture
geometry measured on the real camera (pan up = picture moves right), a slow AI
(3 s per frame, like the i5) whose detector only sees the animal when it is big
enough, scores it by its size and sometimes misses it.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import config
from main import Runtime
from tests.sim_camera import SimCamera, measured_pan_rate
from tracking.multi_object_tracker import MultiObjectTracker
from tracking.ptz_tracker import PTZTracker

W, H = 1920, 1080
KX, KY = config.PAN_PIXELS_PER_UNIT_AT_1X, 2600.0


def mag(zoom):
    return 1.0 + config.PTZ_ZOOM_MAG_PER_UNIT * zoom


def slow_down(zoom):
    """Measured 19/10: beyond the proportional slow-down, ~30 % slower again at 20x."""
    return max(0.6, 1.0 - 1.7 * max(0.0, zoom - 0.03))


def measured_tilt_rate(speed):
    """Measured 19/10: <=0.22 crawls at 0.0049/s, >=0.25 runs at 0.069/s."""
    a = abs(speed)
    r = 0.0 if a < 0.12 else (0.0049 if a < 0.223 else 0.069)
    return r if speed >= 0 else -r


class WorldCam(SimCamera):
    """SimCamera whose pan slows down when zoomed in (as the real one does)."""

    def advance(self, dt):
        if self.abs_target is None:
            if self.t + dt > self.vel_until:
                self.vel = (0.0, 0.0, 0.0)
            p, ti, z = self.vel
            self.t += dt
            k = 3.85 / mag(self.zoom) * slow_down(self.zoom)
            self.pan += measured_pan_rate(p) * dt * k
            self.tilt = max(-1.0, min(1.0, self.tilt + measured_tilt_rate(ti) * dt * k))
            self.zoom = max(0.0, min(1.0, self.zoom + z * self.ZOOM_RATE * dt))
            return
        super().advance(dt)
        tp, tt, tz = self.abs_target
        if (abs(self.pan - tp) < 1e-6 and abs(self.tilt - tt) < 1e-6
                and (tz is None or abs(self.zoom - tz) < 1e-6)):
            self.abs_target = None      # reached: the camera stops

    def move(self, pan_speed=0.0, tilt_speed=0.0, zoom_speed=0.0, prevalidated=False,
             travel_mode=False, pulse_seconds=None):
        pan_speed = self.guard_pan_speed(pan_speed) if not prevalidated else pan_speed
        dur = pulse_seconds if pulse_seconds is not None else config.BOUNDARY_PAN_PULSE_SECONDS
        self.log.append(("pulse", pan_speed, tilt_speed, zoom_speed, dur))
        self.abs_target = None          # a new move ends any preset / absolute move (real camera)
        k = 3.85 / mag(self.zoom) * slow_down(self.zoom)
        # start-up delays measured on the real camera (pan ~0.15 s, tilt ~0.32 s)
        self.pan += measured_pan_rate(pan_speed) * max(0.0, dur - 0.15) * k
        self.tilt = max(-1.0, min(1.0, self.tilt + measured_tilt_rate(tilt_speed) * max(0.0, dur - 0.32) * k))
        self.zoom = max(0.0, min(1.0, self.zoom + zoom_speed * self.ZOOM_RATE * config.ZOOM_PULSE_SECONDS))

    def zoom_only(self, zoom):
        self.abs_target = (self.pan, self.tilt, float(zoom))
        return True

    presets = {}

    def goto_preset_safe(self, token, expected_pan, pose=None):
        pose = pose or self.get_pose()
        if abs(pose[0] - float(expected_pan)) > config.PRESET_MAX_APPROACH_PAN:
            return False
        pan, tilt, zoom = self.presets[str(token)]
        self.abs_target = (pan, tilt, zoom)
        self.vel = (0.0, 0.0, 0.0)
        return True


class World:
    def __init__(self, cam, animal_pan, animal_tilt, size_1x=6.0, seed=1):
        self.cam = cam
        self.ax, self.ay = animal_pan, animal_tilt
        self.size_1x = size_1x
        self.present = True
        self.walk = 0.0                  # pan units per second
        self.rng = np.random.default_rng(seed)
        small = self.rng.integers(40, 220, (220, 420), dtype=np.uint8)
        tex = cv2.resize(small, (8400, 4400), interpolation=cv2.INTER_CUBIC)
        self.tex = cv2.GaussianBlur(tex, (0, 0), 3)
        fur = self.rng.integers(10, 90, (12, 15), dtype=np.uint8)            # darker than the ground, textured
        self.fur = cv2.cvtColor(cv2.resize(fur, (60, 48), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)

    def screen(self, pose):
        pan, tilt, zoom = pose
        m = mag(zoom)
        x = W / 2 + (pan - self.ax) * KX * m
        y = H / 2 + (tilt - self.ay) * KY * m
        return x, y, self.size_1x * m

    def frame(self):
        pan, tilt, zoom = self.cam.get_pose()
        m = mag(zoom)
        u0, v0 = 6000.0, 1200.0
        mat = np.float32([[-2.0 / m, 0, pan * KX + (W / 2) / m + u0 - 3000],      # half-res render
                          [0, 2.0 / m, -tilt * KY - (H / 2) / m + v0 - 2400]])
        img = cv2.warpAffine(self.tex, mat, (W // 2, H // 2), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self.present:                     # the animal itself, with some texture (fur, legs, shade)
            x, y, s = self.screen((pan, tilt, zoom))
            hw, hh = int(s / 2), int(s * 0.4)
            x1, y1, x2, y2 = int(x) - hw, int(y) - hh, int(x) + hw, int(y) + hh
            if x2 > 0 and y2 > 0 and x1 < W and y1 < H and hw >= 2 and hh >= 2:
                patch = cv2.resize(self.fur, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                out[cy1:cy2, cx1:cx2] = patch[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
        return out

    def detect(self, pose):
        if not self.present:
            return []
        x, y, s = self.screen(pose)
        if s < 18 or not (0 <= x < W and 0 <= y < H):
            return []
        if self.rng.random() < 0.2:          # the AI misses it now and then
            return []
        conf = float(min(0.9, 0.30 + s / 200.0))
        # SpeciesNet as measured 21/10: it NAMED 6 of the 7 real animals seen during the sweep;
        # like the pipeline (_doubtful_label), a name on a tiny box in a wide view is no proof
        named = self.rng.random() < self.NAMED_RATE
        trusted = not (pose[2] < config.ZOOMED_VIEW_MIN_ZOOM and s < config.FAR_TINY_BOX_PX)
        return [{"box": [x - s / 2, y - s * 0.4, x + s / 2, y + s * 0.4], "class_id": 0, "det_confidence": conf,
                 "species": "african buffalo" if named else "unidentified",
                 "species_confidence": 0.8 if named else 0.3, "species_trusted": trusted,
                 "source": "megadetector", "site_model_active": True, "in_water_zone": False}]

    NAMED_RATE = 0.8


class SlowWorker:
    """AI thread stand-in: one frame at a time, AI_SECONDS per frame."""
    AI_SECONDS = 3.0

    def __init__(self, clock, world):
        self.clock, self.world = clock, world
        self.job, self.done_at = None, 0.0
        self.tracks = MultiObjectTracker()
        self.last_seconds, self.last_stages = self.AI_SECONDS, None

    @property
    def busy(self):
        return self.job is not None

    def submit(self, job):
        if self.job is not None:
            return False
        # the AI sees the picture as it was when it was CAPTURED (results arrive later)
        self.job, self.done_at = job, self.clock() + self.AI_SECONDS
        self.job_dets = self.world.detect(job["pose"])
        return True

    def request_reset(self):
        self.tracks.reset()

    def take_result(self):
        if self.job is None or self.clock() < self.done_at:
            return None
        job, self.job = self.job, None
        dets = self.job_dets
        tracks = self.tracks.update(dets, W, H, pose=job["pose"])
        return dict(job, tracks=tracks, metrics={"inference_seconds": self.AI_SECONDS, "conclusive": True},
                    done_at=self.clock())


class WorldScenarioTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        for name, val in (("PTZ_CAPABILITIES_FILE", os.path.join(tmp, "caps.json")),
                          ("PTZ_TRACE_CSV", os.path.join(tmp, "trace.csv")),
                          ("METRICS_JSONL", os.path.join(tmp, "m.jsonl")),
                          ("MOTION_CUE_ENABLED", False), ("OPERATOR_DETECT", False),
                          ("PATROL_TILT_GLIDE", False)):
            p = patch.object(config, name, val)
            p.start()
            self.addCleanup(p.stop)
        for name in ("log_detection", "save_review", "maybe_save_negative_sample", "write_health"):
            p = patch(f"main.{name}", Mock(return_value=None))
            p.start()
            self.addCleanup(p.stop)

    def run_world(self, animal_pan=0.37, seconds=300, events=None, start_pan=0.385):
        # the REAL station presets (ptz_presets.json); plain views at the -1.0 tilt reading
        import json
        real = json.load(open(os.path.join(os.path.dirname(__file__), "..", "ptz_presets.json")))
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "presets.json")
        json.dump(real, open(path, "w"))
        pp = patch.object(config, "PTZ_PRESETS_FILE", path)
        pp.start()
        self.addCleanup(pp.stop)
        WorldCam.presets = {str(v["token"]): (float(v["pan"]), float(v.get("tilt", -1.0)), float(v["zoom"]))
                            for v in real.values()}
        now = [1000.0]
        clock = lambda: now[0]
        p = patch("time.monotonic", clock)
        p.start()
        self.addCleanup(p.stop)
        cam = WorldCam(pan=start_pan, tilt=-1.0, zoom=0.03)
        world = World(cam, animal_pan, -0.99)        # a little above the centre of the plain view
        tracker = PTZTracker(cam)
        tracker.observation_patrol.clock = clock
        tracker.observation_patrol.motion.clock = clock
        tracker.observation_patrol.focus = None
        rt = Runtime()
        S = SimpleNamespace(last_seq=-1, last_motion=clock(), last_epoch=-1, tracks=[], result_pose=None,
                            result_t=0.0, full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0,
                            prev_done=clock(), health=None, allowed=False, epoch=-1, soft_since=0.0)
        trace_f = open(config.PTZ_TRACE_CSV, "w", newline="")
        self.addCleanup(trace_f.close)
        import csv
        trace = csv.writer(trace_f)
        seq = [0]

        def latest():
            seq[0] += 1
            return world.frame(), clock() - 0.05, seq[0]
        grabber = SimpleNamespace(latest=latest, ok=True)
        focus = Mock(last_result="", busy=False, enabled=False)
        worker = SlowWorker(clock, world)
        log = []
        dt = 0.125
        for i in range(int(seconds / dt)):
            now[0] += dt
            t = now[0] - 1000.0
            for at, fn in (events or []):
                if abs(t - at) < dt / 2:
                    fn(world)
            world.ax += world.walk * dt
            cam.advance(dt)
            rt._tick(cam, grabber, worker, tracker, tracker.observation_patrol, focus, None, S, trace, trace_f, now[0])
            x, y, s = world.screen(cam.get_pose())
            log.append(dict(t=t, x=x, y=y, s=s, zoom=cam.zoom, locked=tracker.locked,
                            manual=rt.manual is not None, phase=tracker.observation_patrol.phase,
                            searching=tracker.searching, present=world.present))
        return log

    def test_far_animal_is_approached_kept_sharp_and_watched(self):
        log = self.run_world(seconds=240)
        first_lock = next((r["t"] for r in log if r["locked"]), None)
        self.assertIsNotNone(first_lock, "the far animal was never locked on")
        self.assertLess(first_lock, 120)
        after = [r for r in log if r["t"] > first_lock + 40]
        in_frame = sum(1 for r in after if 0 <= r["x"] < W and 0 <= r["y"] < H) / len(after)
        centred = sum(1 for r in after if abs(r["x"] - W / 2) < 0.25 * W and abs(r["y"] - H / 2) < 0.3 * H) / len(after)
        self.assertGreater(in_frame, 0.95, "the animal left the picture")
        self.assertGreater(centred, 0.80, "the animal was not kept in the centre")
        self.assertGreater(max(r["s"] for r in after), 55, "never zoomed in close enough")
        self.assertLessEqual(max(r["zoom"] for r in log), max(config.TRACK_MAX_ZOOM, config.LOOK_MAX_ZOOM) + 0.02,
                             "zoomed in beyond the sharp limit")
        self.assertFalse(any(r["searching"] and not r["locked"] and not r["manual"] for r in after),
                         "went back to patrol while the animal was still there")

    def test_walking_animal_is_followed(self):
        def start_walking(w):
            w.walk = 0.0012                     # ~6 px/s at 1x, ~25-120 px/s on screen
        log = self.run_world(seconds=300, events=[(150.0, start_walking)])
        walk = [r for r in log if r["t"] > 160]
        in_frame = sum(1 for r in walk if 0 <= r["x"] < W and 0 <= r["y"] < H) / len(walk)
        self.assertGreater(in_frame, 0.90, "lost the walking animal")
        self.assertTrue(all(r["locked"] or r["manual"] for r in walk[-80:]), "stopped following it")

    def _walk_case(self, animal_pan, walk, seed):
        orig = World.__init__

        def init(w, cam, a, b, size_1x=6.0, seed=seed):
            orig(w, cam, a, b, size_1x=size_1x, seed=seed)
        p = patch.object(World, "__init__", init)
        p.start()
        self.addCleanup(p.stop)
        log = self.run_world(animal_pan=animal_pan, seconds=300,
                             events=[(150.0, lambda w: setattr(w, "walk", walk))])
        walking = [r for r in log if r["t"] > 160]
        return sum(1 for r in walking if 0 <= r["x"] < W and 0 <= r["y"] < H) / len(walking)

    def test_animal_walking_left_is_followed(self):
        self.assertGreater(self._walk_case(0.40, -0.0012, 2), 0.85)

    def test_animal_next_to_a_station_walking_right_is_followed(self):
        self.assertGreater(self._walk_case(0.405, 0.0015, 6), 0.85)

    def test_view_is_held_after_the_animal_disappears_then_patrol(self):
        def leave(w):
            w.present = False
        log = self.run_world(seconds=330, events=[(180.0, leave)])
        gone = [r for r in log if r["t"] > 180.0]
        t_unlock = next(r["t"] for r in gone if not r["locked"] and not r["manual"])
        start_pan_moves = [r for r in gone if r["t"] < t_unlock + config.LOST_HOLD_SECONDS - 5]
        xs = [r["x"] for r in start_pan_moves]
        self.assertLess(max(xs) - min(xs), 60, "the camera left the spot right after losing the animal")
        self.assertTrue(any(r["searching"] for r in gone if r["t"] > t_unlock + config.LOST_HOLD_SECONDS + 5),
                        "the patrol never resumed")


if __name__ == "__main__":
    unittest.main()
