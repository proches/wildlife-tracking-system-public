"""Birds flying past the camera (26/10/2026).

24-26/10: 10 bird detections in 16 156 AI results; checked by eye, 4 of 8 were real birds -
all four FLYING past, each seen by one AI result and dropped (confirmation needs two) - and
4 were trees / stumps, two of them at night.  tracking/flight.py tells a bird in flight from
a tree, a stump and a branch in the wind; main.Runtime._birds_in_flight logs and saves such a
bird without following it; DetectionWorker drops night "birds".
"""

import csv
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import config
from main import DetectionWorker, FrameGrabber, Runtime
from tests.sim_camera import SimCamera
from tracking.flight import flight_score, is_bird, to_small
from tracking.ptz_tracker import PTZTracker

W, H = 1920, 1080


def savanna(seed=1, w=W + 400, h=H + 200):
    """Grass / bush texture of the water-hole view, a little larger than one frame."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.normal(120, 40, (h, w)).astype(np.float32), (0, 0), 4)
    fine = cv2.GaussianBlur(rng.normal(0, 12, (h, w)).astype(np.float32), (0, 0), 1)
    grey = np.clip(base + fine, 0, 255)
    return cv2.merge([grey * 0.8, grey * 0.95, grey]).astype(np.uint8)


def view(scene, ox, oy=100, seed=0, noise=2.5):
    """The camera picture: a window of the scene, with sensor noise."""
    rng = np.random.default_rng(seed)
    out = scene[oy:oy + H, ox:ox + W].astype(np.float32)
    out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_bird(img, cx, cy, size, wings_up=True):
    """A dark bird with spread wings, about size px across."""
    s = size / 2.0
    pts = np.array([[cx - s, cy - (0.35 * s if wings_up else -0.2 * s)], [cx - 0.2 * s, cy],
                    [cx, cy - 0.15 * s], [cx + 0.2 * s, cy],
                    [cx + s, cy - (0.35 * s if wings_up else -0.2 * s)], [cx, cy + 0.25 * s]], np.int32)
    cv2.fillPoly(img, [pts], (35, 30, 28))
    cv2.ellipse(img, (int(cx), int(cy)), (max(2, int(0.18 * s)), max(2, int(0.12 * s))), 0, 0, 360,
                (25, 22, 20), -1)
    return [cx - s, cy - 0.4 * s, cx + s, cy + 0.3 * s]


def small(frame):
    return to_small(frame)


class IsBirdTests(unittest.TestCase):
    def test_site_and_speciesnet_bird_names(self):
        for name in ("bird", "Bird", "superb starling", "helmeted guineafowl", "marabou stork",
                     "white-backed vulture", "aves", "lilac-breasted roller", "grey crowned crane"):
            self.assertTrue(is_bird(name), name)

    def test_mammals_and_empty_are_not_birds(self):
        for name in (None, "", "baboon", "olive baboon", "african buffalo", "unidentified", "other",
                     "plains zebra", "thomson's gazelle", "warthog", "human"):
            self.assertFalse(is_bird(name), name)


class FlightScoreTests(unittest.TestCase):
    """flight_score on pictures built like the water-hole view (texture + sensor noise)."""

    def setUp(self):
        self.scene = savanna()

    def pair(self, ox_a, ox_b, bird_a=None, bird_b=None, size=60, seeds=(1, 2)):
        a, b = view(self.scene, ox_a, seed=seeds[0]), view(self.scene, ox_b, seed=seeds[1])
        box = None
        if bird_a is not None:
            box = draw_bird(a, *bird_a, size)
        if bird_b is not None:
            draw_bird(b, *bird_b, size, wings_up=False)
        return a, b, box

    def test_a_bird_flying_past_a_still_camera(self):
        a, b, box = self.pair(200, 200, bird_a=(900, 400), bird_b=(990, 380))
        score, dt = flight_score(box, small(a), [(0.5, small(b))])
        self.assertGreaterEqual(score, config.FLIGHT_MIN_RATIO)
        self.assertEqual(dt, 0.5)

    def test_a_bird_flying_past_while_the_patrol_travels(self):
        """The patrol turns ~25 px per 0.25 s (cine speed): the camera's own motion is removed."""
        a, b, box = self.pair(200, 224, bird_a=(700, 500), bird_b=(620, 470))
        score, _ = flight_score(box, small(a), [(-0.25, small(b))])
        self.assertGreaterEqual(score, config.FLIGHT_MIN_RATIO)

    def test_a_small_far_bird(self):
        """24/10: the smallest real bird was 27 px wide."""
        a, b, box = self.pair(200, 210, bird_a=(1500, 300), bird_b=(1530, 300), size=28)
        score, _ = flight_score(box, small(a), [(0.25, small(b))])
        self.assertGreaterEqual(score, config.FLIGHT_MIN_RATIO)

    def test_a_big_close_bird_leaving_the_frame(self):
        """24/10 the biggest was 448 px: half a second later it was out of the picture."""
        a, b, box = self.pair(200, 200, bird_a=(1700, 500), size=440)
        score, _ = flight_score(box, small(a), [(0.5, small(b))])
        self.assertGreaterEqual(score, config.FLIGHT_MIN_RATIO)

    def test_the_best_of_several_neighbours_counts(self):
        a, b, box = self.pair(200, 200, bird_a=(900, 400), bird_b=(1000, 400))
        still = view(self.scene, 200, seed=7)
        draw_bird(still, 902, 400, 60)              # 0.2 s: the bird has hardly moved yet
        score, dt = flight_score(box, small(a), [(0.2, small(still)), (0.75, small(b))])
        self.assertGreaterEqual(score, config.FLIGHT_MIN_RATIO)
        self.assertEqual(dt, 0.75)

    def test_a_tree_or_stump_called_bird_is_not_flying(self):
        """24-26/10: 4 of 8 "birds" were trees, branches and stumps - nothing moved."""
        a, b, _ = self.pair(200, 213)
        stump = [880, 380, 960, 440]
        score, _ = flight_score(stump, small(a), [(0.25, small(b)), (-0.5, small(view(self.scene, 190, seed=3)))])
        self.assertEqual(score, 0.0)

    def test_a_dark_stump_called_bird_is_not_flying(self):
        a, b, _ = self.pair(200, 200)
        box = draw_bird(a, 900, 400, 60)
        draw_bird(b, 900, 400, 60)                  # the same dark shape, still there
        score, _ = flight_score(box, small(a), [(0.5, small(b))])
        self.assertEqual(score, 0.0)

    def test_a_branch_swaying_in_the_wind_is_not_flying(self):
        """The dark shape shifted 6 px: it changed a lot, but it is still there."""
        a, b, _ = self.pair(200, 200)
        box = draw_bird(a, 900, 400, 80)
        draw_bird(b, 906, 403, 80)
        score, _ = flight_score(box, small(a), [(0.5, small(b))])
        self.assertEqual(score, 0.0)

    def test_a_zoom_step_gives_no_verdict(self):
        """Between the two frames the camera zoomed: everything changed, the box no more than the rest."""
        a, _, box = self.pair(200, 200, bird_a=(900, 400))
        b = cv2.resize(a, None, fx=1.4, fy=1.4)[200:200 + H, 300:300 + W]
        draw_bird(b, 900, 400, 84)
        score, _ = flight_score(box, small(a), [(0.5, small(b))])
        self.assertEqual(score, 0.0)

    def test_nothing_to_compare(self):
        a, _, box = self.pair(200, 200, bird_a=(900, 400))
        self.assertEqual(flight_score(box, small(a), []), (0.0, None))
        self.assertEqual(flight_score([5, 5, 6, 6], small(a), [(0.5, small(a))]), (0.0, None))


class FrameRingTests(unittest.TestCase):
    def test_around_returns_frames_within_the_window_on_both_sides(self):
        g = FrameGrabber("rtsp://x", threading.Event(), Mock())
        for k in range(40):
            g._ring.append((100.0 + k * 0.25, np.full((4, 4), k, np.uint8)))
        got = g.around(105.0, 0.2, 1.0)
        self.assertEqual(sorted(round(dt, 2) for dt, _ in got), [-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0])

    def test_the_ring_is_bounded(self):
        g = FrameGrabber("rtsp://x", threading.Event(), Mock())
        self.assertEqual(g._ring.maxlen, int(config.FLIGHT_RING_SECONDS * config.FLIGHT_RING_FPS))
        self.assertGreater(config.FLIGHT_RING_SECONDS, 13.0 + config.FLIGHT_NEIGHBOUR_MAX_S,
                           "the slowest AI results (13 s) would find no frame after theirs")


class NightBirdTests(unittest.TestCase):
    """DetectionWorker: a "bird" in a night frame is dropped before the tracker sees it."""

    def run_job(self, low_light, dets):
        pipe = Mock()
        pipe.process_frame.return_value = [dict(d) for d in dets]
        pipe.last_metrics = {"frame_quality": {"low_light": low_light}, "inference_seconds": 1.0}
        stop = threading.Event()
        w = DetectionWorker(pipe, stop)
        w.submit(dict(frame=np.zeros((H, W, 3), np.uint8), scan_mode="full", pose=(0.4, -1.0, 0.03), roi=None,
                      deep=True, epoch=1, seq=1, captured=1.0))
        w.start()
        res = None
        for _ in range(200):
            res = w.take_result()
            if res is not None:
                break
            stop.wait(0.02)
        stop.set()
        w.join(2)
        self.assertIsNotNone(res)
        return res

    def test_night_birds_are_dropped(self):
        dets = [{"box": [100, 100, 140, 130], "det_confidence": 0.6, "species": "bird", "class_id": 1},
                {"box": [600, 400, 800, 600], "det_confidence": 0.7, "species": "african buffalo", "class_id": 0}]
        res = self.run_job(True, dets)
        self.assertEqual([d["species"] for d in res["tracks"]], ["african buffalo"])
        self.assertEqual(res["metrics"]["night_birds_dropped"], 1)

    def test_day_birds_are_kept(self):
        dets = [{"box": [100, 100, 140, 130], "det_confidence": 0.6, "species": "bird", "class_id": 1}]
        res = self.run_job(False, dets)
        self.assertEqual([d["species"] for d in res["tracks"]], ["bird"])
        self.assertNotIn("night_birds_dropped", res["metrics"])


class BirdsInFlightTickTests(unittest.TestCase):
    """main.Runtime._tick with one AI result holding a single-shot bird box."""

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.tmp = tmp
        for name, val in (("PTZ_CAPABILITIES_FILE", os.path.join(tmp, "caps.json")),
                          ("PTZ_TRACE_CSV", os.path.join(tmp, "trace.csv")),
                          ("METRICS_JSONL", os.path.join(tmp, "m.jsonl")),
                          ("PIPELINE_TRACE_FILE", os.path.join(tmp, "pipeline_trace.jsonl")),
                          ("STATIC_FP_FILE", os.path.join(tmp, "static.json")),
                          ("EVENTS_CSV", os.path.join(tmp, "events.csv")),
                          ("EVENT_CLIP_DIR", os.path.join(tmp, "clips")),
                          ("REVIEW_DIR", os.path.join(tmp, "review"))):
            p = patch.object(config, name, val)
            p.start()
            self.addCleanup(p.stop)
        self.scene = savanna(4)

    def tick(self, flying=True, low_light=False, conf=0.62, species="bird", confirmed=False, rt=None,
             S=None, seq=1):
        from utils.events import WildlifeEventEngine
        analysed = view(self.scene, 200, seed=1)
        box = draw_bird(analysed, 1100, 300, 50)
        later = view(self.scene, 200, seed=2)
        if not flying:
            draw_bird(later, 1100, 300, 50)
        else:
            draw_bird(later, 1180, 280, 50, wings_up=False)
        ring = [(10.0 + k * 0.25, small(view(self.scene, 200, seed=10 + k))) for k in range(-6, 0)]
        ring.append((10.5, small(later)))
        bird = {"box": box, "class_id": 1, "det_confidence": conf, "species": species, "species_confidence": conf,
                "track_id": 7, "track_confirmed": confirmed, "priority_score": conf}
        res = dict(frame=analysed, pose=(0.45, -1.0, 0.03), epoch=-1, seq=seq, captured=10.0, done_at=16.0,
                   tracks=[bird], metrics={"conclusive": True, "inference_seconds": 6.0,
                                           "frame_quality": {"low_light": low_light}})
        worker = Mock(busy=True, take_result=Mock(return_value=res), last_seconds=6.0, last_stages=None)
        S = S or SimpleNamespace(last_seq=-1, last_motion=0.0, last_epoch=-1, tracks=[], result_pose=None, result_t=0.0,
                                 full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0,
                                 prev_done=0.0, health=None, allowed=False, epoch=-1, soft_since=0.0)
        self.S = S
        cam = SimCamera(pan=0.45, tilt=-1.0, zoom=0.03)
        rt = rt or Runtime()
        tracker = PTZTracker(cam)
        tracker.pose_hint = cam.get_pose()
        events = WildlifeEventEngine()

        def around(t, lo, hi):
            return [(ft - t, img) for ft, img in ring if lo <= abs(ft - t) <= hi]

        grabber = SimpleNamespace(latest=lambda: (later, 16.1, 2), ok=True, around=around)
        trace_f = open(os.path.join(self.tmp, "t.csv"), "w", newline="")
        self.addCleanup(trace_f.close)
        self.log_detection, self.save_review = Mock(), Mock()
        with patch("main.write_health", Mock()), patch("main.log_detection", self.log_detection), \
                patch("main.save_review", self.save_review), patch("main.maybe_save_negative_sample", Mock()), \
                patch.object(config, "OPERATOR_DETECT", False):
            rt._tick(cam, grabber, worker, tracker, tracker.observation_patrol, None, events, S,
                     csv.writer(trace_f), trace_f, 16.1)
        events.close()
        with open(config.EVENTS_CSV, newline="", encoding="utf-8") as f:
            self.events = list(csv.DictReader(f))
        return rt, tracker

    def flight_saves(self):
        return [c for c in self.save_review.call_args_list if c.args[2] == "bird_flight"]

    def test_a_bird_flying_past_is_logged_saved_and_an_event_but_not_followed(self):
        rt, tracker = self.tick()
        self.assertEqual(len(self.flight_saves()), 1, "the flying bird was not saved for review")
        logged = [c.args[0] for c in self.log_detection.call_args_list]
        self.assertEqual(len(logged), 1)
        self.assertTrue(logged[0]["flight"] and logged[0]["track_confirmed"])
        self.assertEqual([(e["type"], e["species"]) for e in self.events], [("BIRD_IN_FLIGHT", "bird")])
        self.assertFalse(tracker.locked, "a bird that is long gone must not be followed")
        self.assertIsNone(rt.manual, "a bird that is long gone must not be approached")

    def test_a_bird_that_stays_is_not_a_bird_in_flight(self):
        self.tick(flying=False)
        self.assertEqual(self.flight_saves(), [])
        self.assertEqual(self.events, [])

    def test_no_flight_check_at_night_for_weak_boxes_or_other_animals(self):
        for kwargs in ({"low_light": True}, {"conf": 0.30}, {"species": "olive baboon"}):
            self.tick(**kwargs)
            self.assertEqual(self.flight_saves(), [], kwargs)
            self.assertEqual(self.events, [], kwargs)

    def test_a_confirmed_bird_goes_the_normal_way(self):
        self.tick(confirmed=True)
        self.assertEqual(self.flight_saves(), [])
        self.assertNotIn("BIRD_IN_FLIGHT", [e["type"] for e in self.events])

    def test_one_bird_is_reported_once(self):
        rt, _ = self.tick()
        self.tick(rt=rt, S=self.S, seq=2)
        self.assertEqual(self.flight_saves(), [], "the same track was reported twice")

    def test_switched_off(self):
        with patch.object(config, "FLIGHT_BIRDS_ENABLED", False):
            self.tick()
        self.assertEqual(self.flight_saves(), [])


if __name__ == "__main__":
    unittest.main()
