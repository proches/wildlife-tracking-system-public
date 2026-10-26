import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config
from detection.pipeline import WildlifePipeline
from detection.validator import StaticFalsePositiveMemory, is_verified, prefilter
from detection.water_zone import WaterZone
from detection.frame_quality import FrameQuality
from tracking.multi_object_tracker import MultiObjectTracker

W, H = 1920, 1080


def det(box, conf=0.5, species=None, sconf=0.0, cls=0):
    return {"box": list(box), "class_id": cls, "det_confidence": conf, "species": species,
            "species_confidence": sconf, "source": "t"}


class PrefilterTests(unittest.TestCase):
    def test_camera_timestamp_and_zoom_label_are_rejected(self):
        # the "10/2026" date text and the "Z: 11.08X" label from the field logs
        self.assertEqual(prefilter(det([20, 10, 300, 70]), W, H)[1], "osd_text")
        self.assertEqual(prefilter(det([1700, 1020, 1900, 1070]), W, H)[1], "osd_text")

    def test_edge_sliver_from_field_log_is_rejected(self):
        # [1903, 349, 1920, 403] 'animal' conf 0.117 sat on the frame edge for hours
        self.assertEqual(prefilter(det([1903.1, 349.7, 1920.0, 403.1]), W, H)[1], "edge_sliver")

    def test_real_animal_passes(self):
        self.assertTrue(prefilter(det([188, 360, 267, 411]), W, H)[0])
        self.assertTrue(prefilter(det([900, 500, 960, 540]), W, H)[0])

    def test_absurd_boxes_rejected(self):
        self.assertFalse(prefilter(det([0, 100, 1900, 1000]), W, H)[0])
        self.assertFalse(prefilter(det([500, 500, 505, 505]), W, H)[0])
        self.assertFalse(prefilter(det([500, 500, 900, 505]), W, H)[0])

    def test_near_field_strip_rejected(self):
        self.assertEqual(prefilter(det([800, 1030, 880, 1075]), W, H)[1], "near_field")


class VerificationTests(unittest.TestCase):
    def test_only_real_confident_species_are_verified(self):
        self.assertTrue(is_verified(det([0, 0, 9, 9], species="zebra", sconf=0.9)))
        self.assertFalse(is_verified(det([0, 0, 9, 9], species="zebra", sconf=0.3)))
        for label in ("unidentified", "other", "blank", "human", None):
            self.assertFalse(is_verified(det([0, 0, 9, 9], species=label, sconf=0.99)))


class StaticMemoryTests(unittest.TestCase):
    def setUp(self):
        self.t = [1000.0]
        self.path = os.path.join(tempfile.mkdtemp(), "static.json")
        self.mem = StaticFalsePositiveMemory(self.path, clock=lambda: self.t[0])
        self.pose = (0.5, -1.0, 0.16)
        self.stump = det([400, 500, 470, 560], conf=0.22, species="unidentified", sconf=0.3)

    def test_unmoving_unidentified_clutter_is_learned_and_persisted(self):
        results = []
        for _ in range(8):
            results.append(self.mem.observe(self.stump, self.pose, W, H))
            self.t[0] += 8
        self.assertFalse(results[0])
        self.assertTrue(results[-1])
        self.mem.save(force=True)
        reloaded = StaticFalsePositiveMemory(self.path, clock=lambda: self.t[0])
        self.assertTrue(reloaded.is_known_static(self.stump, self.pose, W, H))
        # a different camera pose must not inherit the clutter
        self.assertFalse(reloaded.is_known_static(self.stump, (0.3, -1.0, 0.16), W, H))

    def test_verified_animal_is_never_learned_as_clutter(self):
        zebra = det([400, 500, 470, 560], conf=0.8, species="zebra", sconf=0.95)
        for _ in range(12):
            self.assertFalse(self.mem.observe(zebra, self.pose, W, H))
            self.t[0] += 8

    def test_moving_object_is_not_clutter(self):
        for i in range(12):
            moving = det([400 + 60 * i, 500, 470 + 60 * i, 560], conf=0.25, species="unidentified")
            self.assertFalse(self.mem.observe(moving, self.pose, W, H))
            self.t[0] += 8

    def test_real_animal_appearing_on_clutter_spot_clears_it(self):
        for _ in range(8):
            self.mem.observe(self.stump, self.pose, W, H)
            self.t[0] += 8
        self.assertTrue(self.mem.is_known_static(self.stump, self.pose, W, H))
        zebra = det([400, 500, 470, 560], conf=0.8, species="zebra", sconf=0.95)
        self.mem.observe(zebra, self.pose, W, H)
        self.assertFalse(self.mem.is_known_static(self.stump, self.pose, W, H))


class TrackConfirmationTests(unittest.TestCase):
    def feed(self, tracker, n, **kw):
        out = []
        for _ in range(n):
            out = tracker.update([det([300, 300, 360, 350], **kw)], W, H)
        return out

    def test_weak_unidentified_blob_never_confirms(self):
        out = self.feed(MultiObjectTracker(), 12, conf=0.20, species="unidentified")
        self.assertFalse(out[0]["track_confirmed"])

    def test_strong_detection_confirms_after_enough_hits(self):
        t = MultiObjectTracker()
        self.assertFalse(self.feed(t, 2, conf=0.6, species="unidentified")[0]["track_confirmed"])
        self.assertTrue(self.feed(t, 1, conf=0.6, species="unidentified")[0]["track_confirmed"])

    def test_identified_animal_is_believed_immediately(self):
        out = self.feed(MultiObjectTracker(), 1, conf=0.7, species="zebra", sconf=0.9)
        self.assertTrue(out[0]["track_confirmed"])

    def test_moderate_detection_in_water_zone_confirms(self):
        t = MultiObjectTracker()
        out = []
        for _ in range(3):
            d = det([300, 300, 360, 350], conf=0.37, species="unidentified")
            d["in_water_zone"] = True
            out = t.update([d], W, H)
        self.assertTrue(out[0]["track_confirmed"])

    def test_faint_reed_at_water_edge_is_not_confirmed(self):
        """v3.1: 0.27 at the water edge is typically reeds / a bush, not an animal."""
        t = MultiObjectTracker()
        out = []
        for _ in range(6):
            d = det([300, 300, 360, 350], conf=0.27, species="unidentified")
            d["in_water_zone"] = True
            out = t.update([d], W, H)
        self.assertFalse(out[0]["track_confirmed"])


class FakeDetector:
    IMAGE_SIZE = 1280

    def __init__(self, items):
        self.items = items
        self.calls = 0
        self.sizes = []               # inference size seen by each call (PytorchWildlife reads predictor.args)
        self.predictor = SimpleNamespace(args=SimpleNamespace(imgsz=1280))

    def single_image_detection(self, img, det_conf_thres=0.2):
        self.calls += 1
        self.sizes.append(self.predictor.args.imgsz)
        keep = [i for i in self.items if i[1] >= det_conf_thres]
        return {"detections": SimpleNamespace(
            xyxy=np.array([k[0] for k in keep], dtype=float).reshape(-1, 4),
            confidence=np.array([k[1] for k in keep], dtype=float),
            class_id=np.array([k[2] for k in keep], dtype=int))}


class FakeBackend:
    name = "fake"
    cache_seconds = 0.0
    max_new_per_frame = None
    threshold = 0.45

    def __init__(self, table):
        self.table = table

    def classify(self, crop):
        h = crop.shape[0]
        return self.table.get(h, ("unidentified", 0.1))


def make_pipeline(items, table=None, water=None):
    p = WildlifePipeline.__new__(WildlifePipeline)
    p.detector = FakeDetector(items)
    p.custom_detector = None
    p.backend = FakeBackend(table or {})
    p.quality = FrameQuality()
    p.water = water or WaterZone(path=os.path.join(tempfile.mkdtemp(), "none.json"))
    p.static = StaticFalsePositiveMemory(os.path.join(tempfile.mkdtemp(), "s.json"))
    p._cache, p._warned_classify, p._frame_index, p._tile_cursor, p.last_metrics = [], False, 0, 0, {}
    return p


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((H, W, 3), 120, np.uint8)

    def test_ghosts_are_dropped_before_classification(self):
        items = [
            ([20, 10, 300, 70], 0.55, 0),          # camera timestamp
            ([1903, 349, 1920, 403], 0.40, 0),     # edge sliver
            ([500, 500, 560, 540], 0.70, 0),       # real
            ([800, 400, 900, 500], 0.90, 2),       # 'vehicle' on a pole  -> never an animal
            ([100, 100, 160, 160], 0.80, 1),       # person
        ]
        p = make_pipeline(items)
        with patch("config.ENHANCED_DETECTION_ENABLED", False):
            out = p.process_frame(self.frame, scan_mode=True, pose=(0.5, -1, 0.1))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["box"], [500, 500, 560, 540])
        self.assertEqual(p.last_metrics["rejected"], {"osd_text": 1, "edge_sliver": 1})

    def test_confident_blank_label_is_vetoed(self):
        # crop height 40*1.3=52 -> backend says blank 0.6 (>= veto 0.50)
        p = make_pipeline([([500, 500, 560, 540], 0.5, 0)], table={52: ("blank", 0.6)})
        with patch("config.ENHANCED_DETECTION_ENABLED", False):
            self.assertEqual(p.process_frame(self.frame, pose=(0.5, -1, 0.1)), [])
        self.assertEqual(p.last_metrics["dropped_non_wildlife"], 1)

    def test_weak_candidates_survive_for_the_tracker(self):
        p = make_pipeline([([500, 500, 560, 540], 0.20, 0)])
        with patch("config.ENHANCED_DETECTION_ENABLED", False):
            out = p.process_frame(self.frame, pose=(0.5, -1, 0.1))
        self.assertEqual(len(out), 1)       # kept as candidate; confirmation happens in the tracker

    def test_static_clutter_is_learned_then_skipped(self):
        p = make_pipeline([([500, 500, 560, 540], 0.30, 0)])
        clock = [0.0]
        p.static.clock = lambda: clock[0]
        with patch("config.ENHANCED_DETECTION_ENABLED", False):
            seen = []
            for _ in range(10):
                seen.append(len(p.process_frame(self.frame, pose=(0.5, -1, 0.1))))
                clock[0] += 8
        self.assertEqual(seen[0], 1)
        self.assertEqual(seen[-1], 0)

    def test_tiles_run_native_resolution_and_respect_time_budget(self):
        p = make_pipeline([([500, 500, 560, 540], 0.5, 0)])
        tiles = p._tile_rects(W, H)
        self.assertEqual(len(tiles), config.ENHANCED_TILE_COLS * config.ENHANCED_TILE_ROWS)
        for x1, y1, x2, y2 in tiles:
            self.assertLessEqual(x2, W)
            self.assertLessEqual(y2, H)
        with patch("config.ENHANCED_TIME_BUDGET_SECONDS", 0.0):
            p._enhanced_detect(self.frame)
        self.assertEqual(p.detector.calls, 1)   # budget exhausted after the first tile

    def test_empty_scene_also_respects_time_budget(self):
        """v3.1 regression: with nothing detected every tile ran (40 s per frame)."""
        p = make_pipeline([])
        with patch("config.ENHANCED_TIME_BUDGET_SECONDS", 0.0):
            p._enhanced_detect(self.frame)
            self.assertEqual(p.detector.calls, 1)
            p._enhanced_detect(self.frame)      # next frame: the next tile (round-robin)
        self.assertEqual(p.detector.calls, 2)
        self.assertEqual(p._tile_cursor, 2)


class FakeCustom:
    enabled = True

    def __init__(self, items):
        self.items = items

    def detect(self, frame_bgr, imgsz=None, conf=None):
        if frame_bgr.shape[:2] != (H, W):   # magnified crop passes: nothing extra in these scenes
            return []
        return [{"box": list(b), "class_id": 0, "det_confidence": c, "species": None,
                 "species_confidence": None, "source": "custom_yolo", "source_label": lbl}
                for b, c, lbl in self.items]


class SiteModelPrimaryTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((H, W, 3), 120, np.uint8)

    def test_weak_megadetector_only_box_is_dropped_confirmed_box_kept(self):
        bush = ([100, 600, 180, 660], 0.30, 0)          # MD only, weak: a bush
        buffalo = ([900, 500, 980, 560], 0.40, 0)       # MD + site model agree
        strong = ([1500, 500, 1580, 560], 0.60, 0)      # MD only but confident
        p = make_pipeline([bush, buffalo, strong])
        p.custom_detector = FakeCustom([([902, 502, 979, 561], 0.7, "animal")])
        p._frame_index = 1                              # next frame index 2 -> MD runs
        out = p.process_frame(self.frame, pose=(0.5, -1, 0.1), deep=True)
        xs = sorted(int(d["box"][0]) for d in out)
        self.assertEqual(len(out), 2, out)
        self.assertNotIn(100, xs)

    def test_site_model_bird_label_becomes_trusted_species(self):
        p = make_pipeline([])
        p.custom_detector = FakeCustom([([900, 500, 930, 520], 0.6, "bird")])
        out = p.process_frame(self.frame, pose=(0.5, -1, 0.1), deep=True)
        self.assertEqual(out[0]["species"], "bird")
        self.assertTrue(out[0]["species_trusted"])


class ParkLabelTests(unittest.TestCase):
    def test_far_buffalo_called_domestic_horse_is_shown_as_buffalo_guess(self):
        """19/10 17:01: a far buffalo was logged as 'domestic horse?'."""
        p = make_pipeline([([800, 500, 900, 580], 0.45, 0)])
        p.backend.classify = lambda crop: ("domestic horse", 0.57)
        with patch("config.ENHANCED_DETECTION_ENABLED", False):
            out = p.process_frame(np.full((H, W, 3), 120, np.uint8), scan_mode=True, pose=(0.51, -1, 0.03))
        self.assertEqual(out[0]["species"], "african buffalo")


class TopFoliageTests(unittest.TestCase):
    def test_tree_crown_against_the_sky_is_rejected(self):
        """19/10 16:33: box [994,1,1357,343] on a tree crown was followed by the tracker."""
        from detection.validator import prefilter
        self.assertEqual(prefilter({"box": [994.6, 1.6, 1357.4, 343.9]}, W, H), (False, "top_foliage"))
        self.assertTrue(prefilter({"box": [994.6, 400, 1357.4, 700]}, W, H)[0])     # on the ground
        self.assertTrue(prefilter({"box": [900, 1.0, 960, 40]}, W, H)[0])            # small far animal


class DoubtfulLabelTests(unittest.TestCase):
    """18/10: "domestic horse 0.69" on a 22x19 px acacia at the horizon was confirmed."""

    def _run(self, box, label, pose):
        p = make_pipeline([(box, 0.45, 0)])
        p.backend.classify = lambda crop: (label, 0.69)
        frame = np.full((H, W, 3), 120, np.uint8)
        with patch("config.ENHANCED_DETECTION_ENABLED", False):
            return p.process_frame(frame, scan_mode=True, pose=pose)

    def test_tiny_far_box_label_does_not_verify(self):
        out = self._run([1360, 300, 1382, 319], "plains zebra", (0.62, -1, 0.03))
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]["species_trusted"])
        t = MultiObjectTracker()
        for _ in range(2):
            tracks = t.update([dict(out[0], site_model_active=True)], W, H)
        self.assertFalse(tracks[0]["track_confirmed"])

    def test_livestock_label_does_not_verify(self):
        out = self._run([800, 500, 900, 580], "domestic horse", (0.62, -1, 0.03))
        self.assertEqual(out[0]["species"], "african buffalo")       # no livestock in the park

    def test_normal_animal_still_verified(self):
        out = self._run([800, 500, 900, 580], "plains zebra", (0.62, -1, 0.03))
        self.assertTrue(out[0]["species_trusted"])
        out = self._run([1360, 300, 1382, 319], "plains zebra", (0.72, -1, 0.2193))   # small but zoomed in
        self.assertTrue(out[0]["species_trusted"])


class DetectorSizeTests(unittest.TestCase):
    """18/10: IMAGE_SIZE changes did nothing - every pass ran at 1280 px (~7 s)."""

    def test_zoomed_view_and_tracking_roi_use_smaller_inputs(self):
        p = make_pipeline([([500, 500, 560, 540], 0.70, 0)])
        frame = np.full((H, W, 3), 120, np.uint8)
        with patch("config.ENHANCED_DETECTION_ENABLED", False):
            p.process_frame(frame, scan_mode=True, pose=(0.72, -1, 0.2193))       # water-hole, 21.8x
            p.process_frame(frame, scan_mode=True, pose=(0.50, -1, 0.03))         # wide scan
            p.process_frame(frame, pose=(0.50, -1, 0.2), roi=(300, 300, 900, 800))  # tracking region
        self.assertEqual(p.detector.sizes, [config.MEGADETECTOR_ZOOMED_IMGSZ, 1280, config.ROI_DETECTOR_IMAGE_SIZE])
        self.assertEqual(p.detector.predictor.args.imgsz, 1280)                   # restored


class SiteModelConfirmationTests(unittest.TestCase):
    """17/10: branches, bushes, stumps, fallen logs must not become animals."""

    def _det(self, conf, source):
        return {"box": [500, 500, 560, 540], "class_id": 0, "det_confidence": conf,
                "species": "unidentified", "species_confidence": 0.2, "species_trusted": True,
                "source": source, "site_model_active": True}

    def test_stump_seen_only_by_megadetector_is_not_confirmed(self):
        t = MultiObjectTracker()
        out = []
        for _ in range(4):
            out = t.update([self._det(0.48, "megadetector")], W, H)
        self.assertFalse(out[0]["track_confirmed"])

    def test_megadetector_plus_named_species_twice_is_confirmed(self):
        """19/10: baboons are invisible to the site model; MD 0.47-0.69 + SpeciesNet named them."""
        t = MultiObjectTracker()
        det = dict(self._det(0.47, "megadetector"), species="sciuridae family", species_confidence=0.5)
        out = t.update([det], W, H)
        self.assertFalse(out[0]["track_confirmed"])
        out = t.update([det], W, H)
        self.assertTrue(out[0]["track_confirmed"])

    def test_megadetector_twice_without_a_species_is_not_enough(self):
        t = MultiObjectTracker()
        for _ in range(3):
            out = t.update([self._det(0.47, "megadetector")], W, H)
        self.assertFalse(out[0]["track_confirmed"])

    def test_object_seen_twice_by_site_model_is_confirmed(self):
        t = MultiObjectTracker()
        out = t.update([self._det(0.30, "custom_tile")], W, H)
        self.assertFalse(out[0]["track_confirmed"])
        out = t.update([self._det(0.32, "fused:megadetector+custom_yolo")], W, H)
        self.assertTrue(out[0]["track_confirmed"])

    def test_site_model_bird_survives_speciesnet_blank(self):
        p = make_pipeline([], table={})
        p.backend = FakeBackend({})
        p.backend.trusted = True
        p.backend.classify = lambda crop: ("blank", 0.97)
        p.custom_detector = FakeCustom([([900, 500, 930, 520], 0.6, "bird")])
        out = p.process_frame(np.full((H, W, 3), 120, np.uint8), pose=(0.5, -1, 0.1), deep=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["species"], "bird")

    def test_bush_with_speciesnet_blank_is_still_dropped(self):
        p = make_pipeline([([100, 600, 180, 660], 0.62, 0)])
        p.backend.trusted = True
        p.backend.classify = lambda crop: ("blank", 0.97)
        p.custom_detector = FakeCustom([])
        p._frame_index = 1
        out = p.process_frame(np.full((H, W, 3), 120, np.uint8), pose=(0.5, -1, 0.1), deep=True)
        self.assertEqual(out, [])


class WaterZoneTests(unittest.TestCase):
    def test_zone_only_active_at_its_own_pose(self):
        path = os.path.join(tempfile.mkdtemp(), "w.json")
        import json
        json.dump({"zones": [{"name": "w", "pose": {"pan": 0.729, "tilt": -1.0, "zoom": 0.32},
                              "polygon": [[0.2, 0.5], [0.8, 0.5], [0.8, 0.8], [0.2, 0.8]]}]}, open(path, "w"))
        z = WaterZone(path)
        box = [800, 600, 900, 700]
        self.assertTrue(z.in_zone(box, W, H, (0.729, -1.0, 0.32)))
        self.assertFalse(z.in_zone(box, W, H, (0.5, -1.0, 0.32)))
        self.assertFalse(z.in_zone([100, 100, 160, 160], W, H, (0.729, -1.0, 0.32)))
        self.assertIsNotNone(z.roi_rect(W, H, (0.729, -1.0, 0.32)))
        self.assertIsNone(z.roi_rect(W, H, (0.4, -1.0, 0.32)))


if __name__ == "__main__":
    unittest.main()
