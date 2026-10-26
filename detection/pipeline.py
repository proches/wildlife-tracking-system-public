"""Wildlife detection + species classification pipeline (v3).

Stages
------
1. MegaDetector (general animal detector), optionally fused with a site-specific
   YOLO model trained on this camera's view.
2. Time-boxed native-resolution tiles (water-hole first) for small / distant
   animals and birds - only when the camera is settled and the AI thread has time.
3. Geometry pre-filter + learned static-clutter memory (``detection.validator``)
   BEFORE classification, so clutter costs no classifier time.
4. SpeciesNet (or AI4G fallback) naming layer with a non-wildlife veto.

All heavy libraries (torch, PytorchWildlife, speciesnet, ultralytics) are imported
lazily so the rest of the system - and the unit tests - import without them.
"""

import logging
import os
import time

import cv2
import numpy as np

import config
from detection.constants import (  # noqa: F401  (re-exported for older imports)
    CLASS_ANIMAL, CLASS_NAMES, CLASS_PERSON, CLASS_VEHICLE, DOMESTIC_LABELS, NON_ANIMAL_LABELS,
)
from detection.frame_quality import FrameQuality
from detection.validator import StaticFalsePositiveMemory, is_verified, prefilter
from detection.water_zone import WaterZone

logger = logging.getLogger("wildlife_tracker.detection")


def _display_name(raw_label):
    parts = [p.strip() for p in str(raw_label).split(";")]
    for part in reversed(parts):
        if part:
            return part
    return str(raw_label)


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _AI4GBackend:
    """Closed-set Snapshot-Serengeti classifier.

    It has no 'empty' / 'not an animal' answer, so it names EVERY crop - tree
    trunks and the camera's own date stamp came out as "zebra 0.83".  Its labels
    are shown (with a '?') but never count as verification (``trusted=False``).
    """
    name = "ai4g"
    trusted = False
    cache_seconds = 0.0
    max_new_per_frame = None

    def __init__(self, device):
        from PytorchWildlife.models import classification as pw_classification
        self.model = pw_classification.AI4GSnapshotSerengeti(device=device)
        self.threshold = config.CLASSIFICATION_CONFIDENCE_THRESHOLD

    def classify(self, crop_rgb):
        result = self.model.single_image_classification(crop_rgb)
        return result["prediction"], float(result["confidence"])


class _SpeciesNetBackend:
    name = "speciesnet"
    trusted = True     # open-set: answers 'blank' / 'animal' when unsure

    def __init__(self, device):
        import PIL.Image
        from speciesnet.classifier import SpeciesNetClassifier
        from detection.speciesnet_geofence import SpeciesNetGeofence

        self._Image = PIL.Image
        self.classifier = SpeciesNetClassifier(config.SPECIESNET_MODEL, device=device)
        geofence_parent = getattr(getattr(self.classifier, "model_info", None), "geofence", None)
        geofence_parent = getattr(geofence_parent, "parent", None)
        self.geofence = SpeciesNetGeofence(str(geofence_parent), config.SPECIESNET_COUNTRY) if geofence_parent else None
        self.threshold = config.SPECIESNET_CONFIDENCE_THRESHOLD
        self.cache_seconds = config.SPECIESNET_CACHE_SECONDS
        self.max_new_per_frame = config.SPECIESNET_MAX_NEW_PER_FRAME

    def classify(self, crop_rgb):
        if crop_rgb.size == 0:
            return "unidentified", 0.0
        h, w = crop_rgb.shape[:2]
        if min(h, w) < 96:
            scale = min(4.0, 96.0 / max(1, min(h, w)))
            crop_rgb = cv2.resize(crop_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img = self._Image.fromarray(np.ascontiguousarray(crop_rgb))
        preprocessed = self.classifier.preprocess(img, bboxes=None)
        result = self.classifier.predict("live_frame", preprocessed)
        classifications = result.get("classifications") or []
        if not classifications:
            return "unidentified", 0.0
        if self.geofence is not None:
            label, score = self.geofence.apply(classifications)
        else:
            label, score = classifications[0].get("label", "unidentified"), classifications[0].get("score", 0.0)
        return _display_name(label), float(score)


class WildlifePipeline:
    def __init__(self, device=config.DEVICE, water_zone=None, static_memory=None):
        try:
            import torch
            torch.set_num_threads(max(1, int(config.AI_TORCH_THREADS)))
        except Exception:
            pass
        self.detector = None
        self.custom_detector = None
        if config.CUSTOM_YOLO_ENABLED:
            try:
                from detection.custom_yolo import CustomYOLODetector
                self.custom_detector = CustomYOLODetector(device=device)
            except Exception as exc:
                logger.warning("Custom YOLO unavailable (%s: %s); continuing with MegaDetector", type(exc).__name__, exc)
        # The species classifier is loaded FIRST, in this thread: SpeciesNet and
        # PytorchWildlife both import yolov5, and importing it from two threads at
        # once failed ("partially initialized module ... circular import") -
        # randomly leaving the system without MegaDetector or without SpeciesNet.
        self.backend = self._load_backend(device)
        if self.custom_detector is not None and config.MEGADETECTOR_BACKGROUND_LOAD:
            # The site model is the primary detector: start working at once and let
            # MegaDetector (23 s to import on this PC) join as a second opinion later.
            import threading
            threading.Thread(target=self._load_megadetector, args=(device,), name="md-loader", daemon=True).start()
        else:
            self._load_megadetector(device)
        self.quality = FrameQuality()
        self.water = water_zone if water_zone is not None else WaterZone()
        self.static = static_memory if static_memory is not None else StaticFalsePositiveMemory()
        self._cache = []
        self._warned_classify = False
        self._frame_index = 0
        self._tile_cursor = 0
        self._custom_tile_cursor = 0
        self.last_metrics = {}

    def _load_megadetector(self, device):
        """MegaDetectorV6 from the project's own copy of the weights.

        PytorchWildlife 1.3 looks for 'MDV6b-yolov9-c.pt' but saves the download as
        'MDV6-yolov9-c.pt', so it re-downloaded 51 MB on EVERY start (and could not
        start at all without internet).  Passing the local file avoids that.
        """
        for attempt in (1, 2):
            try:
                from PytorchWildlife.models import detection as pw_detection
                weights = config.MEGADETECTOR_WEIGHTS if os.path.exists(config.MEGADETECTOR_WEIGHTS) else None
                logger.info("Loading MegaDetectorV6 (%s)...", weights or "download")
                md = pw_detection.MegaDetectorV6(device=device, version="MDV6-yolov9-c", weights=weights)
                # Ultralytics printed "0: 1280x1280 15 animals" for every RAW pass - boxes
                # before any filtering, which looked like real detections in the log.
                try:
                    md.predictor.args.verbose = False
                except AttributeError:
                    pass
                self.detector = md
                logger.info("MegaDetector ready (input size %s)", getattr(md, "IMAGE_SIZE", "n/a"))
                return
            except Exception as exc:
                if self.custom_detector is None and attempt == 2:
                    raise
                if attempt == 1 and isinstance(exc, ImportError):
                    logger.info("MegaDetector import hiccup (%s); retrying", exc)
                    time.sleep(3.0)
                    continue
                if self.custom_detector is None:
                    raise
                logger.warning("MegaDetector unavailable (%s: %s); running on the site model only",
                               type(exc).__name__, exc)
                self._md_failed = True
                return

    def _load_backend(self, device):
        if config.CLASSIFIER_BACKEND == "speciesnet":
            try:
                backend = _SpeciesNetBackend(device)
                logger.info("Species classifier: Google SpeciesNet")
                return backend
            except Exception as exc:
                logger.warning(
                    "SpeciesNet unavailable (%s: %s); falling back to AI4G Snapshot Serengeti. "
                    "AI4G knows only ~50 camera-trap classes and almost no birds. "
                    "SpeciesNet needs Python 3.9-3.13 (3.11 recommended) - recreate the venv with it.",
                    type(exc).__name__, exc)
        return _AI4GBackend(device)

    # ------------------------------------------------------------ detection
    def _detect(self, frame_rgb, threshold, imgsz=None):
        if self.detector is None:          # MegaDetector still loading in the background
            return []
        # PytorchWildlife reads the inference size from predictor.args.imgsz (its
        # IMAGE_SIZE attribute is only used at construction - changing it did
        # nothing, so every "640" ROI pass really ran at 1280 and took ~7 s).
        args = getattr(getattr(self.detector, "predictor", None), "args", None)
        old = None
        if imgsz is not None and args is not None:
            old = args.imgsz
            args.imgsz = int(imgsz)
        try:
            result = self.detector.single_image_detection(frame_rgb, det_conf_thres=threshold)
        finally:
            if old is not None:
                args.imgsz = old
        detections = result["detections"]
        out = []
        for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            out.append({
                "box": [float(v) for v in xyxy],
                "class_id": int(class_id),
                "det_confidence": float(confidence),
                "species": None,
                "species_confidence": None,
                "source": "megadetector",
            })
        return out

    def _detect_region(self, frame_rgb, rect, threshold, source, imgsz=None):
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in rect]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return []
        crop = frame_rgb[y1:y2, x1:x2]
        items = self._detect(crop, threshold, imgsz=imgsz)
        for it in items:
            bx1, by1, bx2, by2 = it["box"]
            it["box"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
            it["source"] = source
        return items

    def _tile_rects(self, w, h):
        cols, rows = max(1, int(config.ENHANCED_TILE_COLS)), max(1, int(config.ENHANCED_TILE_ROWS))
        ov = min(0.45, max(0.0, float(config.ENHANCED_TILE_OVERLAP)))
        tw, th = int(w / cols * (1 + ov)), int(h / rows * (1 + ov))
        tw, th = min(w, tw), min(h, th)
        xs = [0] if cols == 1 else [int(round(i * (w - tw) / (cols - 1))) for i in range(cols)]
        ys = [0] if rows == 1 else [int(round(i * (h - th) / (rows - 1))) for i in range(rows)]
        return [(x, y, x + tw, y + th) for y in ys for x in xs]

    def _enhanced_detect(self, frame_rgb, water_rect=None):
        """Native-resolution tiles, water-hole first, stopped by a time budget."""
        h, w = frame_rgb.shape[:2]
        deadline = time.perf_counter() + float(config.ENHANCED_TIME_BUDGET_SECONDS)
        results = []
        if water_rect is not None and config.WATER_ZONE_ROI_PASS:
            results += self._detect_region(frame_rgb, water_rect, config.WATER_ZONE_ROI_CONFIDENCE,
                                           "megadetector_water_roi")
        tiles = self._tile_rects(w, h)
        # Tiles overlapping the water rect first, then round-robin from a moving cursor
        # so every tile is eventually covered even on a slow CPU.
        def overlaps(t):
            if water_rect is None:
                return False
            return not (t[2] <= water_rect[0] or t[0] >= water_rect[2] or t[3] <= water_rect[1] or t[1] >= water_rect[3])
        first = [t for t in tiles if overlaps(t)]
        rest = [t for t in tiles if not overlaps(t)]
        if rest:
            k = self._tile_cursor % len(rest)
            rest = rest[k:] + rest[:k]
        done = 0
        for tile in first + rest:
            # v3.1: the budget used to apply only once something was found, so an
            # empty scene ran ALL tiles (~6 x 5.5 s = 40 s per frame on this CPU)
            # and the patrol stalled waiting for results.  One tile always runs;
            # the round-robin cursor covers the rest on the next frames.
            if done and time.perf_counter() > deadline:
                break
            results += self._detect_region(frame_rgb, tile, config.ENHANCED_DETECTION_CONFIDENCE_THRESHOLD,
                                           "megadetector_tile")
            done += 1
        if rest:
            self._tile_cursor += max(1, done - len(first))
        return self._class_aware_nms(results, config.ENHANCED_NMS_IOU)

    def _custom_region(self, frame_bgr, rect, imgsz, conf, source):
        x1, y1, x2, y2 = [int(v) for v in rect]
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return []
        items = self.custom_detector.detect(frame_bgr[y1:y2, x1:x2], imgsz=imgsz, conf=conf)
        for it in items:
            bx1, by1, bx2, by2 = it["box"]
            it["box"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
            it["source"] = source
        return items

    def _custom_magnified(self, frame_bgr, pose, deep):
        """Extra site-model passes on MAGNIFIED crops - far animals and small birds.

        * water-hole: the water polygon (+ bank) every frame, ~2x magnified;
        * elsewhere, while scanning: CUSTOM_TILES_PER_FRAME native tiles per
          frame (round robin), ~1.5x magnified - the whole view every few frames.
        The site model is fast (~0.4-1 s per pass on this CPU) and did not put a
        single box on this camera's vegetation in validation.
        """
        h, w = frame_bgr.shape[:2]
        out = []
        water_rect = self.water.roi_rect(w, h, pose, pad=0.04) if self.water.enabled else None
        if water_rect is not None and config.WATER_MAGNIFY_ENABLED:
            x1, y1, x2, y2 = water_rect
            if (x2 - x1) * (y2 - y1) > 0.5 * w * h:
                # close zoom: the water fills the frame - two halves, ~2x magnified
                mid = (x1 + x2) // 2
                halves = [(x1, y1, mid + 40, y2), (mid - 40, y1, x2, y2)]
            else:
                halves = [water_rect]
            for rect in halves:
                out += self._custom_region(frame_bgr, rect, config.WATER_MAGNIFY_IMGSZ,
                                           config.WATER_MAGNIFY_CONFIDENCE, "custom_water_magnified")
        elif deep and config.CUSTOM_TILES_PER_FRAME > 0:
            tiles = self._tile_rects(w, h)
            for k in range(config.CUSTOM_TILES_PER_FRAME):
                tile = tiles[(getattr(self, '_custom_tile_cursor', 0) + k) % len(tiles)]
                out += self._custom_region(frame_bgr, tile, config.CUSTOM_TILE_IMGSZ,
                                           config.CUSTOM_YOLO_CONFIDENCE, "custom_tile")
            self._custom_tile_cursor = (getattr(self, '_custom_tile_cursor', 0) + config.CUSTOM_TILES_PER_FRAME) % len(tiles)
        return out

    @staticmethod
    def _class_aware_nms(items, iou_threshold):
        items = sorted(items, key=lambda d: d["det_confidence"], reverse=True)
        kept = []
        for item in items:
            if not any(item["class_id"] == o["class_id"] and _iou(item["box"], o["box"]) >= iou_threshold for o in kept):
                kept.append(item)
        return kept

    def _fuse(self, base, custom):
        if not custom:
            return base
        fused = list(base)
        weight = float(config.CUSTOM_YOLO_FUSION_WEIGHT)
        for item in custom:
            best, best_iou = None, 0.0
            for existing in fused:
                if existing["class_id"] == item["class_id"]:
                    overlap = _iou(existing["box"], item["box"])
                    if overlap > best_iou:
                        best_iou, best = overlap, existing
            if item.get("source_label"):
                item.setdefault("detector_label", item["source_label"])
                item.setdefault("detector_label_confidence", item["det_confidence"])
            if best is not None and best_iou >= config.ENHANCED_NMS_IOU:
                best["det_confidence"] = max(
                    best["det_confidence"],
                    (1.0 - weight) * best["det_confidence"] + weight * item["det_confidence"],
                )
                best["source"] = "fused:megadetector+custom_yolo"
                if item.get("detector_label"):
                    best["detector_label"] = item["detector_label"]
                    best["detector_label_confidence"] = item["detector_label_confidence"]
            else:
                fused.append(item)
        return self._class_aware_nms(fused, config.ENHANCED_NMS_IOU)

    # -------------------------------------------------------------- main API
    def process_frame(self, frame_bgr, scan_mode=False, pose=None, roi=None, deep=None, settled=True):
        """Detect + classify one frame.

        ``pose``     (pan, tilt, zoom) at capture time; enables water-zone and static-clutter logic.
        ``roi``      pixel rect: fast tracking pass restricted to that region.
        ``deep``     run the tiled / water-ROI passes (default: only in scan mode).
        ``settled``  the camera was still and focused - only then may the static-clutter
                     memory learn from this frame.
        """
        started = time.perf_counter()
        quality = self.quality.analyze(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame_rgb.shape[:2]
        self._frame_index += 1
        threshold = (config.LOW_LIGHT_DETECTION_CONFIDENCE_THRESHOLD if quality["low_light"]
                     else config.DETECTION_CONFIDENCE_THRESHOLD)
        deep = bool(scan_mode) if deep is None else bool(deep)
        stage = {"md": 0.0, "site": 0.0, "magnify": 0.0, "classify": 0.0, "classified": 0}
        md_ran = False

        use_custom = self.custom_detector is not None and self.custom_detector.enabled
        # Zoomed in (water-hole, a tracked animal): animals are big, so both
        # detectors run at a smaller input size - measured 3.9 s instead of 6.7 s
        # for MegaDetector, with HIGHER scores on the zebras at the water.
        zoomed = pose is not None and float(pose[2]) >= config.ZOOMED_VIEW_MIN_ZOOM
        if roi is not None and (self.detector is not None or use_custom):
            # Tracking a locked animal: only the region around it (fast).
            t = time.perf_counter()
            md_ran = self.detector is not None
            base = (self._detect_region(frame_rgb, roi, threshold, "megadetector_roi",
                                        imgsz=config.ROI_DETECTOR_IMAGE_SIZE)
                    if self.detector is not None else [])
            stage["md"] = time.perf_counter() - t
            if use_custom:
                t = time.perf_counter()
                base = self._fuse(base, self._custom_region(frame_bgr, roi, config.ROI_SITE_IMAGE_SIZE,
                                                            config.CUSTOM_YOLO_CONFIDENCE, "custom_roi"))
                stage["site"] = time.perf_counter() - t
        else:
            # With the site-trained model (no false boxes on this camera's
            # vegetation) it is the primary detector; MegaDetector is a second
            # opinion every N frames.
            every = config.MEGADETECTOR_EVERY_N_ZOOMED if zoomed else config.MEGADETECTOR_EVERY_N_WITH_CUSTOM
            if not scan_mode:          # following a locked animal: the strong detector more often
                every = min(every, config.MEGADETECTOR_EVERY_N_TRACKING)
            if pose is not None and float(pose[1]) > -0.98:
                # looking DOWN at the near ground (low patrol row, hand-aimed): baboons
                # rest there, and only MegaDetector sees them (site model scored 0)
                every = min(every, config.MEGADETECTOR_EVERY_N_LOW_VIEW)
            run_md = (not use_custom) or self._frame_index % max(1, every) == 0
            t = time.perf_counter()
            md_ran = run_md and self.detector is not None
            base = (self._detect(frame_rgb, threshold, imgsz=config.MEGADETECTOR_ZOOMED_IMGSZ if zoomed else None)
                    if run_md else [])
            stage["md"] = time.perf_counter() - t
            if use_custom:
                t = time.perf_counter()
                custom = self.custom_detector.detect(frame_bgr, imgsz=config.CUSTOM_YOLO_ZOOMED_IMGSZ if zoomed else None)
                stage["site"] = time.perf_counter() - t
                t = time.perf_counter()
                custom += self._custom_magnified(frame_bgr, pose, deep)
                stage["magnify"] = time.perf_counter() - t
                base = self._fuse(base, custom)
                # Consensus: a box only MegaDetector sees must be clearly confident;
                # weak MD-only boxes were bushes/branches - drop them before the
                # (1.5 s per crop) species classifier.
                base = [d for d in base if d.get("source") != "megadetector"
                        or d["det_confidence"] >= config.MD_ONLY_MIN_CONFIDENCE]
            tiles_needed = not (use_custom and config.SKIP_TILES_WITH_CUSTOM)
            if (tiles_needed and deep and config.ENHANCED_DETECTION_ENABLED
                    and config.ENHANCED_DETECTION_MODE != "off"):
                water_rect = self.water.roi_rect(frame_w, frame_h, pose) if self.water.enabled else None
                base = self._fuse(base, self._enhanced_detect(frame_rgb, water_rect))

        # ---- geometry pre-filter + known clutter (before the classifier) ----
        rejected = {}
        animals = []
        for d in base:
            if d["class_id"] != CLASS_ANIMAL:
                continue
            ok, why = prefilter(d, frame_w, frame_h)
            if not ok:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            if self.static.is_known_static(d, pose, frame_w, frame_h):
                rejected["static_clutter"] = rejected.get("static_clutter", 0) + 1
                continue
            animals.append(d)
        animals.sort(key=lambda d: d["det_confidence"], reverse=True)

        results = []
        now = time.monotonic()
        self._expire_cache(now)
        diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        budget = self.backend.max_new_per_frame
        dropped_non_wildlife = 0

        for entry in animals:
            cached = self._lookup_cache(entry["box"], diag)
            if cached is not None:
                species, species_conf = cached
            elif budget is None or budget > 0:
                crop = self._context_crop(frame_rgb, entry["box"], context=0.15)
                t = time.perf_counter()
                species, species_conf = self._classify(crop)
                stage["classify"] += time.perf_counter() - t
                stage["classified"] += 1
                self._store_cache(entry["box"], species, species_conf, now)
                if budget is not None:
                    budget -= 1
            else:
                species, species_conf = "unidentified", 0.0
            entry["species"] = species
            entry["species_confidence"] = species_conf
            entry["species_trusted"] = bool(getattr(self.backend, "trusted", True))
            if entry.get("detector_label") == "bird" and (
                    not entry["species_trusted"] or species in ("unidentified", "", None)):
                # The site-trained detector saw a bird: better than a closed-set guess.
                entry["species"] = "bird"
                entry["species_confidence"] = float(entry.get("detector_label_confidence", 0.0))
                entry["species_trusted"] = True

            entry["site_model_active"] = bool(self.custom_detector is not None and self.custom_detector.enabled)
            label_l = (species or "").strip().lower()
            # A tiny far bird often looks "blank" to SpeciesNet; when the site model
            # itself is confident it is a bird, the blank veto does not apply.
            site_bird = (entry.get("detector_label") == "bird"
                         and float(entry.get("detector_label_confidence", 0.0)) >= config.SITE_BIRD_VETO_EXEMPT)
            if site_bird and label_l in ("blank", "unidentified", ""):
                entry["species"], entry["species_confidence"] = "bird", float(entry["detector_label_confidence"])
                entry["species_trusted"] = True
                label_l = "bird"
            park = getattr(config, "PARK_LABEL_MAP", {}).get(label_l)
            if park:
                # inside the national park "domestic horse/cattle" = what SpeciesNet makes of a
                # small far buffalo (19/10); shown with '?' - a guess, never verification
                entry["species"], label_l = park, park
                # a renamed guess: shown with '?' only when the box is tiny (doubtful below)
            if entry["species_trusted"] and self._doubtful_label(entry, label_l, pose, frame_w, frame_h):
                # the label alone proves nothing: the object must earn confirmation
                # through repeated site-model hits (multi_object_tracker)
                entry["species_trusted"] = False
            if (getattr(config, "DROP_NON_WILDLIFE_SPECIES", True) and label_l in NON_ANIMAL_LABELS
                    and float(species_conf or 0.0) >= config.NON_ANIMAL_VETO_CONFIDENCE):
                dropped_non_wildlife += 1
                continue

            entry["in_water_zone"] = bool(self.water.enabled and self.water.in_zone(entry["box"], frame_w, frame_h, pose))
            if settled and roi is None and self.static.observe(entry, pose, frame_w, frame_h):
                rejected["static_clutter"] = rejected.get("static_clutter", 0) + 1
                continue
            results.append(entry)
        self.static.save()

        elapsed = time.perf_counter() - started
        self.last_metrics = {
            "inference_seconds": elapsed,
            # False: only the site model looked.  It does not see every species
            # (baboons!), so an empty result then is NOT evidence the animal left.
            "conclusive": bool(md_ran or self.detector is None and not self._md_expected()),
            "stages": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stage.items()},
            "frame_quality": quality,
            "base_detections": len(base),
            "animal_detections": len(results),
            "rejected": rejected,
            "dropped_non_wildlife": dropped_non_wildlife,
            "deep": deep,
            "roi": roi is not None,
            "scan_mode": bool(scan_mode),
            "custom_detector": bool(self.custom_detector and self.custom_detector.enabled),
            "classifier": self.backend.name,
        }
        return results

    def _md_expected(self):
        """MegaDetector is still loading in the background (not failed)."""
        return (self.custom_detector is not None and config.MEGADETECTOR_BACKGROUND_LOAD
                and not getattr(self, "_md_failed", False))

    @staticmethod
    def _doubtful_label(entry, label_l, pose, frame_w, frame_h):
        """True when a classifier label must not count as verification.

        * livestock labels (DOMESTIC_LABELS) - not expected in the park;
        * tiny boxes in a wide view: a 20-40 px object at the horizon is as
          likely an acacia as an animal, and SpeciesNet guesses on it.
        """
        if label_l in DOMESTIC_LABELS:
            return True
        zoom = float(pose[2]) if pose is not None else 0.0
        x1, y1, x2, y2 = entry["box"]
        return zoom < config.ZOOMED_VIEW_MIN_ZOOM and max(x2 - x1, y2 - y1) < config.FAR_TINY_BOX_PX

    @staticmethod
    def _context_crop(frame_rgb, box, context=0.15):
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * context))
        y1 = max(0, int(y1 - bh * context))
        x2 = min(w, int(x2 + bw * context))
        y2 = min(h, int(y2 + bh * context))
        return frame_rgb[y1:y2, x1:x2]

    def _classify(self, crop):
        if crop.size == 0:
            return "unidentified", 0.0
        try:
            label, confidence = self.backend.classify(crop)
        except Exception as exc:
            if not self._warned_classify:
                logger.warning("Species classification failed (%s: %s)", type(exc).__name__, exc)
                self._warned_classify = True
            return "unidentified", 0.0
        label = _display_name(label)
        if label.lower() in NON_ANIMAL_LABELS:
            if confidence >= config.NON_ANIMAL_VETO_CONFIDENCE:
                return label, confidence
            return "unidentified", confidence
        if confidence < self.backend.threshold:
            return "unidentified", confidence
        return label, confidence

    def invalidate_classification_cache(self):
        self._cache.clear()

    def _expire_cache(self, now):
        ttl = self.backend.cache_seconds
        self._cache = [c for c in self._cache if ttl > 0 and now - c["t"] <= ttl]

    def _lookup_cache(self, box, diag):
        if self.backend.cache_seconds <= 0:
            return None
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        max_dist = min(0.11 * diag, max(50.0, min(150.0, 0.25 * ((box[2] - box[0]) + (box[3] - box[1])))))
        best, best_dist = None, None
        for c in self._cache:
            dist = ((cx - c["cx"]) ** 2 + (cy - c["cy"]) ** 2) ** 0.5
            if dist <= max_dist and (best_dist is None or dist < best_dist):
                best, best_dist = c, dist
        if best is not None:
            best["cx"], best["cy"], best["t"] = cx, cy, time.monotonic()
            return best["species"], best["conf"]
        return None

    def _store_cache(self, box, species, conf, now):
        if self.backend.cache_seconds <= 0:
            return
        self._cache.append({
            "cx": (box[0] + box[2]) / 2, "cy": (box[1] + box[3]) / 2,
            "species": species, "conf": conf, "t": now,
        })
        self._cache = self._cache[-32:]
