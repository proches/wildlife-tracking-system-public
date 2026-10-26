"""False-positive suppression for wildlife detections.

Everything here is cheap, deterministic and runs *before* the expensive species
classifier, so rejected boxes also save CPU.  Three layers:

1. ``prefilter`` - geometry: camera on-screen-display text, near-field objects
   (poles, tanks, staff), 1-px slivers on the frame edge, absurd sizes/aspects.
2. ``is_verified`` - does the species classifier actually believe this is a
   real animal/bird?  (used to exempt real animals from layer 3)
3. ``StaticFalsePositiveMemory`` - a stump / bush / rock that shows up at the
   same place, from the same camera pose, over and over, never moves and is
   never identified, is learned (and persisted) as static clutter.
"""

import json
import logging
import os
import time

import config
from detection.constants import NON_ANIMAL_LABELS, UNVERIFIED_LABELS

logger = logging.getLogger("wildlife_tracker.validator")


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def is_verified(det):
    """Real species with a believable score (not 'unidentified', 'other', 'blank'...).

    Labels from a closed-set classifier (AI4G: ``species_trusted=False``) never
    verify anything - it calls tree trunks and on-screen text "zebra".
    """
    if not det.get("species_trusted", True):
        return False
    label = str(det.get("species") or "").strip().lower()
    if label in UNVERIFIED_LABELS or label in NON_ANIMAL_LABELS:
        return False
    return float(det.get("species_confidence") or 0.0) >= config.VERIFIED_SPECIES_CONFIDENCE


def _animal_evidence(det):
    label = str(det.get("species") or "").strip().lower()
    conf = float(det.get("species_confidence") or 0.0)
    if label and label not in UNVERIFIED_LABELS and label not in NON_ANIMAL_LABELS \
            and conf >= config.STATIC_FP_ANIMAL_LABEL_MIN:
        return True
    # (the site model alone is NOT evidence: it boxed a tree crown at 0.57, 19/10)
    return False


def species_label(det):
    """Species name for logs / overlay; an unverifiable guess gets a '?'."""
    label = det.get("species") or "unidentified"
    if not det.get("species_trusted", True) and str(label).lower() not in UNVERIFIED_LABELS:
        return f"{label}?"
    return label


def prefilter(det, frame_w, frame_h):
    """Return (ok, reason).  Pure geometry on one detection."""
    x1, y1, x2, y2 = det["box"]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False, "empty"
    if w < config.MIN_BOX_PIXELS or h < config.MIN_BOX_PIXELS:
        return False, "tiny"
    if (w * h) / float(frame_w * frame_h) > config.MAX_BOX_AREA_RATIO:
        return False, "too_large"
    aspect = w / h
    if aspect < config.MIN_BOX_ASPECT or aspect > config.MAX_BOX_ASPECT:
        return False, "aspect"
    # Slivers hugging the frame edge: a 17-px-wide "animal" at x=1903..1920 was
    # logged for hours.  Real animals entering the frame are wider than this.
    sliver = config.EDGE_SLIVER_MAX_PX
    touches_x = x1 <= 1.5 or x2 >= frame_w - 1.5
    touches_y = y1 <= 1.5 or y2 >= frame_h - 1.5
    if (touches_x and w < sliver) or (touches_y and h < sliver):
        return False, "edge_sliver"
    # 19/10: a tree crown against the sky (box glued to the top edge, 360x340 px)
    # was "confirmed" by the site model and the tracker turned the camera to it.
    # A big box that touches the top edge and stays in the upper part of the
    # picture is foliage / sky, not an animal standing on the ground.
    if (y1 <= config.TOP_EDGE_PX and h >= config.TOP_FOLIAGE_MIN_HEIGHT * frame_h
            and y2 <= config.TOP_FOLIAGE_MAX_BOTTOM * frame_h):
        return False, "top_foliage"
    cx, cy = _center(det["box"])
    nx, ny = cx / frame_w, cy / frame_h
    for (mx1, my1, mx2, my2) in list(config.OSD_MASKS):
        if mx1 <= nx <= mx2 and my1 <= ny <= my2:
            return False, "osd_text"
    for (mx1, my1, mx2, my2) in list(config.NEAR_FIELD_MASKS):
        if mx1 <= nx <= mx2 and my1 <= ny <= my2:
            return False, "near_field"
    return True, "ok"


class StaticFalsePositiveMemory:
    """Learns immobile, unidentified detections per camera pose."""

    def __init__(self, path=None, clock=time.time):
        self.path = path or config.STATIC_FP_FILE
        self.clock = clock
        self.entries = []   # dict(pose, box(norm), first, last, hits, static, moved)
        self._dirty = False
        self._last_save = 0.0
        self.load()

    # -- persistence
    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = [e for e in data.get("entries", []) if e.get("static")]
            logger.info("Static false-positive memory: %d known clutter spots", len(self.entries))
        except (OSError, ValueError):
            self.entries = []

    def save(self, force=False):
        now = self.clock()
        if not self._dirty or (not force and now - self._last_save < 30.0):
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            keep = [e for e in self.entries if e.get("static")]
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"entries": keep}, f, indent=1)
            self._dirty = False
            self._last_save = now
        except OSError:
            pass

    # -- matching
    @staticmethod
    def _pose_close(a, b):
        tp, tt, tz = config.STATIC_FP_POSE_TOLERANCE
        return abs(a[0] - b[0]) <= tp and abs(a[1] - b[1]) <= tt and abs(a[2] - b[2]) <= tz

    def _find(self, pose, nbox):
        best, best_iou = None, 0.0
        for e in self.entries:
            if not self._pose_close(e["pose"], pose):
                continue
            iou = _iou(e["box"], nbox)
            ecx, ecy = _center(e["box"])
            ncx, ncy = _center(nbox)
            near = ((ecx - ncx) ** 2 + (ecy - ncy) ** 2) ** 0.5 <= config.STATIC_FP_MOVE_TOLERANCE
            if (iou >= 0.35 or near) and iou >= best_iou:
                best, best_iou = e, iou
        return best

    def is_known_static(self, det, pose, frame_w, frame_h):
        if not config.STATIC_FP_ENABLED or pose is None:
            return False
        nbox = [det["box"][0] / frame_w, det["box"][1] / frame_h, det["box"][2] / frame_w, det["box"][3] / frame_h]
        e = self._find(pose, nbox)
        return bool(e and e.get("static"))

    def observe(self, det, pose, frame_w, frame_h):
        """Record one *settled-camera* detection.  Returns True if it is static clutter."""
        if not config.STATIC_FP_ENABLED or pose is None:
            return False
        now = self.clock()
        nbox = [det["box"][0] / frame_w, det["box"][1] / frame_h, det["box"][2] / frame_w, det["box"][3] / frame_h]
        if is_verified(det) and float(det.get("det_confidence", 0.0)) >= 0.30:
            # A real, identified animal.  If it sits on a learned clutter spot, the spot is wrong.
            e = self._find(pose, nbox)
            if e is not None and e.get("static"):
                self.entries.remove(e)
                self._dirty = True
            return False
        if _animal_evidence(det):
            # 19/10: resting baboons ("sciuridae family 0.60") were learned as
            # clutter after 25 s of sitting still.  Anything a classifier calls an
            # animal, or the site model is fairly sure of, is never clutter.
            return False
        e = self._find(pose, nbox)
        if e is None:
            self.entries.append(dict(pose=list(pose), box=nbox, first=now, last=now, hits=1, static=False))
            self.entries = self.entries[-400:]
            return False
        if e.get("static"):
            e["last"] = now
            return True
        # still being evaluated: did it move?
        ecx, ecy = _center(e["box"])
        ncx, ncy = _center(nbox)
        dist = ((ecx - ncx) ** 2 + (ecy - ncy) ** 2) ** 0.5
        if dist > config.STATIC_FP_MOVE_TOLERANCE * 0.5:
            e["first"], e["hits"] = now, 1       # it moved: restart the clock, it may be an animal
            e["box"] = nbox
            e["last"] = now
            return False
        e["hits"] += 1
        e["last"] = now
        if e["hits"] >= config.STATIC_FP_MIN_HITS and now - e["first"] >= config.STATIC_FP_MIN_SECONDS:
            e["static"] = True
            self._dirty = True
            logger.info("Learned static clutter at pan=%.3f zoom=%.3f box=%s (never moved, never identified)",
                        pose[0], pose[2], [round(v, 3) for v in nbox])
            return True
        return False
