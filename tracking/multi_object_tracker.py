"""Lightweight persistent multi-object tracker for wildlife PTZ scenes.

It is intentionally dependency-free. It uses constant-velocity prediction,
IoU/center gating and temporal persistence. A heavier ReID tracker can be
plugged in later without changing the PTZ API.
"""

from dataclasses import dataclass
from typing import Optional

import config
from detection.constants import CLASS_ANIMAL
from detection.validator import is_verified


@dataclass
class Track:
    track_id: int
    box: list
    velocity: tuple = (0.0, 0.0)
    hits: int = 1
    age: int = 1
    misses: int = 0
    species: str = "unidentified"
    species_confidence: float = 0.0
    last_confidence: float = 0.0
    confirmed: bool = False
    candidate_species: str = "unidentified"
    candidate_species_hits: int = 0
    max_confidence: float = 0.0
    conf_sum: float = 0.0
    water_hits: int = 0
    verified: bool = False
    in_water: bool = False
    species_trusted: bool = True
    site_model: bool = False      # the site-trained detector was running
    custom_hits: int = 0          # frames in which the site model itself saw this object

    @property
    def mean_confidence(self):
        return self.conf_sum / max(1, self.hits)

    def evaluate_confirmation(self):
        """Multi-frame confirmation (v3): kills one-off glints, leaves real animals."""
        strong = self.max_confidence >= config.STRONG_DETECTION_CONFIDENCE
        # An identified animal with a decent box score is believed immediately.
        # (0.25: SpeciesNet said "plains zebra 0.86" on a zebra boxed at 0.28 by the
        # detector - a real animal that was never followed.)
        if self.verified and self.max_confidence >= config.VERIFIED_MIN_BOX_CONFIDENCE:
            return True
        if self.site_model and not self.verified:
            # Branches, bushes, stumps and fallen logs fooled the generic detectors.
            # The site model was trained on this exact view to ignore them, so an
            # unidentified object needs the site model to have seen it repeatedly...
            if self.custom_hits >= config.CONFIRM_SITE_MODEL_HITS and self.hits >= 2:
                return True
            # A clearly visible animal: a strong box twice (19/10: a bird at 0.78 on a branch,
            # SpeciesNet "unidentified", was zoomed in on and still dropped).  Bushes and
            # stumps are "blank" to SpeciesNet and were vetoed before reaching this point.
            if self.hits >= 2 and self.max_confidence >= config.CONFIRM_STRONG_BOX:
                return True
            # Two independent models agree: MegaDetector boxed it (>= 0.40) twice and
            # SpeciesNet names an animal (it calls bushes/stumps "blank", which are
            # vetoed before this).  19/10: baboons - invisible to the site model -
            # were never confirmed in a 12-25 s station watch.
            if (self.hits >= 2 and self.max_confidence >= config.CONFIRM_TWO_MODEL_MIN_BOX
                    and self.species_trusted and self.species not in ("unidentified", "other", "", "blank")
                    and self.species_confidence >= config.CONFIRM_TWO_MODEL_MIN_SPECIES):
                return True
            # ...or overwhelming, persistent evidence from the other detectors.
            return (self.hits >= config.TRACK_CONFIRM_HITS + 2 and strong
                    and self.mean_confidence >= config.CONFIRM_MEAN_CONFIDENCE)
        if self.hits < config.TRACK_CONFIRM_HITS:
            return False
        if strong or self.verified:
            return True
        if self.water_hits >= 1 and self.mean_confidence >= config.WATER_ZONE_MIN_CONFIDENCE:
            return True
        return self.mean_confidence >= config.CONFIRM_MEAN_CONFIDENCE

    def predict_center(self):
        cx, cy = _center(self.box)
        return cx + self.velocity[0], cy + self.velocity[1]


def camera_shift(prev_pose, pose, frame_width):
    """(dx, k): pixel shift and zoom scale of the picture between two camera poses.

    Pan: PAN_PIXELS_PER_UNIT_AT_1X x magnification pixels per pan unit (measured
    19/10 on a buffalo tracked during the sweep).  None for unusable poses.
    """
    try:
        dpan = float(pose[0]) - float(prev_pose[0])
        m0 = 1.0 + config.PTZ_ZOOM_MAG_PER_UNIT * max(0.0, float(prev_pose[2]))
        m1 = 1.0 + config.PTZ_ZOOM_MAG_PER_UNIT * max(0.0, float(pose[2]))
    except (TypeError, ValueError, IndexError):
        return None
    if abs(dpan) > 0.2:          # glitched pose read - never trust it
        return None
    dx = dpan * config.PAN_PIXELS_PER_UNIT_AT_1X * m1 * frame_width / 1920.0
    try:
        t0, t1 = float(prev_pose[1]), float(pose[1])
    except (TypeError, ValueError, IndexError):
        t0 = t1 = 0.0
    # tilt reads -1.0 over a whole range (clamped): only a change between two
    # unclamped reads is real.  Tilt up (reading falls) moves the picture down.
    dy = 0.0
    if t0 > -0.999 and t1 > -0.999 and abs(t1 - t0) < 0.3:
        dy = -(t1 - t0) * config.TILT_PIXELS_PER_UNIT_AT_1X * m1 * frame_width / 1920.0
    return dx, m1 / m0, dy


def shift_box(box, dx, k, frame_width, frame_height, dy=0.0):
    cx, cy = frame_width / 2.0, frame_height / 2.0
    x1, y1, x2, y2 = box
    return [cx + (x1 - cx) * k + dx, cy + (y1 - cy) * k + dy, cx + (x2 - cx) * k + dx, cy + (y2 - cy) * k + dy]


def _from_site_model(det):
    src = str(det.get("source", ""))
    return src.startswith("custom") or "custom_yolo" in src


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


class MultiObjectTracker:
    def __init__(self):
        self.tracks = {}
        self._next_id = 1
        self._last_pose = None

    def _compensate(self, pose, frame_width, frame_height):
        """Move the existing tracks with the CAMERA before associating.

        19/10: a buffalo found while sweeping moved ~400 px between two AI
        results only because the camera turned - it became a "new" animal every
        frame and a locked one was lost after the tracker re-centred it.
        Pan shift: PAN_PIXELS_PER_UNIT_AT_1X x magnification pixels per pan unit
        (measured on the tracked buffalo); zoom change: scale about the centre.
        """
        prev, self._last_pose = self._last_pose, pose
        if pose is None or prev is None or not self.tracks:
            return
        shift = camera_shift(prev, pose, frame_width)
        if shift is None:
            return
        dx, k, dy = shift
        if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(k - 1.0) < 1e-3:
            return
        for track in self.tracks.values():
            track.box = shift_box(track.box, dx, k, frame_width, frame_height, dy)

    def update(self, detections, frame_width, frame_height, pose=None):
        self._compensate(pose, frame_width, frame_height)
        animals = [d for d in detections if d.get("class_id") == CLASS_ANIMAL]
        diagonal = (frame_width ** 2 + frame_height ** 2) ** 0.5
        unmatched_tracks = set(self.tracks)
        unmatched_detections = set(range(len(animals)))
        pairs = []

        # Greedy association over the strongest compatibility scores.
        candidates = []
        for tid, track in self.tracks.items():
            predicted = track.predict_center()
            for idx, det in enumerate(animals):
                dc = _center(det["box"])
                dist = ((dc[0] - predicted[0]) ** 2 + (dc[1] - predicted[1]) ** 2) ** 0.5 / diagonal
                overlap = _iou(track.box, det["box"])
                if overlap < config.TRACK_ASSOC_MIN_IOU and dist > config.TRACK_ASSOC_MAX_DISTANCE:
                    continue
                score = dist - 0.35 * overlap - 0.05 * float(det.get("det_confidence", 0.0))
                candidates.append((score, tid, idx))
        candidates.sort(key=lambda item: item[0])
        used_tracks, used_dets = set(), set()
        for score, tid, idx in candidates:
            if tid in used_tracks or idx in used_dets:
                continue
            pairs.append((tid, idx))
            used_tracks.add(tid)
            used_dets.add(idx)
            unmatched_tracks.discard(tid)
            unmatched_detections.discard(idx)

        alpha = config.TRACK_VELOCITY_SMOOTHING
        for tid, idx in pairs:
            track = self.tracks[tid]
            det = animals[idx]
            old_cx, old_cy = _center(track.box)
            new_cx, new_cy = _center(det["box"])
            raw_v = (new_cx - old_cx, new_cy - old_cy)
            track.velocity = (
                alpha * track.velocity[0] + (1 - alpha) * raw_v[0],
                alpha * track.velocity[1] + (1 - alpha) * raw_v[1],
            )
            track.box = list(det["box"])
            track.hits += 1
            track.age += 1
            track.misses = 0
            track.last_confidence = float(det.get("det_confidence", 0.0))
            track.max_confidence = max(track.max_confidence, track.last_confidence)
            track.conf_sum += track.last_confidence
            track.in_water = bool(det.get("in_water_zone"))
            track.water_hits += 1 if track.in_water else 0
            species = det.get("species")
            species_conf = float(det.get("species_confidence") or 0.0)
            if species and species != "unidentified":
                if species == track.species:
                    track.candidate_species = species
                    track.candidate_species_hits = 0
                    track.species_confidence = max(track.species_confidence, species_conf)
                elif species == track.candidate_species:
                    track.candidate_species_hits += 1
                    if track.candidate_species_hits >= config.TRACK_SPECIES_CONFIRM_HITS:
                        track.species = species
                        track.species_confidence = species_conf
                        track.candidate_species = "unidentified"
                        track.candidate_species_hits = 0
                else:
                    track.candidate_species = species
                    track.candidate_species_hits = 1
            track.species_trusted = bool(det.get("species_trusted", True))
            track.site_model = track.site_model or bool(det.get("site_model_active"))
            track.custom_hits += 1 if _from_site_model(det) else 0
            track.verified = track.verified or is_verified(det) or (
                track.species_trusted
                and track.species not in ("unidentified", "other", "")
                and track.species_confidence >= config.VERIFIED_SPECIES_CONFIDENCE)
            track.confirmed = track.confirmed or track.evaluate_confirmation()

        for tid in list(unmatched_tracks):
            track = self.tracks[tid]
            track.misses += 1
            track.age += 1
            if track.misses > config.TRACK_MAX_MISSES:
                del self.tracks[tid]

        for idx in unmatched_detections:
            det = animals[idx]
            tid = self._next_id
            self._next_id += 1
            conf0 = float(det.get("det_confidence", 0.0))
            track = Track(
                track_id=tid,
                box=list(det["box"]),
                species=det.get("species") or "unidentified",
                species_confidence=float(det.get("species_confidence") or 0.0),
                last_confidence=conf0,
                max_confidence=conf0,
                conf_sum=conf0,
                in_water=bool(det.get("in_water_zone")),
                water_hits=1 if det.get("in_water_zone") else 0,
                verified=is_verified(det),
                species_trusted=bool(det.get("species_trusted", True)),
                site_model=bool(det.get("site_model_active")),
                custom_hits=1 if _from_site_model(det) else 0,
            )
            track.confirmed = track.evaluate_confirmation()
            self.tracks[tid] = track

        output = []
        for tid, track in self.tracks.items():
            # Only emit active tracks; missed tracks are retained internally for
            # short re-association but should not be treated as fresh detections.
            if track.misses == 0:
                d = dict(animals[self._track_detection_index(tid, animals)] if False else {})
                # Rebuild from track state without depending on pair order.
                d.update({
                    "box": list(track.box),
                    "class_id": CLASS_ANIMAL,
                    "det_confidence": track.last_confidence,
                    "species": track.species,
                    "species_confidence": track.species_confidence,
                    "species_trusted": track.species_trusted,
                    "track_id": tid,
                    "track_hits": track.hits,
                    "track_age": track.age,
                    "track_misses": 0,
                    "track_confirmed": track.confirmed,
                    "in_water_zone": track.in_water,
                    "source": "track",
                })
                output.append(d)
        for d in output:
            area_ratio = _area(d["box"]) / max(1.0, frame_width * frame_height)
            cx, cy = _center(d["box"])
            center_dist = (((cx - frame_width / 2) ** 2 + (cy - frame_height / 2) ** 2) ** 0.5) / diagonal
            center_score = max(0.0, 1.0 - min(1.0, center_dist * 1.6))
            size_score = max(0.0, min(1.0, area_ratio / max(0.001, config.TARGET_BOX_AREA_RATIO)))
            persistence = min(1.0, d["track_hits"] / 10.0)
            species_score = min(1.0, float(d["species_confidence"] or 0.0)) if d["species_trusted"] else 0.0
            d["priority_score"] = (
                config.TARGET_PRIORITY_CONFIDENCE_WEIGHT * d["det_confidence"] +
                config.TARGET_PRIORITY_PERSISTENCE_WEIGHT * persistence +
                config.TARGET_PRIORITY_SIZE_WEIGHT * size_score +
                config.TARGET_PRIORITY_CENTER_WEIGHT * center_score +
                config.TARGET_PRIORITY_SPECIES_WEIGHT * species_score
            ) + (config.WATER_ZONE_PRIORITY_BONUS if d.get("in_water_zone") else 0.0)
        return output

    def _track_detection_index(self, tid, animals):
        # Kept as a small compatibility hook. Track state is the authoritative
        # representation; detections are not looked up by index in production.
        return 0

    def reset(self):
        """Forget all tracks (the camera moved to a new view; old pixel boxes are meaningless)."""
        self.tracks.clear()
        self._last_pose = None

    def get(self, track_id) -> Optional[Track]:
        return self.tracks.get(track_id)
