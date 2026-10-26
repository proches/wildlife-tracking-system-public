"""
Decides WHICH detected animal the camera should actually point at, frame
to frame. This is deliberately separate from PTZTracker (which only
decides HOW to move once a target is chosen) - it fixes the camera
swinging onto a stray "animal" detection that's really foliage/a termite
mound/wind-blown grass, or whipping between two real animals whenever
their confidences happen to cross over.

Two simple ideas, no heavyweight multi-object tracker needed:

  1. Stickiness: once locked onto an animal, keep following THAT SAME
     box (by position) as long as something appears near where it last
     was, instead of re-picking "whatever's highest confidence" fresh
     every single frame.
  2. Switch confirmation: a different candidate must be the best
     available option for several consecutive frames before the camera
     commits to switching onto it - a one-frame flicker is never enough
     on its own to pull the camera off a real, currently-tracked animal.
"""

import config
from detection.constants import CLASS_ANIMAL


class TargetSelector:
    def __init__(self):
        self.current_box = None
        self._lock_miss_count = 0
        self._candidate_box = None
        self._candidate_count = 0

    def select(self, detections, frame_width, frame_height):
        """
        detections: this frame's list of dicts from WildlifePipeline.
        Returns the box [x1,y1,x2,y2] to point the camera at this frame,
        or None if nothing is currently locked/confirmed.
        """
        diagonal = (frame_width ** 2 + frame_height ** 2) ** 0.5
        eligible = [
            d for d in detections
            if d["class_id"] == CLASS_ANIMAL
            and d["det_confidence"] >= config.TRACKING_CONFIDENCE_THRESHOLD
        ]

        if self.current_box is not None:
            nearby = [d for d in eligible if self._near(d["box"], self.current_box, diagonal)]
            if nearby:
                best = max(nearby, key=lambda d: d["det_confidence"])
                self.current_box = best["box"]
                self._lock_miss_count = 0
                self._candidate_box = None
                self._candidate_count = 0
                return self.current_box

            # Locked animal not seen near its last spot this frame. Give
            # PTZTracker a clean "no target" so its own missed-frame/patrol
            # logic can respond, but don't give up the lock instantly -
            # a brief occlusion shouldn't need re-confirming from scratch.
            self._lock_miss_count += 1
            if self._lock_miss_count >= config.LOST_TARGET_FRAMES:
                self.current_box = None
                self._lock_miss_count = 0
            return None

        # No lock right now - require a candidate to be the best option
        # for several frames running before committing to it.
        if not eligible:
            self._candidate_box = None
            self._candidate_count = 0
            return None

        best = max(eligible, key=lambda d: d["det_confidence"])
        if self._candidate_box is not None and self._near(best["box"], self._candidate_box, diagonal):
            self._candidate_count += 1
        else:
            self._candidate_box = best["box"]
            self._candidate_count = 1

        if self._candidate_count >= config.TARGET_SWITCH_CONFIRM_FRAMES:
            self.current_box = self._candidate_box
            self._candidate_box = None
            self._candidate_count = 0
            return self.current_box

        return None

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _near(self, box_a, box_b, diagonal):
        ax, ay = self._center(box_a)
        bx, by = self._center(box_b)
        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return dist / diagonal <= config.TARGET_STICKINESS_RADIUS
