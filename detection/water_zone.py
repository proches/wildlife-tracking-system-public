"""Water-hole region of interest.

The water-hole is where animals and birds gather, so the system treats it
specially: detections inside it get a priority bonus and a lower confirmation
bar, an extra full-resolution detection pass is run over it, and visits are
logged.  The region is a polygon drawn once with ``tools/calibrate_water_roi.py``
while the camera is at the water-hole pose; it is stored per camera pose
because pixel positions only mean something for the pose they were drawn at.

File format (water_roi.json)::

    {"zones": [{"name": "water_wide", "pose": {"pan": .., "tilt": .., "zoom": ..},
                "polygon": [[x, y], ...]}]}      # x, y normalised 0..1
"""

import json
import logging
import os

import config

logger = logging.getLogger("wildlife_tracker.water")


def _point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


class WaterZone:
    def __init__(self, path=None):
        self.path = path or config.WATER_ROI_FILE
        self.zones = []
        self.load()

    @property
    def enabled(self):
        return bool(self.zones)

    def load(self):
        self.zones = []
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for z in data.get("zones", []):
                poly = [(float(x), float(y)) for x, y in z["polygon"]]
                pose = z["pose"]
                if len(poly) >= 3:
                    self.zones.append(dict(
                        name=z.get("name", "water"),
                        pose=(float(pose["pan"]), float(pose["tilt"]), float(pose["zoom"])),
                        polygon=poly,
                    ))
            logger.info("Water zones loaded: %s", [z["name"] for z in self.zones])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Could not load %s: %s", self.path, exc)
            self.zones = []

    def zone_for_pose(self, pose):
        if pose is None:
            return None
        tp, tt, tz = config.STATIC_FP_POSE_TOLERANCE
        for z in self.zones:
            zp, zt, zz = z["pose"]
            if abs(pose[0] - zp) <= tp and abs(pose[1] - zt) <= tt and abs(pose[2] - zz) <= tz:
                return z
        return None

    def in_zone(self, box, frame_w, frame_h, pose):
        """True if the box's centre (or its bottom-centre, where animals touch the
        ground) lies inside the water polygon valid for the current pose."""
        z = self.zone_for_pose(pose)
        if z is None:
            return False
        cx = (box[0] + box[2]) / 2.0 / frame_w
        cy = (box[1] + box[3]) / 2.0 / frame_h
        fy = box[3] / frame_h
        poly = z["polygon"]
        return _point_in_polygon(cx, cy, poly) or _point_in_polygon(cx, fy, poly)

    def roi_rect(self, frame_w, frame_h, pose, pad=0.08):
        """Pixel rectangle (x1, y1, x2, y2) around the zone for the current pose, or None."""
        z = self.zone_for_pose(pose)
        if z is None:
            return None
        xs = [p[0] for p in z["polygon"]]
        ys = [p[1] for p in z["polygon"]]
        x1, x2 = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
        y1, y2 = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
        return int(x1 * frame_w), int(y1 * frame_h), int(x2 * frame_w), int(y2 * frame_h)
