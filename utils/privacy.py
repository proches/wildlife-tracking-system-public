"""Guest privacy: views that show the lodge pool / sun loungers are never shown or analysed.

19/10/2026: with the camera tilted down (low patrol row) the pans next to the
water-hole look straight at the pool, the green loungers and towels.  A view
is private when its pan is inside a PRIVACY_ZONES range AND the tilt reading is
lower than that zone's ``max_tilt`` (looking further down than the plain).
"""

import cv2

import config


def in_privacy_zone(pose):
    if pose is None:
        return False
    try:
        pan, tilt = float(pose[0]), float(pose[1])
    except (TypeError, ValueError, IndexError):
        return False
    try:
        zoom = float(pose[2])
    except (TypeError, ValueError, IndexError):
        zoom = 1.0
    for z in getattr(config, "PRIVACY_ZONES", []):
        low, high = z["pan"]
        if low <= pan <= high and (tilt > z["max_tilt"] or zoom < z.get("min_zoom", -1.0)):
            return True
    return False


def path_crosses_privacy(pan_from, pan_to, tilt):
    """Would a pan glide at this tilt pass through a private view?"""
    lo, hi = sorted((float(pan_from), float(pan_to)))
    for z in getattr(config, "PRIVACY_ZONES", []):
        zl, zh = z["pan"]
        if tilt > z["max_tilt"] and lo <= zh and hi >= zl:
            return True
    return False


def privacy_filter(frame, pose):
    """Heavily blurred frame with a notice when the view is private (stream, clips, preview)."""
    if frame is None or not in_privacy_zone(pose):
        return frame
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 48), max(1, h // 48)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.putText(out, "PRIVACY - camera repositioning", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (255, 255, 255), 3)
    return out
