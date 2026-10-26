"""Is a bird box a bird IN FLIGHT?  (26/10/2026)

Of the 8 bird detections of 24-26/10 checked by eye, 4 were real birds - all four
FLYING past the camera (27-448 px, box scores 0.42-0.86) - and each was seen by one
AI result only.  Confirmation needs two sightings and the AI looks every 5-9 s, so
all four were dropped.  The other 4 were trees, branches and stumps (two of them at
night, one even "confirmed").

A flying bird is somewhere else 0.2-1 s later; a tree is not, and a branch in the
wind has only shifted a little.  flight_score() aligns a frame taken just before or
after the analysed one to it (whole-frame phase correlation removes the camera's own
motion) and asks three things of the box:
  - did it change much more than the picture as a whole (noise, compression, light)?
  - did it change much more than its own surroundings (a tree in the wind, a cloud
    shadow change a whole area, not one box)?
  - is the change not undone by a small shift (a branch that swayed a few pixels)?
"""

import math
import re

import cv2
import numpy as np

import config

_BIRD = re.compile(
    r"bird|\b(?:aves|guineafowl|turkey|starling|roller|eagle|vulture|stork|heron|egret|ibis|hornbill|"
    r"bustard|ostrich|dove|pigeon|weaver|sparrow|hawk|kite|falcon|kestrel|buzzard|harrier|goshawk|owl|"
    r"crane|goose|duck|plover|lapwing|francolin|spurfowl|quail|kingfisher|bee-eater|sunbird|shrike|"
    r"drongo|oxpecker|barbet|turaco|coucal|cuckoo|swallow|swift|lark|pipit|wagtail|finch|bunting|"
    r"hammerkop|spoonbill|flamingo|pelican|cormorant|darter|jacana|thick-knee|courser|sandgrouse|"
    r"parrot|lovebird|mousebird|babbler|bulbul|flycatcher|wheatear|robin|thrush)s?\b", re.I)


def is_bird(label):
    """True for 'bird' (site model) and SpeciesNet bird names ('superb starling' ...)."""
    return bool(label) and bool(_BIRD.search(str(label)))


def to_small(frame, scale=None):
    """Grey copy at `scale` (default FLIGHT_SCALE) - what the frame ring keeps."""
    s = config.FLIGHT_SCALE if scale is None else scale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(grey, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _mean(img, x1, y1, x2, y2):
    return float(img[y1:y2, x1:x2].mean())


def align(analysed_small, neighbours):
    """Take the camera's own motion out of each neighbour: [(dt, small grey frame)] ->
    [(dt, aligned frame, |difference|, picture noise)], leaving out pairs phase correlation
    cannot place (a zoom step, a big turn).  Done once per AI result, for every bird box."""
    a = analysed_small.astype(np.float32)
    h, w = a.shape[:2]
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    out = []
    for dt, nb in neighbours:
        if nb.shape != a.shape:
            continue
        b = nb.astype(np.float32)
        # copies: with a window OpenCV 5 multiplies both inputs by it in place
        (dx, dy), resp = cv2.phaseCorrelate(a.copy(), b.copy(), window)
        if resp < 0.02:
            continue
        aligned = cv2.warpAffine(b, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        diff = cv2.absdiff(a, aligned)
        mx, my = int(abs(dx)) + 4, int(abs(dy)) + 4
        core = diff[my:h - my, mx:w - mx]
        if core.size:
            out.append((dt, aligned, diff, float(np.median(core)) + 1.0))
    return out


def box_flight(box, analysed_small, aligned, scale=None):
    """box: full-resolution box in the ANALYSED frame; analysed_small: to_small(analysed frame);
    aligned: align(analysed_small, neighbours).
    -> (score, dt): score = how many times more the box changed than the picture did, for the
    best neighbour in which the box content left; (0.0, None) when no neighbour shows flight."""
    s = config.FLIGHT_SCALE if scale is None else scale
    a = analysed_small.astype(np.float32)
    h, w = a.shape[:2]
    x1, y1, x2, y2 = (float(v) * s for v in box)
    bx1, by1 = int(max(0, math.floor(x1))), int(max(0, math.floor(y1)))
    bx2, by2 = int(min(w, math.ceil(x2))), int(min(h, math.ceil(y2)))
    if bx2 - bx1 < 3 or by2 - by1 < 3:
        return 0.0, None
    size = max(bx2 - bx1, by2 - by1)
    # surroundings: one box size around it
    rx1, ry1, rx2, ry2 = max(0, bx1 - size), max(0, by1 - size), min(w, bx2 + size), min(h, by2 + size)
    box_area = (bx2 - bx1) * (by2 - by1)
    ring_area = (rx2 - rx1) * (ry2 - ry1) - box_area
    reach = max(2, int(round(config.FLIGHT_STAY_SHIFT * size)))
    sx1, sy1, sx2, sy2 = max(0, bx1 - reach), max(0, by1 - reach), min(w, bx2 + reach), min(h, by2 + reach)
    patch = a[by1:by2, bx1:bx2]
    best = (0.0, None)
    for dt, frame, diff, noise in aligned:
        box_level = _mean(diff, bx1, by1, bx2, by2)
        ratio = box_level / noise
        if box_level < config.FLIGHT_MIN_DIFF or ratio < config.FLIGHT_MIN_RATIO:
            continue
        if ring_area > 0:
            ring_level = (_mean(diff, rx1, ry1, rx2, ry2) * (ring_area + box_area) - box_level * box_area) / ring_area
            if box_level < config.FLIGHT_MIN_LOCAL * (ring_level + 1.0):
                continue                               # the whole area changed: wind, a shadow
        # undone by a small shift?  (a branch swaying a few pixels)
        area = frame[sy1:sy2, sx1:sx2]
        if area.shape[0] > patch.shape[0] and area.shape[1] > patch.shape[1]:
            here = float(np.sqrt(np.mean((patch - frame[by1:by2, bx1:bx2]) ** 2)))
            fit = cv2.matchTemplate(area, patch, cv2.TM_SQDIFF)
            closest = math.sqrt(max(0.0, float(fit.min())) / patch.size)
            if closest <= config.FLIGHT_STAY_RESIDUAL * here:
                continue
        if ratio > best[0]:
            best = (ratio, dt)
    return best


def flight_score(box, analysed_small, neighbours, scale=None):
    """box_flight() for one box straight from the neighbour frames."""
    return box_flight(box, analysed_small, align(analysed_small, neighbours), scale)
