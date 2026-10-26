"""Motion cue: something moves on its own in the picture (an animal walking, a bird flying).

19/10/2026: a warthog and resting/walking baboons were missed - the detectors
did not recognise them, so nothing ever told the patrol to stop.  Movement is a
second, detector-independent clue:

* frames are compared at a few Hz on a small grey copy (cheap on the i5);
* while the camera pans, the previous frame is first shifted by the measured
  global image motion (phase correlation), so the camera's own movement cancels;
* a blob counts only if it is compact (not a whole bush shaking), persists over
  several comparisons near the same place, and the scene is not "boiling"
  (wind or heat haze moving everything at once).

``update()`` returns the strongest cue as (cx, cy, w, h) in full-frame pixels, or None.
"""

import cv2
import numpy as np

import config


class MotionCue:
    def __init__(self):
        self.prev = None
        self.prev_t = 0.0
        self.history = []          # [(t, [(cx, cy, w, h), ...]), ...] in small-image pixels
        self.scale = 1.0

    def reset(self):
        self.prev = None
        self.history = []

    def update(self, frame, t, camera_moving=False):
        if frame is None:
            return None
        if self.prev is not None and t - self.prev_t < config.MOTION_CUE_PERIOD:
            return None
        h, w = frame.shape[:2]
        sw = config.MOTION_CUE_WIDTH
        self.scale = w / float(sw)
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (sw, int(h * sw / w))), cv2.COLOR_BGR2GRAY),
                                 (5, 5), 0).astype(np.float32)
        prev, self.prev, self.prev_t = self.prev, small, t
        if prev is None or prev.shape != small.shape:
            return None
        if camera_moving:
            (dx, dy), resp = cv2.phaseCorrelate(prev, small)
            if resp < 0.1 or abs(dx) > sw * 0.2:
                self.history = []
                return None
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            prev = cv2.warpAffine(prev, m, (small.shape[1], small.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            margin = int(abs(dx)) + 3
        else:
            margin = 2
        diff = cv2.absdiff(prev, small)
        diff[:margin, :] = 0
        diff[-margin:, :] = 0
        diff[:, :margin] = 0
        diff[:, -margin:] = 0
        noise = float(np.median(diff))
        thr = max(config.MOTION_CUE_MIN_DIFF, noise * 4.0)
        mask = (diff > thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        blobs = []
        sh, sw2 = small.shape
        em = config.MOTION_CUE_EDGE_MARGIN
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            ccx, ccy = cents[i]
            if not (em * sw2 <= ccx <= (1 - em) * sw2 and em * sh <= ccy <= (1 - em) * sh):
                continue        # frame border: near-field leaves/grass flicking in and out (19/10)
            if config.MOTION_CUE_MIN_AREA <= area <= config.MOTION_CUE_MAX_AREA and max(bw, bh) <= 4 * min(bw, bh) + 4:
                blobs.append((float(cents[i][0]), float(cents[i][1]), float(bw), float(bh)))
        if len(blobs) > config.MOTION_CUE_MAX_BLOBS:     # wind / heat haze: everything moves
            self.history = []
            return None
        self.history = [(ht, hb) for ht, hb in self.history if t - ht <= config.MOTION_CUE_WINDOW] + [(t, blobs)]
        best = None
        for cx, cy, bw, bh in blobs:
            # follow this blob back through the window (nearest match each step)
            chain = [(cx, cy)]
            for _, hb in reversed(self.history[:-1]):
                px, py = chain[-1]
                near = [(x, y) for x, y, _, _ in hb
                        if abs(px - x) <= config.MOTION_CUE_MATCH_PX and abs(py - y) <= config.MOTION_CUE_MATCH_PX]
                if not near:
                    break
                chain.append(min(near, key=lambda q: (q[0] - px) ** 2 + (q[1] - py) ** 2))
            seen = len(chain) - 1
            travel = ((chain[0][0] - chain[-1][0]) ** 2 + (chain[0][1] - chain[-1][1]) ** 2) ** 0.5
            # 19/10: leaves of a big tree swaying in the wind were taken for an animal and
            # zoomed in on.  Foliage shakes IN PLACE; an animal or a bird goes somewhere.
            if (seen >= config.MOTION_CUE_MIN_REPEATS and travel >= config.MOTION_CUE_MIN_TRAVEL
                    and (best is None or seen > best[0])):
                best = (seen, cx, cy, bw, bh)
        if best is None:
            return None
        _, cx, cy, bw, bh = best
        s = self.scale
        return cx * s, cy * s, max(24.0, bw * s), max(24.0, bh * s)
