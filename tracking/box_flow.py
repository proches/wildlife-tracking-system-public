"""Frame-to-frame box tracker (Lucas-Kanade optical flow) between slow AI results.

Measured 20/10/2026: on this CPU an AI result arrives every 4-5.5 s (inference
~3.7 s).  A walking animal moves 500-800 px at 20x in that time, so a tracker
that only reacts to AI results lags and "freezes" on the spot where the animal
was first seen.  This tracker follows the box on EVERY control tick (~8 Hz):

* the camera's own motion between two frames is measured first (phase
  correlation on a small grey copy) and used as the starting guess, so the
  large picture jumps after a PTZ pulse do not break the flow;
* feature points inside the box are tracked with pyramidal Lucas-Kanade and a
  forward-backward check; the box moves by the median displacement of the
  points that survive (scale from their spread, clamped);
* too few consistent points -> ``lost`` (the AI result re-anchors it later).

The AI stays the authority: every fresh AI result re-anchors the box
(``anchor``).  Nothing here commands the camera.
"""

import cv2
import numpy as np

import config


class BoxFlowTracker:
    def __init__(self):
        self.box = None
        self.ok = False
        self.prev = None          # previous small grey frame
        self.scale = 1.0          # full-res px per small px
        self.points = None        # (N, 1, 2) float32 in small coordinates
        self.last_motion = (0.0, 0.0)
        self.updates = 0
        self.fail_reason = ""

    # ------------------------------------------------------------------ helpers
    def _small(self, frame):
        h, w = frame.shape[:2]
        sw = config.FLOW_WIDTH
        self.scale = w / float(sw)
        g = cv2.cvtColor(cv2.resize(frame, (sw, int(round(h * sw / w))), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        return g

    def _seed_points(self, gray, box_s):
        # inner part of the box only: its border is background, which moves with the
        # camera, not with the animal
        bx1, by1, bx2, by2 = box_s
        mx, my = (bx2 - bx1) * (1 - config.FLOW_INNER) / 2, (by2 - by1) * (1 - config.FLOW_INNER) / 2
        x1, y1, x2, y2 = [int(round(v)) for v in (bx1 + mx, by1 + my, bx2 - mx, by2 - my)]
        h, w = gray.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        mask = np.zeros_like(gray)
        mask[y1:y2 + 1, x1:x2 + 1] = 255
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=config.FLOW_MAX_POINTS, qualityLevel=0.01,
                                      minDistance=2, mask=mask, blockSize=3)
        if pts is None or len(pts) < config.FLOW_MIN_POINTS:
            # low-texture animal (camouflage, far): a grid inside the box still flows
            xs = np.linspace(x1 + 1, x2 - 1, 5)
            ys = np.linspace(y1 + 1, y2 - 1, 5)
            grid = np.array([[[x, y]] for y in ys for x in xs], np.float32)
            pts = grid if pts is None else np.vstack([pts.astype(np.float32), grid])
        return pts.astype(np.float32)

    def _global_shift(self, prev, cur):
        # only a starting guess for the point flow: without a window, a smooth picture (high
        # zoom, haze) can report a confident (0, 0) while the camera moves - never use it as
        # proof that the camera did not move (21/10: that dropped a walking animal at 25x)
        (dx, dy), resp = cv2.phaseCorrelate(prev.astype(np.float32), cur.astype(np.float32))
        return (dx, dy) if resp > 0.05 else (0.0, 0.0)

    # ------------------------------------------------------------------ API
    def reset(self):
        self.__init__()

    def anchor(self, frame, box):
        """(Re)start on ``frame`` (the frame ``box`` belongs to)."""
        g = self._small(frame)
        s = self.scale
        box_s = [box[0] / s, box[1] / s, box[2] / s, box[3] / s]
        pts = self._seed_points(g, box_s)
        self.prev, self.points = g, pts
        self.box = list(map(float, box))
        self.ok = pts is not None
        self.fail_reason = "" if self.ok else "box too small"
        self.updates = 0
        return self.ok

    def update(self, frame):
        """Follow the box into ``frame``.  Returns the new full-res box or None (lost)."""
        if not self.ok or self.prev is None or self.points is None:
            return None
        g = self._small(frame)
        gdx, gdy = self._global_shift(self.prev, g)
        self.last_motion = (gdx * self.scale, gdy * self.scale)
        p0 = self.points
        guess = p0 + np.array([gdx, gdy], np.float32)
        lk = dict(winSize=(config.FLOW_WIN, config.FLOW_WIN), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, guess.copy(),
                                              flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
        if p1 is None:
            return self._lose(g, "flow failed")
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, p1, p0.copy(),
                                               flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < config.FLOW_FB_MAX)
        if good.sum() < config.FLOW_MIN_POINTS:
            return self._lose(g, f"{int(good.sum())} consistent points")
        a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        d = b - a
        med = np.median(d, axis=0)
        spread = np.median(np.linalg.norm(d - med, axis=1))
        if spread > config.FLOW_MAX_SPREAD:
            return self._lose(g, f"inconsistent motion ({spread:.1f} px)")
        # scale from the change of the points' spread around their centre (clamped)
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        ra, rb = np.median(np.linalg.norm(a - ca, axis=1)), np.median(np.linalg.norm(b - cb, axis=1))
        k = float(np.clip(rb / ra, 0.85, 1.18)) if ra > 1.0 else 1.0
        s = self.scale
        x1, y1, x2, y2 = self.box
        cx, cy = (x1 + x2) / 2 + med[0] * s, (y1 + y2) / 2 + med[1] * s
        hw, hh = (x2 - x1) / 2 * k, (y2 - y1) / 2 * k
        h, w = frame.shape[:2]
        if not (-hw < cx < w + hw and -hh < cy < h + hh):
            return self._lose(g, "left the picture")
        self.box = [cx - hw, cy - hh, cx + hw, cy + hh]
        self.prev = g
        self.points = b.reshape(-1, 1, 2).astype(np.float32)
        if len(self.points) < config.FLOW_MAX_POINTS // 2:       # top up from the new box
            more = self._seed_points(g, [v / s for v in self.box])
            if more is not None:
                self.points = np.vstack([self.points, more])[:config.FLOW_MAX_POINTS]
        self.updates += 1
        return list(self.box)

    def _lose(self, g, why):
        self.ok, self.prev, self.points, self.fail_reason = False, g, None, why
        return None
