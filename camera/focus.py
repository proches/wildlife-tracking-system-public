"""Focus management for the PTZ lens (v3.1, measured on the SUNBA camera).

What the real lens does (tools measured on 2026-10-16, zoom 14.8x):

* In AUTO the lens refocuses by itself after every zoom move and is sharp
  ~2-3 s after the zoom stops.  While the zoom travels the picture is soft
  (sharpness 5 000 vs 20 000 when settled).
* Focus is extremely critical when zoomed: +-0.01 of focus position drops the
  sharpness 10-25x.
* Switching MANUAL -> AUTO does NOT trigger a refocus.  The v3 "nudge, then
  hand back to AUTO" hunt could therefore leave the lens badly out of focus.

So this controller:

1. keeps the lens in AUTO, which is what refocuses after zooms;
2. after a zoom (``request``) waits until sharpness is *stable*, then compares
   it with the best sharpness seen for that pan/zoom;
3. only if the picture is clearly soft, does a small software search with
   ABSOLUTE focus positions around the current one.  It keeps the better
   position (MANUAL, "locked") or returns the lens exactly to where it was;
4. ``prepare_for_motion()`` puts a locked lens back in AUTO before the next
   zoom so the lens's own AF follows the zoom again;
5. ``frame_is_sharp()`` lets the control loop refuse to send soft frames to
   the AI - blurred frames were a source of ghost animals.

Sharpness is the variance of the Laplacian over the centre of the frame (the
camera's on-screen text is outside that area).  Everything is time-boxed, runs
in its own thread, and does nothing when the camera has no Imaging service.
"""

import logging
import threading
import time

import cv2

import config

logger = logging.getLogger("wildlife_tracker.focus")


def sharpness(frame_bgr):
    """Laplacian variance of the central part of the frame (higher = sharper)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0
    h, w = frame_bgr.shape[:2]
    crop = frame_bgr[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FocusController:
    def __init__(self, camera, frame_source, pose_source=None, clock=time.monotonic, sleep=time.sleep):
        self.camera = camera
        self.frame_source = frame_source      # callable -> latest BGR frame or None
        self.pose_source = pose_source        # callable -> (pan, tilt, zoom) or None
        self.clock = clock
        self.sleep = sleep
        self._thread = None
        self._lock = threading.Lock()
        self._best = {}                       # pose bucket -> decaying best sharpness
        self.locked = False                   # True while a searched MANUAL focus is held
        self.last_sharpness = 0.0
        self.last_result = ""
        self.enabled = bool(config.FOCUS_CONTROL_ENABLED and getattr(camera, "focus_available", False))
        if self.enabled:
            info = camera.get_focus_info()
            logger.info("Focus control enabled; lens reports: %s", info)
            if config.FOCUS_AUTOFOCUS_MODE.upper() == "AUTO":
                camera.set_autofocus(True)
        else:
            logger.info("Focus control not available (no ONVIF Imaging service or disabled)")

    @property
    def busy(self):
        t = self._thread
        return t is not None and t.is_alive()

    @staticmethod
    def _bucket(pose):
        if pose is None:
            return None
        return (round(pose[0] / 0.02), round(pose[2] / 0.02))

    def _best_now(self, bucket):
        """Best sharpness seen at this pose, faded by FOCUS_BEST_DECAY per second."""
        value, t = self._best.get(bucket, (0.0, 0.0))
        if value <= 0.0:
            return 0.0
        return value * config.FOCUS_BEST_DECAY ** max(0.0, self.clock() - t)

    def _note_best(self, bucket, value):
        if value >= self._best_now(bucket):
            self._best[bucket] = (float(value), self.clock())

    # ------------------------------------------------------------ public API
    def request(self, reason=""):
        """Check (and if needed fix) focus in the background after a zoom."""
        if not self.enabled or self.busy:
            return False
        with self._lock:
            self._thread = threading.Thread(target=self._run, args=(reason,), name="focus", daemon=True)
            self._thread.start()
        return True

    def prepare_for_motion(self):
        """Call before a zoom move: a locked (MANUAL) lens goes back to AUTO."""
        if self.enabled and self.locked and not self.busy:
            self.locked = False
            self.camera.set_autofocus(True)

    def frame_is_sharp(self, frame_bgr, pose):
        """True if the frame is sharp enough to be worth running the AI on.

        Compared with a slowly decaying best-seen value for this pose, so heat
        haze at midday (everything a bit softer) does not block the AI forever.
        """
        if not config.DETECT_REQUIRE_SHARP_FRAME:
            return True
        s = sharpness(frame_bgr)
        self.last_sharpness = s
        bucket = self._bucket(pose)
        best = self._best_now(bucket)
        self._note_best(bucket, s)
        return best <= 0.0 or s >= config.DETECT_MIN_SHARPNESS_RATIO * best

    # ------------------------------------------------------------ internals
    def _measure(self, n=3, pause=0.25):
        vals = []
        for _ in range(n):
            self.sleep(pause)
            vals.append(sharpness(self.frame_source()))
        vals.sort()
        return vals[len(vals) // 2] if vals else 0.0

    def _wait_stable(self):
        """Wait until two consecutive sharpness readings agree (AF finished)."""
        started = self.clock()
        prev = self._measure(n=2, pause=0.3)
        while self.clock() - started < config.FOCUS_SETTLE_MAX_SECONDS:
            cur = self._measure(n=2, pause=0.3)
            if prev > 0 and abs(cur - prev) <= 0.12 * max(cur, prev):
                return max(cur, prev)
            prev = cur
        return prev

    def _probe(self, position):
        self.camera.focus_absolute(position)
        self.sleep(config.FOCUS_MOVE_WAIT_SECONDS)
        return self._measure(n=2, pause=0.2)

    def _search(self, f0, s0, deadline):
        """Hill-climb with absolute focus positions around ``f0``."""
        step = config.FOCUS_SEARCH_STEP
        best_pos, best_val = f0, s0
        up = self._probe(f0 + step)
        down = self._probe(f0 - step) if self.clock() < deadline else 0.0
        if max(up, down) <= s0 * 1.03:
            direction = 0
        else:
            direction = 1 if up >= down else -1
            best_pos, best_val = (f0 + step, up) if direction > 0 else (f0 - step, down)
        while direction and self.clock() < deadline:
            nxt = best_pos + direction * step
            if not (0.0 <= nxt <= 1.0):
                break
            val = self._probe(nxt)
            if val <= best_val * 1.03:
                break
            best_pos, best_val = nxt, val
        # refine around the best with half steps
        for offset in (step / 2, -step / 2):
            if self.clock() >= deadline:
                break
            val = self._probe(best_pos + offset)
            if val > best_val * 1.03:
                best_pos, best_val = best_pos + offset, val
        return best_pos, best_val

    def _run(self, reason):
        started = self.clock()
        deadline = started + config.FOCUS_HUNT_MAX_SECONDS
        try:
            s0 = self._wait_stable()
            self.last_sharpness = s0
            pose = self.pose_source() if self.pose_source else None
            bucket = self._bucket(pose)
            best = self._best_now(bucket)
            if best <= 0 or s0 >= best * config.FOCUS_SHARPNESS_MIN_RATIO:
                self._note_best(bucket, s0)
                self.last_result = f"sharp ({s0:.0f})"
                return
            f0 = self.camera.get_focus_position()
            if f0 is None:
                self.last_result = f"soft ({s0:.0f} vs {best:.0f}); no focus position"
                return
            logger.info("Image soft after %s (sharpness %.0f vs best %.0f, focus=%.4f): searching",
                        reason, s0, best, f0)
            self.camera.set_autofocus(False)
            pos, val = self._search(f0, s0, deadline)
            if val > s0 * 1.15:
                self.camera.focus_absolute(pos)
                self.locked = True
                self._note_best(bucket, val)
                self.last_result = f"focus locked at {pos:.4f} ({s0:.0f}->{val:.0f})"
            else:
                # Put the lens back EXACTLY where AF had it, then hand back to AUTO
                # (switching to AUTO alone does not refocus on this camera).
                self.camera.focus_absolute(f0)
                self.sleep(config.FOCUS_MOVE_WAIT_SECONDS)
                self.camera.set_autofocus(True)
                self.locked = False
                self.last_result = f"autofocus kept ({s0:.0f}, search found {val:.0f})"
            logger.info("Focus: %s", self.last_result)
        except Exception as exc:  # never let focus work crash the system
            logger.warning("Focus task failed: %s", exc)
            self.last_result = f"error: {exc}"
            try:
                self.camera.set_autofocus(True)
                self.locked = False
            except Exception:
                pass
