"""Click-to-zoom: the operator points at an animal the AI cannot see yet.

19/10: a camouflaged baboon (40x55 px at 16x, back turned) was missed by
MegaDetector, the site model and SpeciesNet even when cropped and magnified.
Optical zoom is what makes such an animal visible to them, so a click in the
preview window:

1. ``center`` - brings the clicked point to the middle with short slow pulses,
   measuring the real picture shift by image correlation after every pulse
   (works with the mirrored pan and the tilt the camera cannot report);
2. ``zoom``   - glides the zoom in (MANUAL_ZOOM_FACTOR, AbsoluteMove zoom steps,
   pan/tilt untouched, so the animal stays centred);
3. ``hold``   - keeps still for MANUAL_HOLD_SECONDS with the AI running.  A
   confirmed animal is then followed by the normal tracker.

``tick()`` is called from the 8 Hz control thread and never sleeps.
"""

import logging
import math
import time

import cv2
import numpy as np

import config

logger = logging.getLogger("wildlife_tracker.tracking")


def _small_gray(frame):
    h, w = frame.shape[:2]
    return cv2.cvtColor(cv2.resize(frame, (w // 2, h // 2)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def _mag(zoom):
    return 1.0 + config.PTZ_ZOOM_MAG_PER_UNIT * max(0.0, float(zoom))


class ManualLook:
    def __init__(self, camera, point, frame, pose, focus=None, clock=None, hold_seconds=None,
                 zoom_factor=None):
        self.camera, self.focus = camera, focus
        self.clock = clock or (lambda: time.monotonic())
        self.zoom_factor = config.MANUAL_ZOOM_FACTOR if zoom_factor is None else float(zoom_factor)
        self.approach = False          # set by main: going closer to an ALREADY confirmed animal
        self.auto = False              # set by main: an automatic look at something half-seen
        self.operator = False          # set by main: the operator clicked an animal
        self.hold_seconds = config.MANUAL_HOLD_SECONDS if hold_seconds is None else float(hold_seconds)
        self.point = (float(point[0]), float(point[1]))
        self.h, self.w = frame.shape[:2]
        self.ref = _small_gray(frame)
        self.win = cv2.createHanningWindow(self.ref.shape[::-1], cv2.CV_32F)
        self.phase, self.since = "center", self.clock()
        self.zoom0 = float(pose[2]) if pose is not None else 0.03
        self.gain = {"pan": None, "tilt": None}     # measured px of picture shift per pulse-second (signed)
        self.pulse = None                           # (axis, sign, seconds, started, shift_before)
        self.settled_at = self.clock()
        self.last_shift = (0.0, 0.0)
        self.glide = None
        self.done = False
        logger.info("Operator click at (%.0f, %.0f): centring, then zooming in", *self.point)

    # ---------------------------------------------------------------- state
    @property
    def detection_allowed(self):
        return self.phase == "hold"

    @property
    def status(self):
        return f"LOOK {self.phase}"

    @property
    def gate_time(self):
        return self.since

    def _set(self, phase):
        self.phase, self.since = phase, self.clock()

    def _shift(self, frame):
        (dx, dy), resp = cv2.phaseCorrelate(self.ref, _small_gray(frame), self.win)
        return dx * 2.0, dy * 2.0, resp

    # ---------------------------------------------------------------- tick
    def tick(self, frame, frame_t, pose):
        """Advance one control tick; returns True while the view may be changing."""
        if self.done:
            return False
        now = self.clock()
        if self.phase == "center":
            return self._center(frame, frame_t, pose, now)
        if self.phase == "zoom":
            return self._zoom(pose, now)
        if self.phase == "hold":
            if now - self.since >= self.hold_seconds:
                logger.info("Operator look finished; patrol resumes")
                self.done = True
            return False
        return False

    def _center(self, frame, frame_t, pose, now):
        """Bring the clicked point to the middle with the TRACKER's pulses.

        19/10: the old slow 0.20-level pulses (0.6 deg/s) gave up long before an
        animal 300 px off was centred, so the zoom-in went past it.  The tracker's
        pulse (direction and size measured on the real camera: one pulse halves a
        530-px error) is used instead; the remaining offset is MEASURED after each
        pulse by image correlation (no pose model needed).
        """
        if frame is None or frame_t < self.settled_at + config.MANUAL_SETTLE_SECONDS:
            return True
        sx, sy, resp = self._shift(frame)
        x, y = self.point[0] + sx, self.point[1] + sy
        ex, ey = x - self.w / 2.0, y - self.h / 2.0
        tol = config.MANUAL_CENTER_TOLERANCE * self.w * (0.5 if self.operator else 1.0)
        lost = resp < 0.05 or not (0 <= x < self.w and 0 <= y < self.h)
        if (abs(ex) <= tol and abs(ey) <= tol) or lost or now - self.since > config.MANUAL_CENTER_MAX_SECONDS:
            if lost:
                logger.warning("Operator look: lost the reference while centring; zooming here")
            self._start_zoom(pose)
            return True
        from tracking.ptz_tracker import plan_pulse
        pan, tilt, pulse = plan_pulse(ex, ey, self.w, self.h, deadzone=config.MANUAL_CENTER_TOLERANCE * 0.7,
                                      zoom=pose[2] if pose is not None else None)
        self.camera.move(pan, tilt, 0.0, pulse_seconds=pulse)      # blocking pulse, then stop
        self.settled_at = self.clock()
        return True

    def _start_zoom(self, pose):
        self.camera.stop()
        z0 = float(pose[2]) if pose is not None else self.zoom0
        cap = config.ZOOM_POSITION_MAX if not (self.auto or self.approach) else config.LOOK_MAX_ZOOM
        goal_mag = min(_mag(cap), _mag(z0) * self.zoom_factor)
        goal = (goal_mag - 1.0) / config.PTZ_ZOOM_MAG_PER_UNIT
        if self.focus is not None:
            self.focus.prepare_for_motion()
        dur = abs(math.log(goal_mag / _mag(z0))) / max(0.05, config.PTZ_ZOOM_GLIDE_RATE)
        self.glide = dict(z0=z0, goal=goal, t0=self.clock(), last=-1e9, duration=dur, final=False)
        logger.info("Operator look: zoom %.3f -> %.3f (%.1fx)", z0, goal, goal_mag)
        self._set("zoom")

    def _zoom(self, pose, now):
        g = self.glide
        if not g["final"]:
            frac = min(1.0, (now - g["t0"]) / max(1e-3, g["duration"]))
            if frac >= 1.0 or now - g["last"] >= config.PTZ_ZOOM_GLIDE_STEP_SECONDS:
                m0, m1 = _mag(g["z0"]), _mag(g["goal"])
                m = m0 * (m1 / m0) ** frac
                self.camera.zoom_only(g["goal"] if frac >= 1.0 else (m - 1.0) / config.PTZ_ZOOM_MAG_PER_UNIT)
                g["last"], g["final"] = now, frac >= 1.0
            return True
        if (pose is not None and abs(float(pose[2]) - g["goal"]) <= 0.01) or now - g["t0"] > g["duration"] + 8:
            if self.focus is not None and config.FOCUS_REFOCUS_AFTER_ZOOM:
                self.focus.request("zoom")
            self._set("hold")
            return True
        return True


class OperatorHold:
    """Someone steers the camera by hand (web page, joystick): watch where they point.

    19/10: the operator turned the camera onto resting baboons, the patrol saw
    the move as a fault ("camera moved AWAY from the target"), froze, and
    the baboons were not watched.  Now the patrol pauses, the AI keeps running
    on the still frames (a confirmed animal is followed as usual) and the
    patrol resumes after OPERATOR_RESUME_SECONDS without hand movement.
    """

    phase = "operator"

    def __init__(self, pose, clock=None, resume_seconds=None, reason="camera moved by hand"):
        self.clock = clock or (lambda: time.monotonic())
        self.since = self.last_motion = self.clock()
        self.done = False
        self.reason = reason
        self.resume_seconds = config.OPERATOR_RESUME_SECONDS if resume_seconds is None else float(resume_seconds)
        logger.info("%s (pose %s): patrol paused, watching this view for %.0f s", reason[:1].upper() + reason[1:],
                    None if pose is None else tuple(round(float(v), 4) for v in pose), self.resume_seconds)

    @property
    def detection_allowed(self):
        return self.clock() - self.last_motion >= 1.0

    @property
    def gate_time(self):
        return self.last_motion

    @property
    def status(self):
        left = self.resume_seconds - (self.clock() - self.last_motion)
        return f"OPERATOR VIEW (patrol in {max(0, left):.0f}s)"

    def note_motion(self):
        self.last_motion = self.clock()
        self.resume_seconds = config.OPERATOR_RESUME_SECONDS   # real hand movement: full hold

    def tick(self, frame, frame_t, pose):
        if not self.done and self.clock() - self.last_motion >= self.resume_seconds:
            logger.info("No hand movement for %.0f s: patrol resumes", self.resume_seconds)
            self.done = True
        return False
