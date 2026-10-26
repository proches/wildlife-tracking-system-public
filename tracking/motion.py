"""Closed-loop PTZ positioning with automatic AbsoluteMove / ContinuousMove selection.

Why this exists
---------------
Many ONVIF cameras accept an ``AbsoluteMove`` request (HTTP 200, no error) and
then simply do nothing, or they interpret the coordinates in a vendor space.
The old patrol trusted that call blindly: it "arrived" nowhere, timed out,
moved on to the next station, and the camera stayed parked on one side.

``MotionController`` never trusts a command.  Every ``step`` reads the real
pose, compares it with the target and:

* ``absolute`` mode - issues AbsoluteMove, then verifies the pose really
  changed.  After PTZ_ABSOLUTE_FAILS_BEFORE_FALLBACK silent failures it
  switches (and remembers, in logs/ptz_capabilities.json) to ...
* ``continuous`` mode - proportional velocity control from the measured pose:
  speed = clip(kp * error), refreshed every control tick, with a device-side
  auto-stop timeout, hard pan guards and slow-down near the arc edges.

Both modes report ``arrived`` / ``moving`` / ``stalled``.
"""

import json
import logging
import math
import os
import time

import config

logger = logging.getLogger("wildlife_tracker.motion")

ARRIVED, MOVING, STALLED = "arrived", "moving", "stalled"


def _load_caps():
    try:
        with open(config.PTZ_CAPABILITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}                     # first run: nothing measured yet
    except (OSError, ValueError) as exc:
        logger.warning("PTZ capability file %s unreadable (%s) - AbsoluteMove will be re-tested",
                       config.PTZ_CAPABILITIES_FILE, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("PTZ capability file %s is not a JSON object - ignored", config.PTZ_CAPABILITIES_FILE)
        return {}
    return data


def _save_caps(data):
    try:
        os.makedirs(os.path.dirname(config.PTZ_CAPABILITIES_FILE) or ".", exist_ok=True)
        with open(config.PTZ_CAPABILITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


class MotionController:
    def __init__(self, camera, clock=time.monotonic):
        self.camera = camera
        self.clock = clock
        mode = str(getattr(config, "PTZ_MOVE_MODE", "auto")).lower()
        caps = _load_caps() if mode == "auto" else {}
        self.mode = mode
        self.absolute_ok = {"absolute": True, "continuous": False}.get(mode)
        self.pan_sign = getattr(camera, "pan_command_sign", 1)
        if mode == "auto":
            # Remembered verdict from an earlier run (delete the file to re-test).
            # A verdict taken with a different pan sign is void: before v3.1 the
            # mirrored EFlip camera made every AbsoluteMove "fail".
            if caps.get("pan_command_sign", 1) == self.pan_sign:
                self.absolute_ok = caps.get("absolute_move_works")  # None = unknown
            else:
                self.absolute_ok = None
        # Verified by tools/ptz_selftest.py with a measured pan sign: a few odd
        # readings at runtime must not switch AbsoluteMove off.
        self._selftest_verified = bool(mode == "auto" and caps.get("absolute_move_works")
                                       and caps.get("pan_command_sign") == self.pan_sign
                                       and "checked_at" in caps)
        self._abs_fails = 0
        self.target = None
        self._issued_at = None
        self._issue_err = None
        self._best_err = None
        self._best_at = None
        self.last_command = ""
        self.fault = None   # set by a safety stop; motion disabled until a retry
        self.fault_at = None
        self.fault_count = 0

    # ------------------------------------------------------------------ API
    def clear_fault(self):
        self.fault = None
        self.target = None

    def set_target(self, pan=None, tilt=None, zoom=None, zoom_is_wide=False):
        """Begin a new move. ``None`` leaves that axis alone.

        ``zoom_is_wide=True`` means "zoom out to at least this wide": being
        wider than the target counts as arrived (the lens cannot go below its
        physical minimum, which may read slightly under the target).
        """
        self.target = dict(pan=pan, tilt=tilt, zoom=zoom, zoom_is_wide=bool(zoom_is_wide))
        self._cine_sent, self._cine_speed = None, 0.0
        self._issued_at = None
        self._issue_err = None
        self._best_err = None
        self._best_at = self.clock()
        self.last_command = ""

    def cancel(self):
        self.target = None
        try:
            self.camera.stop()
        except Exception:
            pass

    @staticmethod
    def errors(target, pose):
        pan, tilt, zoom = pose
        ex = None if target["pan"] is None else target["pan"] - pan
        ey = None if target["tilt"] is None else target["tilt"] - tilt
        ez = None if target["zoom"] is None else target["zoom"] - zoom
        return ex, ey, ez

    def _within(self, ex, ey, ez, target, pan_tol=None):
        if ex is not None and abs(ex) > (config.PTZ_PAN_TOLERANCE if pan_tol is None else pan_tol):
            return False
        if ey is not None and abs(ey) > config.PTZ_TILT_TOLERANCE:
            return False
        if ez is not None:
            if target.get("zoom_is_wide"):
                if ez < -config.PTZ_ZOOM_TOLERANCE:
                    return False
            elif abs(ez) > config.PTZ_ZOOM_TOLERANCE:
                return False
        return True

    def step(self, pose):
        """Advance one control tick using the measured ``pose``.

        Returns ARRIVED, MOVING or STALLED.
        """
        if self.target is None:
            return ARRIVED
        if self.fault:
            self.camera.stop()
            # 19/10: an operator moving the camera by hand tripped the stop and the
            # patrol stayed frozen until restart.  The arc guards in the camera
            # class never depend on this, so retry a few times.
            if (self.fault_count < config.PTZ_FAULT_MAX_RETRIES
                    and self.clock() - (self.fault_at or 0.0) >= config.PTZ_FAULT_RETRY_SECONDS):
                logger.warning("Retrying motion after the safety stop (%d/%d)",
                               self.fault_count, config.PTZ_FAULT_MAX_RETRIES)
                self.clear_fault()
            return STALLED
        t = self.target
        ex, ey, ez = self.errors(t, pose)
        now = self.clock()
        cine = config.PTZ_CINEMATIC and ex is not None
        if cine:
            if not t.get("pan_coarse"):
                self._step_cinematic_pan(pose, ex, now)
                return MOVING          # travelling, or just stopped: let it slide and settle first
            elif now < t["settle_until"]:
                return MOVING          # let the coasting camera come to rest
            absolute_usable = self.mode == "absolute" or (self.mode == "auto" and self.absolute_ok is not False)
            if abs(ex) <= config.PTZ_CINEMATIC_FINAL_TOL or not absolute_usable or t.get("no_abs"):
                ex = None              # exact enough (or nothing precise to finish with)
        if self._within(ex, ey, ez, t, pan_tol=config.PTZ_CINEMATIC_FINAL_TOL if cine else None):
            self.camera.stop()
            self.last_command = "arrived"
            return ARRIVED

        total = sum(abs(e) for e in (ex, ey, ez) if e is not None)
        if self._best_err is None or total < self._best_err - config.PTZ_ABSOLUTE_MIN_PROGRESS:
            self._best_err, self._best_at = total, now
        elif now - self._best_at >= config.PTZ_STALL_SECONDS:
            self.camera.stop()
            logger.warning(
                "PTZ stalled: target=%s pose=(%.4f, %.4f, %.4f) mode=%s absolute_ok=%s",
                {k: (None if v is None else round(v, 4)) for k, v in t.items() if k != "zoom_is_wide"},
                pose[0], pose[1], pose[2], self.mode, self.absolute_ok,
            )
            return STALLED

        if self.mode == "absolute" or (self.mode == "auto" and self.absolute_ok is not False):
            self._step_absolute(pose, total, now)
        else:
            self._step_continuous(pose, ex, ey, ez)
        return MOVING

    # ------------------------------------------------------------ absolute
    def _step_absolute(self, pose, total, now):
        t = self.target
        reissue = self._issued_at is None
        if self._issued_at is not None and now - self._issued_at >= config.PTZ_ABSOLUTE_PROGRESS_TIMEOUT:
            made = (self._issue_err or 0.0) - total
            # a tiny finishing move (residual 0.003) cannot "progress" 0.004: judge
            # progress relative to the error the command had to remove
            needed = min(config.PTZ_ABSOLUTE_MIN_PROGRESS, 0.5 * (self._issue_err or 0.0))
            if made >= needed:
                if self.absolute_ok is None and self.mode == "auto":
                    self.absolute_ok = True
                    self._remember(True)
                    logger.info("PTZ: AbsoluteMove verified (moved %.4f in %.1fs)", made, now - self._issued_at)
                self._abs_fails = 0
                reissue = total > 0  # still travelling; refresh the command
            else:
                self._abs_fails += 1
                logger.warning(
                    "PTZ: AbsoluteMove did not move the camera (error %.4f -> %.4f, try %d/%d)",
                    self._issue_err or 0.0, total, self._abs_fails,
                    config.PTZ_ABSOLUTE_FAILS_BEFORE_FALLBACK,
                )
                if (self.mode == "auto" and self._abs_fails >= config.PTZ_ABSOLUTE_FAILS_BEFORE_FALLBACK
                        and not self._selftest_verified):
                    self.absolute_ok = False
                    self._remember(False)
                    logger.warning("PTZ: switching to closed-loop ContinuousMove (AbsoluteMove ignored by this camera)")
                    self._step_continuous(pose, *self.errors(t, pose))
                    return
                reissue = True
        if reissue:
            pan = pose[0] if t["pan"] is None else t["pan"]
            tilt = pose[1] if t["tilt"] is None else t["tilt"]
            zoom = t["zoom"]
            ok = self.camera.goto_position(pan, tilt, zoom=zoom)
            self._issued_at = now
            self._issue_err = total
            self.last_command = f"abs({pan:.3f},{tilt:.3f},{'-' if zoom is None else format(zoom, '.3f')})"
            if not ok:
                self._abs_fails += 1

    def _remember(self, works):
        caps = _load_caps()
        caps["absolute_move_works"] = bool(works)
        caps["pan_command_sign"] = self.pan_sign
        caps["decided_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save_caps(caps)

    # ----------------------------------------------------------- cinematic
    def _step_cinematic_pan(self, pose, ex, now):
        """Smooth pan travel at a constant cinematic speed (~8 deg/s).

        Measured on the SUNBA: speeds 0.24-0.30 all glide at the same constant
        rate; ContinuousMove Timeouts run in whole seconds; one 1-second command
        while gliding carries the camera PTZ_CINEMATIC_FINAL_GLIDE (0.056); the
        AI can freeze this control loop for 2-3 s.

        Safety invariant: a command is only sent if the farthest it can carry the
        camera (one second, then the camera stops by ITS OWN clock) does not pass
        the target (minus an edge shortfall).  So even a frozen PC can never push
        the camera past a station - or out of the wildlife arc.
        """
        t = self.target
        low, high = config.ALLOWED_PAN_RANGE
        window = config.PTZ_CINEMATIC_GLIDE_WINDOW
        # farthest one 1-s command can carry the camera: less from standstill
        # (it ramps up) than when it is already gliding
        moving = self._cine_sent is not None
        final = config.PTZ_CINEMATIC_FINAL_GLIDE if moving else config.PTZ_CINEMATIC_REST_GLIDE
        target_edge = (high - t["pan"]) if ex > 0 else (t["pan"] - low)
        shortfall = config.PTZ_CINEMATIC_EDGE_SHORTFALL if target_edge < config.PTZ_CINEMATIC_EDGE_ZONE else 0.0
        room = abs(ex) - shortfall
        speed = math.copysign(config.PTZ_CINEMATIC_PAN_SPEED, ex)

        def finish(cmd, glide_seconds=0.0):
            t["pan_coarse"] = True
            t["settle_until"] = now + glide_seconds + config.PTZ_CINEMATIC_SETTLE_SECONDS
            self._cine_speed, self._cine_sent = 0.0, None
            self.last_command = cmd

        # Safety net: while gliding the error must shrink.  If the camera moves
        # AWAY from the target (wrong pan sign, mirrored camera, bad pose) stop at
        # once and refuse to move again until restart.
        best = t.get("cine_best_err")
        if best is None or abs(ex) < best:
            t["cine_best_err"] = abs(ex)
        elif moving and abs(ex) > best + config.PTZ_WRONG_WAY_TOLERANCE:
            self.camera.stop()
            self.fault = (f"camera moved AWAY from the target (error {best:.4f} -> {abs(ex):.4f}); "
                          f"pan direction may be wrong - motion disabled")
            logger.error("PTZ SAFETY STOP: %s", self.fault)
            self.fault_at = now
            self.fault_count += 1
            finish("wrong-way-stop")
            return
        if room > final + window:
            # cruise: even if nothing else is ever sent, this cannot pass the target
            if self._cine_sent is None or now - self._cine_sent >= config.PTZ_CINEMATIC_REFRESH_SECONDS:
                sent = self.camera.velocity_move(speed, 0.0, 0.0, pose=pose, edge_margin=config.PATROL_EDGE_MARGIN,
                                                 timeout=1)
                self._cine_speed, self._cine_sent = speed, now
                self.last_command = "cine(%.2f)" % sent[0]
                if sent[0] == 0.0:
                    finish("cine blocked-by-guard")
            return
        if room >= final - window:
            # ONE last 1-s command lands within +-window of the target
            self.camera.velocity_move(speed, 0.0, 0.0, pose=pose, edge_margin=config.PATROL_EDGE_MARGIN, timeout=1)
            finish("cine-final-glide", glide_seconds=1.0)
            return
        # too close for a glide (short hop, or the loop was frozen): stop here;
        # the small precise AbsoluteMove finishes
        self.camera.stop()
        finish("cine-stop")

    # ---------------------------------------------------------- continuous
    @staticmethod
    def _speed(err, kp, vmin, vmax):
        if err is None:
            return 0.0
        v = max(vmin, min(vmax, abs(err) * kp))
        return math.copysign(v, err)

    def _step_continuous(self, pose, ex, ey, ez):
        t = self.target
        pan_v = tilt_v = zoom_v = 0.0
        if ex is not None and abs(ex) > config.PTZ_PAN_TOLERANCE * 0.6:
            pan_v = self._speed(ex, config.PTZ_PAN_KP, config.PTZ_PAN_MIN_SPEED, config.PTZ_PAN_MAX_SPEED)
            low, high = config.ALLOWED_PAN_RANGE
            # Taper the speed when approaching a hard edge of the wildlife arc.
            edge_dist = (high - pose[0]) if pan_v > 0 else (pose[0] - low)
            if edge_dist < config.PATROL_EDGE_SLOW_ZONE:
                pan_v = math.copysign(min(abs(pan_v), config.PATROL_EDGE_SLOW_SPEED), pan_v)
        if ey is not None and abs(ey) > config.PTZ_TILT_TOLERANCE:
            tilt_v = self._speed(ey, 2.0, config.PTZ_TILT_MIN_SPEED, config.PTZ_TILT_MAX_SPEED)
        if ez is not None:
            wide = t.get("zoom_is_wide")
            if (wide and ez < -config.PTZ_ZOOM_TOLERANCE * 0.5) or (not wide and abs(ez) > config.PTZ_ZOOM_TOLERANCE * 0.6):
                zoom_v = self._speed(ez, config.PTZ_ZOOM_KP, config.PTZ_ZOOM_MIN_SPEED, config.PTZ_ZOOM_MAX_SPEED)
        # Zoom first, then pan: moving the lens while panning smears the image
        # and some cameras ignore combined commands.
        if abs(zoom_v) > 0:
            pan_v = tilt_v = 0.0
        sent = self.camera.velocity_move(
            pan_v, tilt_v, zoom_v, pose=pose, edge_margin=config.PATROL_EDGE_MARGIN,
        )
        self.last_command = "vel(%.2f,%.2f,%.2f)" % tuple(sent)
        if sent == (0.0, 0.0, 0.0) and (pan_v or tilt_v or zoom_v):
            # Guards refused the move (limit reached).  Count as no progress;
            # the stall timer will release the patrol to the next station.
            self.last_command += " blocked-by-guard"
