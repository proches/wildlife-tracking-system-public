"""Stable wildlife target selection plus smooth, safe PTZ movement."""

import logging
import math
import time

import config
from detection.constants import CLASS_ANIMAL

logger = logging.getLogger("wildlife_tracker.tracking")


def plan_pulse(ex, ey, frame_width, frame_height, deadzone=None, zoom=None, slow_only=False):
    """Pan/tilt speeds and pulse length that move the picture by ~TRACK_MOVE_FRACTION
    of (ex, ey) pixels (offset of the target from the centre).

    The camera has two speed levels per axis and slows pan/tilt in proportion to
    the zoom, so each level moves the PICTURE at a fixed pixel rate (measured on the
    real camera, 19/10).  Picture moves RIGHT when the pan increases
    (TRACK_PAN_IMAGE_SIGN = -1) and UP when the tilt command is negative.
    """
    dz = config.FRAME_CENTER_DEADZONE if deadzone is None else deadzone
    dz_y = config.FRAME_CENTER_DEADZONE_Y if deadzone is None else deadzone
    need_x = config.TRACK_MOVE_FRACTION * ex if abs(ex) > dz * frame_width / 2 else 0.0
    need_y = config.TRACK_MOVE_FRACTION * ey if abs(ey) > dz_y * frame_height / 2 else 0.0
    levels_x = ((config.TRACK_PAN_FAST, config.TRACK_PAN_FAST_PX_S), (config.TRACK_PAN_SLOW, config.TRACK_PAN_SLOW_PX_S))
    levels_y = ((config.TRACK_TILT_FAST, config.TRACK_TILT_FAST_PX_S), (config.TRACK_TILT_SLOW, config.TRACK_TILT_SLOW_PX_S))

    lat_x, lat_y = config.TRACK_PAN_LATENCY, config.TRACK_TILT_LATENCY
    # measured 19/10: the same pulse moves the picture ~70 % as far at 20x as at 3.85x
    k = 1.0
    if zoom is not None:
        k = max(0.6, 1.0 - config.TRACK_RATE_DROP_PER_ZOOM * max(0.0, float(zoom) - 0.03))
    levels_x = tuple((sp, r * k) for sp, r in levels_x)
    levels_y = tuple((sp, r * k) for sp, r in levels_y)
    if slow_only:
        # 22/10 17:49: back-to-back 0.25-s pulses at the fast level moved the picture
        # ~200 px each (the camera was still rolling - no start-up delay), 2.6x the plan
        levels_x = (levels_x[1], levels_x[1])
        levels_y = (levels_y[1], levels_y[1])

    def best_time(need, levels, lat):
        if need == 0.0:
            return 0.0
        fast_t = abs(need) / levels[0][1] + lat
        if abs(need) / levels[0][1] >= 0.08:          # worth the fast level
            return fast_t
        return min(config.TRACK_MAX_PULSE, abs(need) / levels[1][1] + lat)
    pulse = min(config.TRACK_MAX_PULSE,
                max(config.TRACK_MIN_PULSE, best_time(need_x, levels_x, lat_x), best_time(need_y, levels_y, lat_y)))

    def level_for(need, levels, lat):
        if need == 0.0:
            return 0.0
        moved = max(0.0, pulse - lat)
        options = [(abs(abs(need) - rate * moved), speed) for speed, rate in levels] + [(abs(need), 0.0)]
        return math.copysign(min(options)[1], need)
    return (config.TRACK_PAN_IMAGE_SIGN * level_for(need_x, levels_x, lat_x),
            -level_for(need_y, levels_y, lat_y), pulse)


def _zoomed(box, k, frame_width, frame_height):
    """Where a box lands after the optical zoom magnifies k times about the picture's centre."""
    cx, cy = frame_width / 2.0, frame_height / 2.0
    return [cx + k * (box[0] - cx), cy + k * (box[1] - cy), cx + k * (box[2] - cx), cy + k * (box[3] - cy)]


def _near_edge(box, frame_width, frame_height, margin):
    return (box[0] < margin * frame_width or box[2] > (1 - margin) * frame_width
            or box[1] < margin * frame_height or box[3] > (1 - margin) * frame_height)


class PTZTracker:
    def __init__(self, camera):
        self.camera = camera
        from tracking.observation_patrol import ObservationPatrol
        self.observation_patrol = (
            ObservationPatrol(camera)
            if config.SMART_PATROL_ENABLED and config.ALLOWED_PAN_RANGE is not None
            else None
        )
        self.view_changed = False
        self.missed_frames = 0
        self.searching = False
        self._search_direction = 1
        self._pan_speed = 0.0
        self._tilt_speed = 0.0
        self._zoom_speed = 0.0
        self._patrol_index = 0
        self._patrol_last_move = 0.0
        self._last_seen = 0.0
        self._pending_at = 0.0
        self.pose_hint = None   # latest (pan, tilt, zoom) from the control loop
        self.focus = None       # FocusController (set by main): unlocks the lens before zooming
        self.manual_active = False   # an operator click-to-zoom owns the camera (main / ManualLook)
        self._inconclusive = 0       # site-model-only results in a row while locked
        self.group_count = 0         # animals framed together with the target
        self.hold_seconds = config.LOST_HOLD_SECONDS   # how long the last lost target's view is held
        self.visual_follow = False   # main's frame-to-frame box tracker is following the target
        self.last_selection = (None, "")   # (track id, why) - for the pipeline trace
        self.last_ptz = None         # last tracking PTZ command (for the pipeline trace)
        self.trace = None            # utils.trace.PipelineTrace, set by main
        self._flow_cmd_t = -1e9
        self.target_in_result = False      # the latest fresh AI result contained the target
        self.locked_pose = None            # camera pose of the picture locked_box comes from
        self.group_box = None              # union of the framed group (pixels of the last AI result)
        self.group_offset = (0.0, 0.0)     # group centre minus the locked animal's centre (px)
        self._group_ref_w = None           # locked box width when that offset was measured
        self._hold_until = -1e9      # after losing a target: watch this view until then
        self._centred = False        # pan hysteresis: the target was brought to the centre - hold
        self._last_ex_sign = 0
        self._last_pan_dir, self._last_pan_t = 0, -1e9

        self.locked_box = None
        self.locked_track_id = None
        self._locked_center = None
        self._velocity = (0.0, 0.0)
        self._pending_box = None
        self._pending_track_id = None
        self._pending_count = 0
        self._boundary_block_count = 0
        self.last_boundary_event = False

    @property
    def locked(self):
        return self.locked_box is not None

    def update(self, detections, frame_width, frame_height, fresh=True, now=None, conclusive=True,
               capture_pose=None, capture_time=None):
        """One control tick.

        ``fresh`` is True only when ``detections`` come from a *new* AI result.
        Stale detections are never used to steer the camera: between AI results
        the tracker just keeps the patrol running (when searching) or holds
        still (when locked, waiting for the next result).
        """
        now = time.monotonic() if now is None else now
        self._now = now
        self.view_changed = False
        self.last_boundary_event = False
        self._capture = (capture_pose, capture_time if capture_time is not None else now) if fresh else None

        if fresh:
            animals = [
                d for d in detections
                if d.get("class_id") == CLASS_ANIMAL and (
                    d.get("track_confirmed", True)
                    # locked: after the camera turned, the same animal may come back
                    # as a NEW (unconfirmed) track - near where it should be, keep it
                    or (self.locked_box is not None and config.TRACK_RELOCK_UNCONFIRMED))
            ]
            candidate = self._select_target(animals, frame_width, frame_height)
            self.target_in_result = candidate is not None
            if candidate is not None:
                self._inconclusive = 0
                self._on_target(candidate, frame_width, frame_height, now, animals)
                return
            if conclusive:
                self._inconclusive = 0
            if self.locked_box is not None and not conclusive and self._inconclusive < 4:
                self._inconclusive += 1
                # 19/10: a walking baboon was dropped because 2 of 3 results came
                # from the site model alone (which cannot see baboons).  Such an
                # empty result proves nothing: keep the lock and wait.
                self._last_seen = max(self._last_seen, now - config.LOST_TARGET_SECONDS / 2.0)
                return
            self.missed_frames += 1
            if self.locked_box is not None and self._lock_alive(now) and self._world_coast(frame_width, now):
                return
            if (self.locked_box is not None and self.missed_frames <= config.TRACK_COAST_FRAMES
                    and self._lock_alive(now) and not (config.TRACK_FEEDFORWARD and getattr(self, "_wp_last", None))):
                predicted = self._predict_box(frame_width, frame_height, self.missed_frames)
                if predicted is not None:
                    self._apply_tracking_box(predicted, frame_width, frame_height, coast=True)
                    return

        if not fresh and self.locked_box is not None and self._lock_alive(now):
            self._feedforward(frame_width, now)

        # No usable new target this tick.
        if self.manual_active and self.locked_box is None:
            return      # the operator's look holds the camera; a confirmed animal (above) ends it
        if self.locked_box is not None:
            if self._lock_alive(now):
                return  # keep the shot, wait for the next AI result
            # 26/10 18:29: a herd of zebras walked behind a near tree; after 60 s the patrol
            # took the camera away while they were still there.  A herd gets longer.
            self.hold_seconds = (config.LOST_HOLD_SECONDS_GROUP if self.group_count >= config.LOST_HOLD_GROUP_MIN
                                 else config.LOST_HOLD_SECONDS)
            logger.info("Target lost for %.0fs (%d AI results without it); holding this view %.0f s before patrol",
                        now - self._last_seen, self.missed_frames, self.hold_seconds)
            self._release_lock()
            self.camera.stop()
            # 19/10: animals went behind a bush for a moment and the patrol carried the
            # camera away.  Stay and keep looking; the same animal re-locks if it shows up.
            self._hold_until = now + self.hold_seconds
            return
        if now < self._hold_until:
            return
        elif self._pending_count and now - self._pending_at < config.LOST_TARGET_SECONDS:
            return
        self._pending_box = None
        self._pending_track_id = None
        self._pending_count = 0
        self._patrol()

    def _lock_alive(self, now):
        """Keep a locked animal until the AI has really looked for it.

        One AI pass can take longer than LOST_TARGET_SECONDS on this CPU: a pure
        timeout dropped animals BETWEEN two results and the patrol carried the
        camera away from animals it had just found.  A target is lost only after
        LOST_TARGET_MIN_RESULTS fresh results without it (hard cap
        LOST_TARGET_MAX_SECONDS, in case the AI stalls).
        """
        since = now - self._last_seen
        if since < config.LOST_TARGET_SECONDS:
            return True
        return self.missed_frames < config.LOST_TARGET_MIN_RESULTS and since < config.LOST_TARGET_MAX_SECONDS

    def _group_box(self, candidate, animals, frame_width, frame_height):
        """Union of the target and the confirmed animals close to it (a troop, a herd):
        the camera frames the group instead of one member of it."""
        self.group_count = 1
        if not config.TRACK_FRAME_GROUP or not animals:
            return list(candidate["box"])
        diag = (frame_width ** 2 + frame_height ** 2) ** 0.5
        cx, cy = self._center(candidate["box"])
        box = list(candidate["box"])
        for d in animals:
            if d is candidate:
                continue
            if not d.get("track_confirmed", True) and d.get("det_confidence", 0.0) < config.TRACK_GROUP_MIN_SCORE:
                continue
            dx, dy = self._center(d["box"])
            if ((dx - cx) ** 2 + (dy - cy) ** 2) ** 0.5 <= config.TRACK_GROUP_RADIUS * diag:
                self.group_count += 1
                b = d["box"]
                box = [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
        return box

    # ------------------------------------------------------------ feedforward
    def _px_per_pan_unit(self, zoom, frame_width):
        m = 1.0 + config.PTZ_ZOOM_MAG_PER_UNIT * max(0.0, float(zoom))
        return config.PAN_PIXELS_PER_UNIT_AT_1X * m * frame_width / 1920.0

    def _note_world_position(self, box, frame_width, now):
        """Where the animal is in PAN units (independent of where the camera points)."""
        cap = getattr(self, "_capture", None)
        if not cap or cap[0] is None:
            return
        pose, t = cap
        k = self._px_per_pan_unit(pose[2], frame_width)
        cx = (box[0] + box[2]) / 2.0
        # picture x = W/2 + (pan - animal_pan) * k   (pan up = picture moves right)
        wp = float(pose[0]) - config.TRACK_PAN_IMAGE_SIGN * -1 * (cx - frame_width / 2.0) / k
        hist = [h for h in getattr(self, "_world_hist", []) if t - h[0] <= config.TRACK_FF_WINDOW] + [(t, wp)]
        self._world_hist = hist
        self._wp_last = (t, wp)
        # speed from the two most recent positions (an animal that starts walking must be
        # followed at once - averaging over a long rest underestimated it 9x), lightly smoothed
        if len(hist) >= 2 and hist[-1][0] - hist[-2][0] >= 0.8:
            v_new = (hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0])
            if abs(v_new) >= 0.02:                                   # a glitch, not an animal
                v_new = 0.0
            self._world_vel = 0.4 * getattr(self, "_world_vel", 0.0) + 0.6 * v_new
        self._ff_last = now

    def _feedforward(self, frame_width, now):
        """Between two AI results keep turning with a walking animal (19/10: the camera
        stopped after every correction and the animal walked out of the picture)."""
        v = self._trusted_world_vel()
        if not config.TRACK_FEEDFORWARD or v == 0.0 or self.pose_hint is None or self.visual_follow:
            return
        if now - getattr(self, "_ff_last", now) < config.TRACK_FF_PERIOD:
            return
        dt = now - self._ff_last
        self._ff_last = now
        k = self._px_per_pan_unit(self.pose_hint[2], frame_width)
        wl = getattr(self, "_wp_last", None)
        if wl is not None and abs(v) * max(0.0, now - wl[0]) * k > config.TRACK_FF_MAX_LEAD * frame_width:
            # extrapolated half a picture past the last sighting without seeing the animal again:
            # stop guessing (22/10 17:49-17:50: 0.27 -> 0.53 pan on feedforward alone)
            return
        shift_px = v * dt * k                      # how far the animal moved in the picture meanwhile
        if abs(shift_px) < config.TRACK_FF_MIN_PX:
            return
        # the animal's pan grew by v*dt: in the picture it moved by -shift_px (x falls when its pan grows)
        pan, tilt, pulse = plan_pulse(-shift_px / config.TRACK_MOVE_FRACTION, 0.0, frame_width, 1080,
                                      deadzone=0.0, zoom=self.pose_hint[2], slow_only=True)
        pan = self.camera.guard_pan_speed(pan)
        if pan != 0.0:
            # NOT view_changed: these short pulses must not stop the AI from looking
            # (19/10: every pulse re-armed the "wait until still" gate - no results at all)
            self._command("feedforward", pan, 0.0, 0.0, pulse)

    def _world_coast(self, frame_width, now):
        """The followed animal is missing from a result: turn to where it should be NOW
        (last world position + walking speed) and, if it was walking or is predicted near
        the edge, zoom out one step so the next picture finds it again (19/10: a trotting
        animal left a 19x picture before its speed was known and was never seen again)."""
        wl = getattr(self, "_wp_last", None)
        if not (config.TRACK_FEEDFORWARD and wl is not None and self.pose_hint is not None):
            return False
        t_cap, wp = wl
        v = self._trusted_world_vel()
        k = self._px_per_pan_unit(self.pose_hint[2], frame_width)
        aim_x = (float(self.pose_hint[0]) - (wp + v * max(0.0, now - t_cap))) * k
        walking = abs(v) * config.TRACK_FF_RESULT_SECONDS * k > config.TRACK_FAST_MOVER_FRACTION * frame_width * 0.5
        # one missed result is often just the detector blinking: zoom out only when the animal
        # is predicted near/over the edge, or a walking animal was missed twice
        zoom = (-config.ZOOM_MIN_TRACKING_SPEED
                if (abs(aim_x) > 0.6 * frame_width / 2 or (walking and self.missed_frames >= 2)) else 0.0)
        if self.pose_hint[2] <= config.PATROL_SWEEP_ZOOM + 0.01:
            zoom = 0.0
        pan, _, pulse = plan_pulse(self._hold_x(aim_x, frame_width), 0.0, frame_width, 1080, zoom=self.pose_hint[2])
        pan = self.camera.guard_pan_speed(pan)
        if pan == 0.0 and zoom == 0.0:
            return True
        if zoom != 0.0 and self.focus is not None:
            self.focus.prepare_for_motion()
        self.view_changed = True
        self._command("coast", pan, 0.0, zoom, pulse)
        return True

    def _hold_x(self, ex, frame_width):
        """Pan hysteresis on the horizontal offset ``ex`` (px from the centre).  Centred, or
        just past the centre (an overshoot), the camera holds until the offset exceeds
        TRACK_HOLD_DEADZONE; only then does it correct again, back to FRAME_CENTER_DEADZONE."""
        e = abs(ex) / (frame_width / 2.0)
        sign = (ex > 0) - (ex < 0)
        if e > config.TRACK_HOLD_DEADZONE:
            self._centred = False
        elif e <= config.FRAME_CENTER_DEADZONE or (sign and self._last_ex_sign and sign != self._last_ex_sign):
            self._centred = True
        self._last_ex_sign = sign
        return 0.0 if self._centred else ex

    def _reversal_blocked(self, pan, err=None):
        """A pan pulse the other way within TRACK_REVERSE_HOLD_SECONDS of the last one is the
        ping-pong of 29/10 - drop it, unless the animal is really leaving the picture."""
        if pan == 0.0:
            return False
        now = getattr(self, "_now", None)
        now = time.monotonic() if now is None else now
        d = 1 if pan > 0 else -1
        if (self._last_pan_dir and d != self._last_pan_dir
                and now - self._last_pan_t < config.TRACK_REVERSE_HOLD_SECONDS
                and (err is None or abs(err) < config.TRACK_REVERSE_OVERRIDE)):
            return True
        self._last_pan_dir, self._last_pan_t = d, now
        return False

    def _command(self, source, pan, tilt, zoom, pulse, box=None, frame_width=1920, frame_height=1080, err=None):
        """Every tracking PTZ command goes through here: traced, and a camera error stops
        the motion instead of killing the control loop (it is logged, not swallowed)."""
        pan, tilt, zoom, pulse = float(pan), float(tilt), float(zoom), float(pulse)   # no numpy types to ONVIF
        if self._reversal_blocked(pan, err):
            pan = 0.0
            if tilt == 0.0 and zoom == 0.0:
                return False
        rec = {"source": source, "cmd": [round(pan, 3), round(tilt, 3), round(zoom, 3), round(pulse, 2)],
               "pose": None if self.pose_hint is None else [round(float(v), 4) for v in self.pose_hint]}
        if box is not None:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            rec["target_c"] = [round(cx), round(cy)]
            rec["err_n"] = [round((cx - frame_width / 2) / (frame_width / 2), 3),
                            round((cy - frame_height / 2) / (frame_height / 2), 3)]
        try:
            self.camera.move(pan, tilt, zoom, prevalidated=True, pulse_seconds=pulse)
            rec["ok"] = True
        except Exception as exc:          # camera timeout / ONVIF fault
            logger.warning("Tracking PTZ command failed (%s: %s) - stopping the camera", type(exc).__name__, exc)
            rec["ok"], rec["error"] = False, f"{type(exc).__name__}: {exc}"
            try:
                self.camera.stop()
            except Exception as exc2:
                logger.warning("Camera stop after a failed command also failed: %s", exc2)
        self.last_ptz = rec
        if self.trace is not None:
            self.trace.log("ptz", rec, min_interval=0.25, force=not rec["ok"])
        return rec["ok"]

    def predicted_center_x(self, frame_width, now):
        """Where the locked animal should be in the CURRENT picture (x), from its world
        position at the last AI capture plus its walking speed; None if unknown."""
        wl = getattr(self, "_wp_last", None)
        if wl is None or self.pose_hint is None:
            return None
        t_cap, wp = wl
        wp_now = wp + self._trusted_world_vel() * max(0.0, now - t_cap)
        return frame_width / 2.0 + (float(self.pose_hint[0]) - wp_now) * self._px_per_pan_unit(self.pose_hint[2], frame_width)

    def _wp_of(self, box, pose, frame_width):
        """World pan of a box centre seen at camera pose ``pose`` (as _note_world_position)."""
        k = self._px_per_pan_unit(pose[2], frame_width)
        return float(pose[0]) - config.TRACK_PAN_IMAGE_SIGN * -1 * ((box[0] + box[2]) / 2.0 - frame_width / 2.0) / k

    def _remember_other_animals(self, candidate, animals, frame_width):
        """World positions of the OTHER animals in view (recent ones only)."""
        cap = getattr(self, "_capture", None)
        if not cap or cap[0] is None:
            return
        t = cap[1]
        hist = [h for h in getattr(self, "_others_hist", []) if t - h[0] <= config.TRACK_OTHERS_MEMORY]
        hist += [(t, self._wp_of(d["box"], cap[0], frame_width)) for d in (animals or []) if d is not candidate]
        self._others_hist = hist

    def _hopped_to_another_animal(self, candidate, frame_width):
        """Is the new box another animal seen recently, rather than ours under a new tracker id?
        True when it lies closer to one of the other animals than to where ours should be."""
        cap = getattr(self, "_capture", None)
        wl = getattr(self, "_wp_last", None)
        others = [wp for t, wp in getattr(self, "_others_hist", [])
                  if cap and t >= cap[1] - config.TRACK_OTHERS_MEMORY]
        if not others or wl is None or not cap or cap[0] is None:
            return False          # nobody else around: the same animal, re-numbered
        new_wp = self._wp_of(candidate["box"], cap[0], frame_width)
        t_last, wp = wl
        expected = wp + self._trusted_world_vel() * max(0.0, cap[1] - t_last)
        return min(abs(new_wp - o) for o in others) < abs(new_wp - expected)

    def _trusted_world_vel(self):
        """The walking speed, but only once the SAME animal was seen TRACK_FF_MIN_SIGHTINGS
        times; 0 otherwise (a speed from 2 sightings - or from 2 different animals of a
        group - sent the camera running ahead of grazing waterbuck, 22/10)."""
        if getattr(self, "_same_id_sightings", 0) < config.TRACK_FF_MIN_SIGHTINGS:
            return 0.0
        return getattr(self, "_world_vel", 0.0)

    def follow_box(self, box, frame_width, frame_height, now, keepalive=True):
        """Centre a box from the frame-to-frame tracker (between AI results)."""
        if self.locked_box is None:
            return False
        box = [float(v) for v in box]     # optical flow gives numpy.float32 (21-23/10 failed moves)
        if keepalive:
            self._last_seen = max(self._last_seen, now)      # the animal is still in view
        if now - self._flow_cmd_t < config.FLOW_PTZ_INTERVAL:
            return False
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        if self.group_count >= 2 and self._group_ref_w:
            # the followed box is ONE member: centre the group instead (its offset scaled with
            # the box, which grows / shrinks with our zoom) - otherwise this pulled the camera
            # back to that one animal between AI results
            k = max(0.5, min(2.0, (box[2] - box[0]) / self._group_ref_w))
            cx, cy = cx + self.group_offset[0] * k, cy + self.group_offset[1] * k
        self._now = now
        zoom = self.pose_hint[2] if self.pose_hint is not None else None
        ex = cx - frame_width / 2.0
        pan, tilt, pulse = plan_pulse(self._hold_x(ex, frame_width), cy - frame_height / 2.0,
                                      frame_width, frame_height, zoom=zoom)
        if tilt < 0 and self.pose_hint is not None:
            from utils.privacy import in_privacy_zone
            p = self.pose_hint
            if in_privacy_zone((p[0], p[1] + 0.02, p[2])):
                tilt = 0.0
        wanted_pan = pan
        pan = self.camera.guard_pan_speed(pan) if pan else 0.0
        if (wanted_pan != 0.0 and pan == 0.0 and _near_edge(box, frame_width, frame_height, config.BOUNDARY_LEAVE_MARGIN)
                and now - getattr(self, "_edge_zoom_t", -1e9) >= 2.5 and self.pose_hint is not None
                and self.pose_hint[2] > config.PATROL_SWEEP_ZOOM + 0.005):
            # at the end of the arc and the animal reaches the picture's edge: the camera
            # cannot turn after it - widen the view between AI results (22/10 sim: it walked out
            # of an 18x picture while the camera held at the arc limit).  26/10: not just
            # because the GROUP's centre is off-centre - that zoomed out to 6x on resting zebras
            self._edge_zoom_t = now
            self._zoom_step(-1)
        if pan == 0.0 and tilt == 0.0:
            return False                        # inside the deadband (or at the arc limit)
        self._flow_cmd_t = now
        # NOT view_changed: these short pulses must not stop the AI from looking
        return self._command("flow", pan, tilt, 0.0, pulse, box, frame_width, frame_height,
                             err=ex / (frame_width / 2.0))

    def _on_target(self, candidate, frame_width, frame_height, now, animals=None):
        was_searching = self.searching
        self.searching = False
        self.missed_frames = 0
        self._last_seen = now
        # Only reset the patrol route when leaving a search - not on every brief lock.
        if was_searching and self.observation_patrol is not None:
            self.camera.stop()
            self.observation_patrol.reset()
        tid = candidate.get("track_id")
        prev_tid = getattr(self, "_motion_track_id", None)
        if prev_tid is None or (tid != prev_tid and self._hopped_to_another_animal(candidate, frame_width)):
            # a DIFFERENT animal (another group member): the old one's positions say nothing
            # about this one's speed.  22/10 17:38-17:50: the lock hopped between grazing
            # waterbuck, each hop (up to 856 px) read as running - the camera swung back and
            # forth and once ran from pan 0.27 to 0.53 with no animal in view.  (A new tracker
            # id for the SAME animal - a walker re-numbered 8 times in 44 results - keeps its
            # speed: resetting on every id change lost every walking animal.)
            self._locked_center, self._velocity = None, (0.0, 0.0)
            self._world_hist, self._world_vel, self._wp_last = [], 0.0, None
            self._same_id_sightings = 0
        self._motion_track_id = tid
        self._same_id_sightings = getattr(self, "_same_id_sightings", 0) + 1
        self._remember_other_animals(candidate, animals, frame_width)
        self._update_lock_motion(candidate["box"], frame_width, frame_height)
        before = getattr(self, "group_count", 0)
        group = self._group_box(candidate, animals, frame_width, frame_height)
        if self.group_count >= 2 and self.group_count != before:
            logger.info("Following a group of %d animals", self.group_count)
        # Position and speed come from the LOCKED animal (a stable identity).  21/10 13:40: the
        # group's member count flickered 2-5 between results, its centre jumped by up to
        # ~200 px, read as running animals -> the camera zoomed OUT of a resting group.
        self._note_world_position(candidate["box"], frame_width, now)
        self.locked_box = list(candidate["box"])
        self.locked_track_id = candidate.get("track_id")
        cap = getattr(self, "_capture", None)
        self.locked_pose = cap[0] if cap else self.pose_hint      # the camera pose of that picture
        lcx, lcy = self._center(candidate["box"])
        gcx, gcy = self._center(group)
        self.group_box = list(group)
        self.group_offset = (gcx - lcx, gcy - lcy) if self.group_count >= 2 else (0.0, 0.0)
        self._group_ref_w = max(1.0, candidate["box"][2] - candidate["box"][0])
        aim_x = None
        wl = getattr(self, "_wp_last", None)
        if config.TRACK_FEEDFORWARD and wl is not None and self.pose_hint is not None and getattr(self, "_capture", None):
            # 19/10: centring on the 3-s-old picture while the feedforward had already turned
            # the camera made the two corrections fight.  Where is the animal NOW, compared
            # with where the camera points NOW?  (picture x = W/2 + (pan - animal_pan) * k)
            t_cap, wp = wl
            wp_now = wp + self._trusted_world_vel() * max(0.0, now - t_cap)
            if self.group_count >= 2 and self._capture[0] is not None:
                # aim at the GROUP's centre: it lies group_offset px beside the locked animal
                k_cap = self._px_per_pan_unit(self._capture[0][2], frame_width)
                wp_now -= self.group_offset[0] / k_cap
            aim_x = (float(self.pose_hint[0]) - wp_now) * self._px_per_pan_unit(self.pose_hint[2], frame_width)
        self._apply_tracking_box(group, frame_width, frame_height, coast=False, aim_x=aim_x)

    def _select_target(self, animals, frame_width, frame_height):
        threshold = config.TRACKING_CONFIDENCE_THRESHOLD
        if self.locked_box is not None:
            # 20/10: a vervet monkey the operator clicked scored 0.21 at 48x and was never
            # locked (< 0.22).  An animal we already follow may come back weak - keep it.
            threshold = min(threshold, config.LOCKED_MIN_SCORE)
        eligible = [d for d in animals if d.get("det_confidence", 0.0) >= threshold or d.get("operator_target")]

        # Persistent track IDs are the strongest identity signal available.
        if self.locked_track_id is not None:
            same_id = [d for d in eligible if d.get("track_id") == self.locked_track_id]
            if same_id:
                self._pending_box = None
                self._pending_track_id = None
                self._pending_count = 0
                self.last_selection = (same_id[0].get("track_id"), "same track id as the locked target")
                return same_id[0]

        if self.locked_box is not None:
            if self._locked_center is None:
                self._locked_center = self._center(self.locked_box)
            diagonal = (frame_width ** 2 + frame_height ** 2) ** 0.5
            predicted = self._predicted_center(frame_width, frame_height, 1)
            max_dist = config.TARGET_MAX_PREDICTED_RADIUS * diagonal
            best = None
            best_score = None
            for d in eligible:
                cx, cy = self._center(d["box"])
                dist_now = self._distance((cx, cy), self._locked_center)
                dist_pred = self._distance((cx, cy), predicted)
                iou = self._iou(d["box"], self.locked_box)
                if iou < config.TARGET_MATCH_IOU and min(dist_now, dist_pred) > max_dist:
                    continue
                priority = float(d.get("priority_score", d.get("det_confidence", 0.0)))
                score = (min(dist_now, dist_pred) / diagonal) - (0.12 * iou) - (0.06 * priority)
                if best_score is None or score < best_score:
                    best, best_score = d, score
            if best is not None:
                self._pending_box = None
                self._pending_track_id = None
                self._pending_count = 0
                self.last_selection = (best.get("track_id"), "nearest box to the locked target's predicted position")
                return best
            # 19/10: the followed buffalo went behind a bush while another one stood in
            # plain view - switch to it now instead of waiting for the lock to time out
            others = [d for d in eligible if d.get("track_confirmed", True)]
            if others:
                new = max(others, key=lambda d: (float(d.get("priority_score", d.get("det_confidence", 0.0))),
                                                 self._box_area(d["box"])))
                logger.info("Target not visible; switching to another animal in view (track %s)", new.get("track_id"))
                self._locked_center = self._center(new["box"])
                self._velocity = (0.0, 0.0)
                self.locked_track_id = new.get("track_id")
                self.last_selection = (new.get("track_id"), "locked target not visible - switched to another animal")
                return new
            return None

        if not eligible:
            self._pending_box = None
            self._pending_track_id = None
            self._pending_count = 0
            return None

        best = max(
            eligible,
            key=lambda d: (float(d.get("priority_score", d.get("det_confidence", 0.0))),
                           d.get("det_confidence", 0.0), self._box_area(d["box"])),
        )
        close = self._pending_box is not None and self._is_close(
            best["box"], self._pending_box, frame_width, frame_height
        ) and (self._pending_track_id is None or self._pending_track_id == best.get("track_id"))
        if close:
            self._pending_count += 1
        else:
            self._pending_box = list(best["box"])
            self._pending_track_id = best.get("track_id")
            self._pending_count = 1
        self._pending_at = time.monotonic()

        if self._pending_count >= config.NEW_TARGET_CONFIRM_FRAMES:
            self._pending_box = None
            self._pending_track_id = None
            self._pending_count = 0
            self._locked_center = self._center(best["box"])
            self._velocity = (0.0, 0.0)
            self.last_selection = (best.get("track_id"), "new target: highest priority confirmed animal")
            return best
        return None

    def _update_lock_motion(self, box, frame_width, frame_height):
        center = self._center(box)
        zoom = self.pose_hint[2] if self.pose_hint is not None else None
        if zoom is not None and abs(zoom - getattr(self, "_motion_zoom", zoom)) > 0.005:
            # the zoom changed since the last result: the box "jumped" because of OUR zoom,
            # not because the animal ran - start the speed estimate again
            self._velocity = (0.0, 0.0)
            self._locked_center = None
        self._motion_zoom = zoom
        if self._locked_center is not None:
            raw_vx = center[0] - self._locked_center[0]
            raw_vy = center[1] - self._locked_center[1]
            alpha = config.TARGET_VELOCITY_SMOOTHING
            self._velocity = (
                alpha * self._velocity[0] + (1 - alpha) * raw_vx,
                alpha * self._velocity[1] + (1 - alpha) * raw_vy,
            )
        self._locked_center = center

    def _apply_tracking_box(self, box, frame_width, frame_height, coast=False, aim_x=None):
        x1, y1, x2, y2 = box
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        box_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1, frame_width * frame_height)

        dx = (box_cx - frame_width / 2) / (frame_width / 2)
        dy = (box_cy - frame_height / 2) / (frame_height / 2)
        if coast:
            decay = config.TRACK_COAST_DECAY ** max(0, self.missed_frames - 1)
            dx *= decay
            dy *= decay

        # ---- pan / tilt: a pulse sized in PIXELS (19/10 rewrite) -------------
        # The camera has two speed levels per axis and slows its pan/tilt in
        # proportion to the zoom, so a level moves the PICTURE at a fixed pixel rate
        # (measured on the real camera).  Aim a little ahead of a walking animal
        # and move ~80 % of the way: no overshoot, no creeping.
        vx, vy = (0.0, 0.0) if coast else self._velocity
        if aim_x is None:
            aim_x = (box_cx + config.TRACK_LEAD * vx) - frame_width / 2
            aim_y = (box_cy + config.TRACK_LEAD * vy) - frame_height / 2
        else:
            aim_y = box_cy - frame_height / 2      # world-based pan aim: no image-velocity lead
        aim_err = aim_x / (frame_width / 2)
        if not coast:
            aim_x = self._hold_x(aim_x, frame_width)
        target_pan, target_tilt, pulse = plan_pulse(aim_x, aim_y, frame_width, frame_height,
                                                    zoom=self.pose_hint[2] if self.pose_hint else None)

        # ---- zoom: keep the animal about TRACK_TARGET_PX big -------------------
        size = max(x2 - x1, y2 - y1)
        speed_px = (vx ** 2 + vy ** 2) ** 0.5
        if self.pose_hint is not None:          # world speed: the image speed is small while we follow
            speed_px = max(speed_px, abs(getattr(self, "_world_vel", 0.0)) * config.TRACK_FF_RESULT_SECONDS
                           * self._px_per_pan_unit(self.pose_hint[2], frame_width))
        zoom_now = self.pose_hint[2] if self.pose_hint is not None else 0.0
        target_zoom = 0.0
        if zoom_now > config.TRACK_MAX_ZOOM + 0.02:
            target_zoom = -config.ZOOM_MIN_TRACKING_SPEED      # beyond the sharp limit: back to it
        elif not coast:
            fast_mover = speed_px > config.TRACK_FAST_MOVER_FRACTION * frame_width
            near_edge = abs(dx) > 0.8 or abs(dy) > 0.8
            centred = max(abs(dx), abs(dy)) <= config.ZOOM_CENTER_TOLERANCE
            if self.group_count >= 2:
                # a GROUP: close enough that it fills a good part of the picture, all members
                # inside (21/10 13:45: 5 animals in 22 % x 13 % of the frame, 53-89 px each)
                fill = max((x2 - x1) / frame_width, (y2 - y1) / frame_height)
                touching = (x1 < 0.03 * frame_width or x2 > 0.97 * frame_width
                            or y1 < 0.03 * frame_height or y2 > 0.97 * frame_height)
                if fill > config.TRACK_GROUP_FILL_MAX or ((fast_mover or touching) and fill > 0.45):
                    target_zoom = -config.ZOOM_MIN_TRACKING_SPEED
                elif (fill < config.TRACK_GROUP_FILL_MIN and centred and not fast_mover and not touching
                      and zoom_now < config.TRACK_MAX_ZOOM):
                    target_zoom = config.ZOOM_MIN_TRACKING_SPEED
            # zoom OUT only when really needed (19/10: it zoomed out of a buffalo herd and
            # back in, again and again): about to leave the picture, or far too big
            elif size > config.TRACK_TARGET_PX * 2.6 or ((fast_mover or near_edge) and size > config.TRACK_TARGET_PX * 0.6):
                target_zoom = -config.ZOOM_MIN_TRACKING_SPEED
            elif (size < config.TRACK_TARGET_PX * 0.7 and centred and not fast_mover
                  and zoom_now < config.TRACK_MAX_ZOOM):
                target_zoom = config.ZOOM_MIN_TRACKING_SPEED

        if target_tilt < 0 and self.pose_hint is not None:
            # never tilt DOWN into the lodge pool / guest area (utils/privacy.py)
            from utils.privacy import in_privacy_zone
            p = self.pose_hint
            if in_privacy_zone((p[0], p[1] + 0.02, p[2])):
                target_tilt = 0.0

        allowed_pan = self.camera.guard_pan_speed(target_pan)
        if target_pan != 0.0 and allowed_pan == 0.0 and self._is_pan_boundary_blocked(target_pan):
            # The animal stands at (or beyond) the end of the wildlife arc.  22/10 17:45-17:48:
            # waterbuck at the right end - the lock was released 4 times "scanning inward" and
            # the camera kept leaving them.  Stay at the edge and keep watching: tilt and zoom
            # still follow, the normal loss rules end the lock when it is really gone.
            self._boundary_block_count += 1
            if self._boundary_block_count == 1:
                logger.info("Animal at the end of the allowed arc: holding there and watching it")
            self.last_boundary_event = True
            allowed_pan = 0.0
            target_zoom = self._boundary_zoom(box, frame_width, frame_height, zoom_now, coast)
        else:
            self._boundary_block_count = 0
        self._pan_speed, self._tilt_speed, self._zoom_speed = allowed_pan, target_tilt, target_zoom
        if allowed_pan == 0.0 and target_tilt == 0.0 and target_zoom == 0.0:
            return
        self.view_changed = True
        if target_zoom != 0.0 and self.focus is not None:
            self.focus.prepare_for_motion()
        if allowed_pan != 0.0 or target_tilt != 0.0:
            self._command("ai", allowed_pan, target_tilt, 0.0, pulse, box, frame_width, frame_height, err=aim_err)
        if target_zoom != 0.0:
            self._zoom_step(1 if target_zoom > 0 else -1)

    def _boundary_zoom(self, box, frame_width, frame_height, zoom_now, coast):
        """Zoom while the camera is held at the end of the arc and cannot turn to the animals.

        The optical zoom works about the picture's centre, so here it alone decides what
        stays in view.  26/10 17:19-17:25: zebras drinking left of the water-hole (beyond the
        arc end): "group centre off-centre -> zoom out" and "group small and centred -> zoom
        in" took turns, and the camera drifted from 47x out to 6x.  Now: out only when the
        followed animal itself is about to leave the picture; in only when, one step closer,
        it - and the whole group, if that still fits - stays well inside."""
        own = list(box) if coast or self.locked_box is None else list(self.locked_box)
        # where it will be by the next result or so (the camera cannot turn, so the picture
        # speed IS the animal's speed): a walking animal must not leave before we widen
        vx, vy = self._velocity
        if self.pose_hint is not None:
            # the image speed builds up only slowly after the camera stopped turning with the
            # animal; its world speed is known already (picture x = W/2 + (pan - animal_pan) * k)
            wvx = (-self._trusted_world_vel() * self._px_per_pan_unit(self.pose_hint[2], frame_width)
                   * config.TRACK_FF_RESULT_SECONDS)
            if abs(wvx) > abs(vx):
                vx = wvx
        lead = config.BOUNDARY_LEAD_RESULTS
        ahead = [own[0] + lead * vx, own[1] + lead * vy, own[2] + lead * vx, own[3] + lead * vy]
        if (_near_edge(own, frame_width, frame_height, config.BOUNDARY_LEAVE_MARGIN)
                or _near_edge(ahead, frame_width, frame_height, config.BOUNDARY_LEAVE_MARGIN)):
            return -config.ZOOM_MIN_TRACKING_SPEED if zoom_now > config.PATROL_SWEEP_ZOOM + 0.005 else 0.0
        if coast or zoom_now >= config.TRACK_MAX_ZOOM - 0.005:
            return 0.0
        own = [min(own[0], ahead[0]), min(own[1], ahead[1]), max(own[2], ahead[2]), max(own[3], ahead[3])]
        k = config.TRACK_ZOOM_STEP
        margin = config.BOUNDARY_ZOOM_IN_MARGIN
        if self.group_count >= 2 and not _near_edge(_zoomed(box, k, frame_width, frame_height),
                                                    frame_width, frame_height, margin):
            fill = max((box[2] - box[0]) / frame_width, (box[3] - box[1]) / frame_height)
            return config.ZOOM_MIN_TRACKING_SPEED if fill * k <= config.TRACK_GROUP_FILL_TARGET else 0.0
        if _near_edge(_zoomed(own, k, frame_width, frame_height), frame_width, frame_height, margin):
            return 0.0
        size = max(own[2] - own[0], own[3] - own[1])
        return config.ZOOM_MIN_TRACKING_SPEED if size * k <= config.TRACK_TARGET_PX * 1.3 else 0.0

    def _zoom_step(self, direction):
        """Tracking zoom as ONE bounded AbsoluteMove: magnification x / TRACK_ZOOM_STEP, never past
        TRACK_MAX_ZOOM.  21/10 13:52: the 0.4-s continuous zoom pulse coasted from 26x to 42x
        (the lens keeps going after Stop) - one animal filled the picture, the group was cut off,
        and no rule ever came back below the limit (above it, this step goes straight back)."""
        now = getattr(self, "_now", None)
        now = time.monotonic() if now is None else now
        if direction > 0 and now - getattr(self, "_zoom_out_t", -1e9) < config.TRACK_ZOOM_IN_AFTER_OUT_SECONDS:
            # 26/10 18:15: out, in 10 s later, out again - the zebras grazed at the picture's
            # edge.  Widening stays allowed at once (it keeps the animal in view); coming
            # closer again waits, so the picture does not pump.
            return False
        pose = self.pose_hint if self.pose_hint is not None else self.camera.get_pose()
        if pose is None:
            return False                  # unknown zoom: no blind step
        z = max(0.0, float(pose[2]))
        per = config.PTZ_ZOOM_MAG_PER_UNIT
        m, cap = 1.0 + per * z, 1.0 + per * config.TRACK_MAX_ZOOM
        if direction > 0:
            target_m = min(cap, m * config.TRACK_ZOOM_STEP)
        else:
            target_m = cap if m > cap * 1.02 else max(1.0 + per * config.ZOOM_POSITION_MIN, m / config.TRACK_ZOOM_STEP)
        target = (target_m - 1.0) / per
        if abs(target - z) < 0.003:
            return False
        if direction < 0:
            self._zoom_out_t = now
        rec = {"source": "zoom", "cmd": [0.0, 0.0, float(direction), round(target, 3)],
               "pose": [round(float(v), 4) for v in pose]}
        try:
            rec["ok"] = bool(self.camera.zoom_only(target))
        except Exception as exc:          # camera timeout / ONVIF fault
            logger.warning("Tracking zoom step failed (%s: %s)", type(exc).__name__, exc)
            rec["ok"], rec["error"] = False, f"{type(exc).__name__}: {exc}"
        self.last_ptz = rec
        if self.trace is not None:
            self.trace.log("ptz", rec, force=True)
        return rec["ok"]

    def _is_pan_boundary_blocked(self, requested_speed):
        try:
            status = self.camera.get_status()
            pan = float(status.PanTilt.x)
            low, high, margin = self.camera.pan_limits(getattr(getattr(status, "Zoom", None), "x", None))
            return (requested_speed > 0 and pan >= high - margin) or (requested_speed < 0 and pan <= low + margin)
        except (AttributeError, TypeError, ValueError):
            return False

    def _predict_box(self, frame_width, frame_height, steps):
        if self.locked_box is None or self._locked_center is None:
            return None
        cx, cy = self._predicted_center(frame_width, frame_height, steps)
        x1, y1, x2, y2 = self.locked_box
        bw, bh = x2 - x1, y2 - y1
        cx = max(-bw, min(frame_width + bw, cx))
        cy = max(-bh, min(frame_height + bh, cy))
        return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]

    def _predicted_center(self, frame_width, frame_height, steps):
        if self._locked_center is None:
            return frame_width / 2, frame_height / 2
        vx, vy = self._velocity
        scale = min(1.0, config.TRACK_COAST_DECAY ** max(0, steps - 1))
        return self._locked_center[0] + vx * steps * scale, self._locked_center[1] + vy * steps * scale

    def holding(self, now):
        """Just lost a target: the camera stays on this view (AI keeps looking)."""
        return self.locked_box is None and now < self._hold_until

    def release_for_operator(self):
        """An operator click takes the camera: drop any lock, stop moving."""
        self._release_lock()
        self._hold_until = -1e9
        self._pan_speed = self._tilt_speed = self._zoom_speed = 0.0
        self.camera.stop()

    def _release_lock(self):
        self._world_hist, self._world_vel, self._wp_last = [], 0.0, None
        self.locked_box = None
        self.locked_track_id = None
        self._locked_center = None
        self._velocity = (0.0, 0.0)
        self._pending_box = None
        self._pending_track_id = None
        self._pending_count = 0
        self._boundary_block_count = 0
        self.group_count, self.group_box, self.group_offset, self._group_ref_w = 0, None, (0.0, 0.0), None
        self._motion_track_id, self._same_id_sightings, self.locked_pose = None, 0, None
        self._others_hist = []
        self._centred, self._last_ex_sign = False, 0
        self._last_pan_dir, self._last_pan_t = 0, -1e9

    def _stop_motion(self):
        self._pan_speed = self._tilt_speed = self._zoom_speed = 0.0
        self.camera.stop()

    def _patrol_or_sweep_now(self):
        if self.observation_patrol is not None:
            self._pan_speed = self._tilt_speed = self._zoom_speed = 0.0
            self.observation_patrol.reset()
            self.observation_patrol.tick(pose=self.pose_hint)
        else:
            self._sweep_search()

    def _apply_exclusion_guard(self, target_pan):
        target_pan = self.camera.guard_pan_speed(target_pan)
        if not config.EXCLUDED_PAN_RANGES or target_pan == 0.0:
            return target_pan
        status = self.camera.get_status()
        if status is None:
            return 0.0
        current_pan = status.PanTilt.x
        for low, high in config.EXCLUDED_PAN_RANGES:
            if low <= current_pan <= high:
                logger.warning("Camera is inside a no-go zone (pan=%.2f) - blocking pan", current_pan)
                return 0.0
        return target_pan

    def _ease_toward(self, target_pan, target_tilt, target_zoom, pulse=None):
        self._pan_speed += self._clip(target_pan - self._pan_speed, config.MAX_PAN_TILT_ACCEL)
        self._tilt_speed += self._clip(target_tilt - self._tilt_speed, config.MAX_PAN_TILT_ACCEL)
        if self._zoom_speed * target_zoom < 0 or target_zoom == 0.0:
            # 19/10: leftover zoom "momentum" was lifted back to the minimum zoom speed
            # by _lift() below - the camera kept zooming past every limit (41x, blur)
            self._zoom_speed = 0.0
        self._zoom_speed += self._clip(target_zoom - self._zoom_speed, config.MAX_ZOOM_ACCEL)
        if target_zoom != 0.0 and self._zoom_speed * target_zoom > 0:
            self._zoom_speed = (1 if target_zoom > 0 else -1) * max(
                abs(self._zoom_speed), min(abs(target_zoom), config.ZOOM_MIN_TRACKING_SPEED)
            )
        # The camera ignores small speeds (dead-zone): lift any real request to
        # the lowest speed it executes, otherwise fine corrections never happen.
        if self._tilt_speed < 0 and self.pose_hint is not None:
            # never tilt DOWN into the lodge pool / guest area (utils/privacy.py)
            from utils.privacy import in_privacy_zone
            p = self.pose_hint
            if in_privacy_zone((p[0], p[1] + 0.02, p[2])):
                self._tilt_speed = 0.0
        self._pan_speed = self._lift(self._pan_speed, config.PTZ_PAN_MIN_SPEED)
        self._tilt_speed = self._lift(self._tilt_speed, config.PTZ_TILT_MIN_SPEED)
        self._zoom_speed = self._lift(self._zoom_speed, config.PTZ_ZOOM_MIN_SPEED)
        status = self.camera.get_status()
        self._pan_speed = self.camera.guard_pan_speed(self._pan_speed, status=status)
        self._zoom_speed = self.camera.guard_zoom_speed(self._zoom_speed, status=status)
        if abs(self._pan_speed) < 1e-3 and abs(self._tilt_speed) < 1e-3 and abs(self._zoom_speed) < 1e-3:
            self.camera.stop()
        else:
            self.view_changed = True
            if abs(self._zoom_speed) >= 1e-3 and self.focus is not None:
                self.focus.prepare_for_motion()
            self.camera.move(self._pan_speed, self._tilt_speed, self._zoom_speed, prevalidated=True,
                             pulse_seconds=pulse)

    def _patrol(self):
        if self.observation_patrol is not None:
            self.searching = True
            self._pan_speed = self._tilt_speed = self._zoom_speed = 0.0
            self.view_changed = self.observation_patrol.tick(pose=self.pose_hint)
            return
        if not config.PATROL_PRESET_NAMES:
            self._sweep_search()
            return
        now = time.time()
        if not self.searching:
            logger.info("Target lost - patrolling saved waypoints")
            self.searching = True
            self._patrol_last_move = 0.0
        if now - self._patrol_last_move >= config.PATROL_DWELL_SECONDS:
            name = config.PATROL_PRESET_NAMES[self._patrol_index % len(config.PATROL_PRESET_NAMES)]
            if self.camera.goto_preset_by_name(name):
                logger.info("Patrolling to waypoint '%s'", name)
            self._patrol_index += 1
            self._patrol_last_move = now

    def _sweep_search(self):
        if not self.searching:
            logger.info("Target lost - starting left/right search sweep")
            self.searching = True
        intended = config.SEARCH_PAN_SPEED * self._search_direction
        allowed = self._apply_exclusion_guard(intended)
        if allowed == 0.0 and intended != 0.0:
            self.reverse_search_direction()
            intended = config.SEARCH_PAN_SPEED * self._search_direction
            allowed = self._apply_exclusion_guard(intended)
        target_zoom = 0.0
        status = self.camera.get_status()
        zoom = getattr(getattr(status, "Zoom", None), "x", None)
        if zoom is not None and zoom > config.ZOOM_SEARCH_POSITION + config.ZOOM_POSITION_MARGIN:
            target_zoom = -config.ZOOM_MIN_TRACKING_SPEED
        if target_zoom == 0.0:
            self._zoom_speed = 0.0
        # Use a bit more aggressive ease so the fallback sweep actually moves.
        self._ease_toward(allowed, 0.0, target_zoom)

    def reverse_search_direction(self):
        self._search_direction *= -1

    @staticmethod
    def _center(box):
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    @staticmethod
    def _distance(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _box_area(box):
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    @staticmethod
    def _iou(box_a, box_b):
        ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = PTZTracker._box_area(box_a) + PTZTracker._box_area(box_b) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _is_close(box_a, box_b, frame_width, frame_height):
        diagonal = (frame_width ** 2 + frame_height ** 2) ** 0.5
        return PTZTracker._distance(PTZTracker._center(box_a), PTZTracker._center(box_b)) <= 0.15 * diagonal

    @staticmethod
    def _clip(value, limit):
        return max(-limit, min(limit, value))

    @staticmethod
    def _lift(value, floor):
        if abs(value) < 1e-3:
            return 0.0
        return value if abs(value) >= floor else (floor if value > 0 else -floor)
