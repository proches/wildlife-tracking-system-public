"""Closed-loop patrol across the wildlife arc, with a water-hole priority.

v3 behaviour (all verified against the *measured* camera pose):

    widen -> travel -> settle_wide -> observe_wide -> zoom -> settle_zoom -> observe_zoom -> next station

* Every movement goes through :class:`tracking.motion.MotionController`, which
  verifies that the camera really moved and falls back from AbsoluteMove to
  closed-loop ContinuousMove when the camera ignores absolute commands.
* ``tick()`` is called from the 8 Hz control thread and never sleeps.
* The AI is only allowed to look while the camera is *still and settled*
  (``detection_allowed``).  The patrol will not leave a station before the
  detector produced at least one result on a settled frame (``note_result``).
* The water-hole station is visited every ``WATER_HOLE_EVERY_N_STATIONS`` stops.
"""

import logging
import math
import time

import config
from tracking.motion import ARRIVED, STALLED, MotionController
from utils.privacy import path_crosses_privacy

logger = logging.getLogger("wildlife_tracker.tracking")

OBSERVE_PHASES = ("observe_wide", "observe_zoom")


class ObservationPatrol:
    def __init__(self, camera, clock=time.monotonic, focus=None):
        self.camera = camera
        self.clock = clock
        self.focus = focus
        self.motion = MotionController(camera, clock=clock)
        arc_low, arc_high = config.ALLOWED_PAN_RANGE
        inset = config.PAN_BOUNDARY_MARGIN + config.SMART_PATROL_POSITION_TOLERANCE
        low, high = arc_low + inset, arc_high - inset
        count = max(2, config.SMART_PATROL_STATIONS)
        self.stations = [
            dict(
                name=f"scan_{i + 1}",
                pan=high - i * (high - low) / (count - 1),
                tilt=config.SMART_PATROL_TILT,
                zoom=config.SMART_PATROL_OBSERVE_ZOOM,
                dwell=config.SMART_PATROL_ZOOM_SECONDS,
            )
            for i in range(count)
        ]
        self.water_name = None
        water = config.WATER_HOLE_POSITION
        if water is not None:
            wpan, wtilt, wzoom = float(water["pan"]), float(water["tilt"]), float(water["zoom"])
            if not (
                arc_low <= wpan <= arc_high
                and -1.0 <= wtilt <= 1.0
                and config.ZOOM_SEARCH_POSITION <= wzoom <= config.ZOOM_POSITION_MAX
            ):
                raise ValueError(
                    f"Water-hole position must be inside safe pan/tilt/zoom limits "
                    f"(pan={wpan:.6f} arc=[{arc_low:.6f},{arc_high:.6f}] "
                    f"zoom={wzoom:.3f} max={config.ZOOM_POSITION_MAX})"
                )
            self.stations.append(dict(name="water_hole", pan=wpan, tilt=wtilt, zoom=wzoom,
                                      dwell=config.WATER_HOLE_DWELL_SECONDS))
            self.water_name = "water_hole"
            self.stations = [s for s in self.stations
                             if s["name"] == "water_hole" or abs(s["pan"] - wpan) >= config.SMART_PATROL_WATER_EXCLUSION]
            logger.info("Water-hole station enabled: pan=%.4f tilt=%.2f zoom=%.2f dwell=%.0fs",
                        wpan, wtilt, wzoom, config.WATER_HOLE_DWELL_SECONDS)
        self.presets = self._load_presets()
        # Extra preset-only stations (19/10: "baboon_spot", the near ground by the
        # lodge wall where baboons rest - the plain stations look far above it).
        for name, wide_dwell in getattr(config, "PATROL_EXTRA_STATIONS", {}).items():
            pr = self.presets.get(name)
            if pr and arc_low <= float(pr["pan"]) <= arc_high:
                tilt = float(pr.get("tilt", -1.0))
                self.stations.append(dict(name=name, pan=float(pr["pan"]), tilt=tilt,
                                          zoom=float(pr["zoom"]), dwell=config.SMART_PATROL_ZOOM_SECONDS,
                                          wide_dwell=float(wide_dwell), low=tilt > -0.98,
                                          min_results=(config.PATROL_LOW_MIN_RESULTS if tilt > -0.98
                                                       else config.PATROL_MIN_RESULTS_WIDE)))
                logger.info("Extra station %s: pan=%.4f, watched %.0f s", name, float(pr["pan"]), float(wide_dwell))
        for st in self.stations:
            pr = self.presets.get(st["name"])
            if pr:
                st["pan"] = float(pr["pan"])
                if st["name"] == self.water_name:
                    st["zoom"] = float(pr["zoom"])
        if self.presets:
            logger.info("Station presets in use: %s", sorted(self.presets))
        self._zoom_goal = None
        self._expected_zoom = None
        self._close_done = False
        self._glide = None            # slow zoom in progress (see _zoom_to)
        self._finish_preset = None    # station whose preset finishes the current zoom
        self._last_zoom = None
        self.stations.sort(key=lambda p: p["pan"], reverse=True)
        self.route = self._build_route()
        self.route_pos = 0
        self.activity_scores = {s["name"]: 0.0 for s in self.stations}
        self._activity_last_decay = self.clock()
        self.observe_epoch = 0
        self.results_in_epoch = 0
        self.candidates_pending = 0   # unconfirmed tracks the AI is still verifying (set by main)
        self.candidate_best = 0.0     # best box score among them (set by main)
        self.travel_candidate_best = 0.0   # best score among those worth stopping the sweep for (main)
        self._verify_sites = []       # (pan, time) of recent sweep pauses
        self.results_in_verify = 0
        self.motion_cue = None        # (time, (cx, cy, w, h)) from main's MotionCue
        self.active = False
        self._stall_streak = 0
        self.reset()

    # ----------------------------------------------------------- route
    def _build_route(self):
        """Ping-pong over the arc, returning to the water-hole every N stops.

        With a LOW row of stations (19/10: baboons rest on the near ground below
        the plain views): out along the plain row, back along the low row -
        water-hole -> plain stations -> far end -> low stations -> water-hole.
        """
        low = [i for i, s in enumerate(self.stations) if s.get("low")]
        if low and config.PATROL_TWO_ROWS:
            water = [i for i, s in enumerate(self.stations) if s["name"] == self.water_name]
            high = [i for i, s in enumerate(self.stations) if not s.get("low") and s["name"] != self.water_name]
            high.sort(key=lambda i: self.stations[i]["pan"], reverse=True)   # away from the water-hole
            low.sort(key=lambda i: self.stations[i]["pan"])                  # back towards it
            return water + high + low
        n = len(self.stations)
        order = list(range(n)) + list(range(n - 2, 0, -1))  # 0..n-1..1
        water = next((i for i, s in enumerate(self.stations) if s["name"] == self.water_name), None)
        every = max(0, int(config.WATER_HOLE_EVERY_N_STATIONS))
        if water is None or every == 0:
            return order
        route, since = [], 0
        for idx in order:
            route.append(idx)
            since = 0 if idx == water else since + 1
            if since >= every and idx != water:
                route.append(water)
                since = 0
        # collapse accidental repeats (water, water)
        out = [route[0]]
        for idx in route[1:]:
            if idx != out[-1]:
                out.append(idx)
        return out

    @property
    def index(self):
        return self.route[self.route_pos % len(self.route)]

    @property
    def station(self):
        return self.stations[self.index]

    # -------------------------------------------------------- AI gating
    @property
    def detection_allowed(self):
        if self.focus is not None and self.focus.busy:
            return False
        sweeping = (config.PATROL_DETECT_WHILE_SWEEPING and self.phase == "travel"
                    and abs(config.PTZ_CINEMATIC_PAN_SPEED) < 0.223)   # slow glide only: no blur
        return self.phase in OBSERVE_PHASES or sweeping or self.phase == "verify"

    @property
    def moving(self):
        return self.phase not in OBSERVE_PHASES and self.phase != "verify"

    def note_result(self, epoch):
        """The detector finished a frame that was captured in ``epoch``."""
        if epoch == self.observe_epoch and self.phase in OBSERVE_PHASES:
            self.results_in_epoch += 1
        if self.phase == "verify":
            self.results_in_verify += 1

    def status_text(self):
        return f"{self.station['name']}:{self.phase}"

    # -------------------------------------------------------- activity
    def record_sighting(self, pan, confidence=1.0):
        try:
            pan = float(pan)
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            return
        nearest = min(self.stations, key=lambda s: abs(s["pan"] - pan))
        self.activity_scores[nearest["name"]] = min(
            50.0, self.activity_scores.get(nearest["name"], 0.0) + confidence)

    def _decay_activity(self, now):
        if now - self._activity_last_decay < 60.0:
            return
        minutes = int((now - self._activity_last_decay) // 60.0)
        factor = config.SMART_PATROL_ACTIVITY_DECAY ** max(1, minutes)
        self.activity_scores = {k: v * factor for k, v in self.activity_scores.items()}
        self._activity_last_decay = now

    # ----------------------------------------------------------- state
    def reset(self):
        self.phase = "bootstrap"
        self.since = self.clock()
        self.active = False
        self.results_in_epoch = 0
        self.motion.target = None
        self._glide = None
        self._finish_preset = None

    def _phase(self, name):
        prev = getattr(self, "phase", None)
        self.phase, self.since = name, self.clock()
        if name in OBSERVE_PHASES:
            self.observe_epoch += 1
            self.results_in_epoch = 0
        if prev != name:
            st = self.station
            logger.info("Patrol %s -> %s @ %s (target pan=%.4f zoom=%.3f)",
                        prev, name, st["name"], st["pan"], st["zoom"])

    @staticmethod
    def _load_presets():
        """ptz_presets.json from tools/build_ptz_presets.py: {station: {token, pan, zoom}}."""
        import json
        import os
        path = getattr(config, "PTZ_PRESETS_FILE", None)
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {k: v for k, v in data.items() if isinstance(v, dict) and v.get("token") and "pan" in v}
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("Preset file %s unreadable (%s: %s) - patrolling with absolute moves only",
                           path, type(exc).__name__, exc)
            return {}

    def goto_water_now(self):
        """Operator shortcut (W key): leave the route and go to the water-hole."""
        water = next((i for i, s in enumerate(self.stations) if s["name"] == self.water_name), None)
        if water is None or water not in self.route:
            return False
        self.route_pos = self.route.index(water)
        self.active = True
        self._begin_widen()
        return True

    def near_station(self, pose, pan_tol=0.012, zoom_tol=0.02):
        """Is the camera at a patrol station's view (vs. aimed somewhere by hand)?"""
        try:
            pan, tilt, zoom = float(pose[0]), float(pose[1]), float(pose[2])
        except (TypeError, ValueError, IndexError):
            return True
        if tilt > -0.98:     # plain stations read the -1.0 clamp; a readable tilt = aimed by hand...
            return any(abs(pan - st["pan"]) <= pan_tol and abs(tilt - st.get("tilt", -1.0)) <= 0.03
                       for st in self.stations)   # ...unless it is a low preset station
        for st in self.stations:
            if abs(pan - st["pan"]) <= pan_tol and min(abs(zoom - self._wide_target()), abs(zoom - st["zoom"])) <= zoom_tol:
                return True
        return False

    # ------------------------------------------------------------ tilt glide
    def _tilt_target(self, st):
        """Tilt reading a station is seen at: its preset tilt (low row) or the -1.0 clamp."""
        if st["name"] == self.water_name or not st.get("low"):
            return -1.0
        return float(st.get("tilt", -1.0))

    def _start_tilt_glide(self, st, tilt):
        """19/10: going down to the low row (and back up), GotoPreset swung the tilt
        in under a second.  Glide there at constant speed first; the preset then
        only fixes the last bit."""
        if not (config.PATROL_TILT_GLIDE and self.presets and hasattr(self.camera, "velocity_move")):
            return False
        target = self._tilt_target(st)
        err = target - float(tilt)
        hidden = config.STATION_TILT_UP_SECONDS.get(st["name"], 0.0) if target <= -0.999 else 0.0
        if target <= -0.999 and tilt <= -0.995 and hidden <= 0.0:   # at the clamp: the preset does the rest
            return False
        if abs(err) <= config.PATROL_TILT_GLIDE_MIN:
            return False
        # above the -1.0 clamp the station's preset was raised for `hidden` seconds at
        # the fast level (tools/build_ptz_presets.py): glide that part too
        self._tilt = dict(target=target, last=-1e9, hidden=hidden, hidden_until=None)
        logger.info("Tilt glide %.3f -> %.3f before %s", tilt, target, st["name"])
        self._phase("tilt")
        return True

    def _tilt_glide_step(self, tilt, now, elapsed):
        """One tick of the tilt glide; True when done."""
        t = self._tilt
        err = t["target"] - float(tilt)
        fast = abs(err) > config.PATROL_TILT_SLOW_BAND
        lead = config.PATROL_TILT_FAST_LEAD if fast else config.PATROL_TILT_SLOW_LEAD
        at_clamp = t["target"] <= -0.999 and tilt <= -0.995
        if at_clamp:
            if t["hidden_until"] is None:
                t["hidden_until"] = now + t["hidden"]
            fast, err = True, -1.0          # keep raising through the part the camera cannot report
        reached = (abs(err) <= lead and not at_clamp) or (at_clamp and now >= t["hidden_until"])
        if reached or elapsed > config.PATROL_TILT_GLIDE_MAX_SECONDS:
            self.camera.stop()
            return True
        if now - t["last"] >= 0.5:
            speed = config.PATROL_TILT_FAST_SPEED if fast else config.PATROL_TILT_SLOW_SPEED
            # a positive tilt command raises the camera (the reading falls)
            self.camera.velocity_move(0.0, speed if err < 0 else -speed, 0.0, timeout=1)
            t["last"] = now
        return False

    def _start_prezoom(self, st, zoom):
        """19/10: travel at the travel zoom (the camera pans faster when wider), then
        glide to the station preset's zoom before the preset - no zoom jump."""
        pr = self._water_wide() if st["name"] == self.water_name else self._preset(st)
        if not pr or not hasattr(self.camera, "zoom_only"):
            return False
        target = float(pr.get("zoom", zoom))
        if abs(target - float(zoom)) <= 0.008:
            return False
        self._zoom_to(target)
        self._phase("prezoom")
        return True

    def _arrive(self, st, pose):
        """Final framing at a station (presets), then settle."""
        if self._preset(st) and st["name"] != self.water_name:
            self._goto_station_preset(st, pose=pose)   # exact framing incl. the raised tilt
            self._zoom_goal = None
        elif st["name"] == self.water_name and self._water_wide():
            ww = self._water_wide()
            # the water-hole's own pan/tilt at the sweep zoom: the zoom-in
            # that follows is a slow glide straight onto the water
            if self.camera.goto_preset_safe(ww["token"], ww["pan"], pose=pose):
                self.motion.target = None
            self._zoom_goal = None
        self._phase("settle_wide")

    def _water_wide(self):
        """Preset with the water-hole's pan/tilt at the sweep zoom (tools/center_water_preset.py)."""
        return self.presets.get("water_wide") if hasattr(self.camera, "goto_preset_safe") else None

    def _preset(self, st):
        return self.presets.get(st["name"]) if hasattr(self.camera, "goto_preset_safe") else None

    def _wide_target(self):
        if self.presets:
            return config.PATROL_SWEEP_ZOOM
        return config.ZOOM_SEARCH_POSITION + config.ZOOM_POSITION_MARGIN

    @staticmethod
    def _mag(zoom):
        return 1.0 + config.PTZ_ZOOM_MAG_PER_UNIT * max(0.0, float(zoom))

    def _zoom_to(self, zoom):
        """Zoom without touching pan/tilt (presets mode) or via the motion controller.

        Presets mode glides: small AbsoluteMove zoom steps at a constant
        magnification rate (PTZ_ZOOM_GLIDE_RATE).  Measured on this camera: a
        single zoom command ignores any speed (3.85x -> 21.8x in ~3 s) and
        continuous zoom-in only has one, accelerating, speed.
        """
        if self.presets and hasattr(self.camera, "zoom_only"):
            self.motion.target = None
            self._zoom_goal = zoom
            self._glide = None
            z0 = self._last_zoom
            if config.PTZ_ZOOM_GLIDE and z0 is not None and abs(zoom - z0) > 0.01:
                duration = abs(math.log(self._mag(zoom) / self._mag(z0))) / max(0.05, config.PTZ_ZOOM_GLIDE_RATE)
                self._glide = dict(z0=float(z0), t0=self.clock(), last=-1e9, duration=duration, final=False)
                self._glide_step()
            else:
                self.camera.zoom_only(zoom)
        else:
            self._zoom_goal = None
            self.motion.set_target(zoom=zoom, zoom_is_wide=zoom <= self._wide_target() + 1e-9)

    def _goto_station_preset(self, st, pose=None):
        pr = self._preset(st)
        if pr and self.camera.goto_preset_safe(pr["token"], pr["pan"], pose=pose):
            self.motion.target = None
            self._glide = None
            self._zoom_goal = float(pr["zoom"])
            return True
        return False

    def _glide_step(self):
        """Send the next zoom step of the glide; True once the last step was sent."""
        g = self._glide
        if g is None or g["final"]:
            return True
        now = self.clock()
        frac = min(1.0, (now - g["t0"]) / max(1e-3, g["duration"]))
        if frac < 1.0 and now - g["last"] < config.PTZ_ZOOM_GLIDE_STEP_SECONDS:
            return False
        m0, m1 = self._mag(g["z0"]), self._mag(self._zoom_goal)
        m = m0 * (m1 / m0) ** frac
        self.camera.zoom_only(self._zoom_goal if frac >= 1.0 else (m - 1.0) / config.PTZ_ZOOM_MAG_PER_UNIT)
        g["last"] = now
        g["final"] = frac >= 1.0
        return g["final"]

    def _zoom_reached(self, zoom, elapsed):
        if self._zoom_goal is None:
            return None
        limit = 12.0
        if self._glide is not None:
            if not self._glide_step():
                return False
            limit = self._glide["duration"] + 8.0
        return abs(zoom - self._zoom_goal) <= 0.01 or elapsed >= limit

    def _next(self):
        self.route_pos = (self.route_pos + 1) % len(self.route)
        logger.info("Patrol advancing to %s", self.station["name"])
        self._begin_widen()

    def _prepare_focus(self):
        if self.focus is not None:
            self.focus.prepare_for_motion()

    def _begin_widen(self):
        self._prepare_focus()
        self._close_done = False
        self._finish_preset = None
        self._expected_zoom = None
        self._zoom_to(self._wide_target())
        self._phase("widen")

    def _begin_direct(self):
        """Already looking near this station: go straight to its full pose.

        Without this the patrol always zoomed OUT first - so a camera the user
        had just aimed at the water-hole zoomed out and wandered off.
        """
        st = self.station
        self._prepare_focus()
        self._close_done = False
        if self._goto_station_preset(st):
            self._expected_zoom = self._zoom_goal
            self._phase("zoom")
            return
        self._zoom_goal = None
        self.motion.set_target(pan=st["pan"], tilt=st["tilt"],
                               zoom=min(st["zoom"], config.ZOOM_POSITION_MAX - config.ZOOM_POSITION_MARGIN))
        self._phase("zoom")

    def _begin_travel(self):
        self._hold_ref = None
        st = self.station
        tilt = getattr(self, "_last_tilt", -1.0)
        pan = getattr(self, "_last_pan", None)
        if (pan is not None and tilt > -0.995 and self.presets
                and path_crosses_privacy(pan, st["pan"], tilt)):
            # 19/10: gliding back from the low row at its low tilt passed over the
            # pool and the guests' loungers - raise the camera FIRST
            self._tilt = dict(target=-1.0, last=-1e9, hidden=0.0, hidden_until=None)
            logger.info("Raising the camera before passing the lodge pool (privacy)")
            self._phase("lift")
            return
        self._zoom_goal = None
        if self._preset(st):
            # pan-only glide keeps the raised tilt; the preset fixes the rest
            self.motion.set_target(pan=st["pan"], tilt=None)
            self.motion.target["no_abs"] = True
        else:
            self.motion.set_target(pan=st["pan"], tilt=st["tilt"])
        self._phase("travel")

    def _read_ptz(self):
        pose = self.camera.get_pose() if hasattr(self.camera, "get_pose") else None
        if pose is None:
            status = self.camera.get_status()
            try:
                pose = (float(status.PanTilt.x), float(status.PanTilt.y), float(status.Zoom.x))
            except (AttributeError, TypeError, ValueError):
                return None
        return self._validate(pose)

    @staticmethod
    def _validate(pose):
        pan, tilt, zoom = pose
        low, high = config.ALLOWED_PAN_RANGE
        if not all(math.isfinite(v) for v in pose):
            return None
        if not (low - config.PAN_RECOVERY_TOLERANCE <= pan <= high + config.PAN_RECOVERY_TOLERANCE
                and -1 <= tilt <= 1 and 0 <= zoom <= 1):
            return None
        return pose

    def _drifted(self, st, pan, zoom):
        """Re-aim if the camera moved by itself while observing (park/idle action)."""
        zoomed = self.phase in ("settle_zoom", "observe_zoom")
        want_zoom = None
        if zoomed:
            want_zoom = self._expected_zoom or min(st["zoom"], config.ZOOM_POSITION_MAX - config.ZOOM_POSITION_MARGIN)
        ref = getattr(self, "_hold_ref", None)
        ref = st["pan"] if ref is None else ref
        pan_off = abs(pan - ref) > config.PATROL_HOLD_PAN_TOLERANCE
        zoom_off = want_zoom is not None and abs(zoom - want_zoom) > config.PATROL_HOLD_ZOOM_TOLERANCE
        if not (pan_off or zoom_off):
            return False
        logger.warning("Camera left %s while observing (pan %.4f zoom %.3f) - moved by itself? re-aiming",
                       st["name"], pan, zoom)
        if zoomed:
            self._prepare_focus()
            self._close_done = False
            if self._goto_station_preset(st, pose=(pan, -1.0, zoom)):
                self._expected_zoom = self._zoom_goal
            else:
                self._zoom_goal = None
                self.motion.set_target(pan=st["pan"], tilt=st["tilt"], zoom=want_zoom)
            self._phase("zoom")
        else:
            self._begin_travel()
        return True

    def _motion_recent(self, now, within=1.0):
        return self.motion_cue is not None and now - self.motion_cue[0] <= within

    def _should_verify(self, pan, now):
        # only boxes main.worth_a_closer_look() accepts stop the sweep: 21/10, of 35 stops for
        # an AI box, the generic ones ("unidentified", bare "bird", any at night) were acacias,
        # tree trunks and dark canopy - 1 animal one could follow in 27
        ai = self.travel_candidate_best >= config.PATROL_PAUSE_MIN_SCORE
        if not (config.PATROL_PAUSE_ON_CANDIDATE and (ai or self._motion_recent(now))):
            return False
        if now - self.since < 1.0:          # just (re)started gliding
            return False
        # never pause twice for the same spot (a far tree that keeps scoring 0.3)
        self._verify_sites = [(p, t) for p, t in self._verify_sites if now - t < config.PATROL_PAUSE_REPEAT_SECONDS]
        return all(abs(pan - p) > config.PATROL_PAUSE_REPEAT_PAN for p, _ in self._verify_sites)

    def _recover_into_arc(self, pose, now):
        """Camera slightly outside the arc (moved by hand, old overshoot): bring it
        back with one precise AbsoluteMove.  Far outside -> do nothing (safety)."""
        try:
            pan, tilt, zoom = [float(v) for v in pose]
        except (TypeError, ValueError):
            return False
        low, high = config.ALLOWED_PAN_RANGE
        # (27/10: zoomed in, the tracker may have followed an animal up to PAN_HIGH_ZOOMED_MAX)
        reach_high = max(high + config.PAN_ARC_RECOVERY_MAX, config.PAN_HIGH_ZOOMED_MAX + config.PAN_RECOVERY_TOLERANCE)
        if not (math.isfinite(pan) and low - config.PAN_ARC_RECOVERY_MAX <= pan <= reach_high):
            return False
        if now - getattr(self, "_last_recovery", -1e9) < 4.0:
            return True
        self._last_recovery = now
        inside = min(high - 0.005, max(low + 0.005, pan))
        logger.warning("Camera at pan %.4f is just outside the wildlife arc - moving it back to %.4f", pan, inside)
        self.camera.goto_position(inside, max(-1.0, min(1.0, tilt)), zoom=None)
        self.phase = "bootstrap"
        return True

    def _bootstrap(self, pan, tilt=-1.0):
        # same pan on both rows: the readable tilt says which row the camera is on
        nearest = min(range(len(self.stations)), key=lambda i: abs(self.stations[i]["pan"] - pan)
                      + 0.5 * abs(self.stations[i].get("tilt", -1.0) - (tilt if tilt > -0.98 else -1.0)))
        self.route_pos = self.route.index(nearest)
        logger.info("Patrol bootstrap: pan=%.4f nearest=%s; route=%s", pan,
                    self.stations[nearest]["name"], [self.stations[i]["name"] for i in self.route])
        direct_ok = config.PATROL_ZOOM_EVERY_STATION or self.stations[nearest]["name"] == self.water_name
        if direct_ok and abs(self.stations[nearest]["pan"] - pan) <= config.PATROL_DIRECT_START_PAN:
            self._begin_direct()
        else:
            self._begin_widen()

    # ------------------------------------------------------------ tick
    suspended_by = None     # set by main: another owner (tracking / acquiring / recovery) has the camera

    def tick(self, pose=None):
        """Advance the patrol one control tick; True if the view may be changing."""
        if self.suspended_by:
            # single owner of the PTZ: while a target is being acquired / tracked /
            # recovered the patrol sends nothing (and resumes from where it is later)
            if not getattr(self, "_suspend_logged", False):
                logger.info("Patrol suspended: %s owns the camera", self.suspended_by)
                self._suspend_logged = True
            return False
        self._suspend_logged = False
        now = self.clock()
        self._decay_activity(now)
        if not self.active:
            self.since = now
            self.active = True
            logger.info("Observation patrol ACTIVE: %d stations, route length %d", len(self.stations), len(self.route))

        raw = pose
        pose = self._validate(pose) if pose else self._read_ptz()
        if pose is None:
            self.camera.stop()
            if raw is not None and self._recover_into_arc(raw, now):
                return True
            logger.warning("Patrol: PTZ status unavailable; retrying")
            return False
        pan, tilt, zoom = pose
        self._last_zoom = zoom
        self._last_pan, self._last_tilt = pan, tilt

        if self.phase == "bootstrap":
            self._bootstrap(pan, tilt)
            return True

        st = self.station
        elapsed = now - self.since

        if self.phase in ("settle_wide", "observe_wide", "settle_zoom", "observe_zoom") and self._drifted(st, pan, zoom):
            return True

        if self.phase == "widen":
            reached = self._zoom_reached(zoom, elapsed)
            if reached is not None:
                if reached:
                    self._begin_travel()
                return True
            if zoom <= self._wide_target() + config.PTZ_ZOOM_TOLERANCE:
                self.camera.stop()
                self._begin_travel()
                return True
            res = self.motion.step(pose)
            if res == STALLED:
                logger.warning("Patrol zoom-out stalled at zoom=%.3f; continuing", zoom)
                self._begin_travel()
            elif res == ARRIVED:
                self._begin_travel()
            return True

        if self.phase == "travel" and self._should_verify(pan, now):
            # Something showed up while sweeping: stop and look properly (a still
            # camera gives the tracker several clean frames to confirm it).
            self.camera.stop()
            self.motion.target = None
            self._verify_sites.append((pan, now))
            self._verify_ai = self.travel_candidate_best >= config.PATROL_PAUSE_MIN_SCORE
            logger.info("Sweep paused at pan %.4f: checking a possible animal (%s)", pan,
                        f"AI score {self.travel_candidate_best:.2f}" if self._verify_ai else "something MOVING")
            self.results_in_verify = 0
            self._phase("verify")
            return True

        if self.phase == "verify":
            ai_pause = getattr(self, "_verify_ai", True)
            # a pause for movement alone ends after one clean AI look (dusk noise / wind
            # paused the sweep every half minute, 19/10); an AI candidate gets the full check
            min_s = config.PATROL_VERIFY_MIN_SECONDS if ai_pause else config.PATROL_VERIFY_MOTION_SECONDS
            min_r = config.PATROL_VERIFY_MIN_RESULTS if ai_pause else 1
            if elapsed >= config.PATROL_VERIFY_SECONDS or (
                    elapsed >= min_s and self.travel_candidate_best < config.PATROL_PAUSE_MIN_SCORE
                    and (not self._motion_recent(now, 3.0)
                         or elapsed >= min_s + config.MOTION_CUE_MAX_EXTRA)
                    and self.results_in_verify >= min_r):
                self._begin_travel()
                return True
            return False

        if self.phase == "travel":
            res = self.motion.step(pose)
            if res == ARRIVED:
                self._stall_streak = 0
                logger.info("Patrol arrived %s: pan=%.4f tilt=%.3f zoom=%.3f", st["name"], pan, tilt, zoom)
                if self._start_tilt_glide(st, tilt) or self._start_prezoom(st, zoom):
                    return True
                self._arrive(st, pose)
            elif res == STALLED:
                self._stall_streak += 1
                dx = st["pan"] - pan
                if abs(dx) <= max(config.PTZ_PAN_TOLERANCE * 4, 0.05):
                    logger.warning("Patrol close to %s (dx=%.4f); observing from here", st["name"], dx)
                    self._hold_ref = pan     # 19/10: else "drifted" re-aimed, stalled, ... in a tight loop
                    self._phase("settle_wide")
                else:
                    logger.warning("Patrol could not reach %s (dx=%.4f, streak=%d); skipping",
                                   st["name"], dx, self._stall_streak)
                    self._next()
            return True

        if self.phase == "tilt":
            if self._tilt_glide_step(tilt, now, elapsed) and not self._start_prezoom(st, zoom):
                self._arrive(st, pose)
            return True

        if self.phase == "prezoom":
            reached = self._zoom_reached(zoom, elapsed)
            if reached is None or reached:
                self._zoom_goal, self._glide = None, None
                self._arrive(st, pose)
            return True

        if self.phase == "lift":
            if self._tilt_glide_step(tilt, now, elapsed):
                self._last_tilt = -1.0
                self._begin_travel()
            return True

        if self.phase == "settle_wide":
            settle = config.SETTLE_AFTER_PRESET_SECONDS if self._preset(st) else config.SETTLE_AFTER_PAN_SECONDS
            if elapsed >= settle:
                self._phase("observe_wide")
            return False

        if self.phase == "observe_wide":
            hot = min(1.0, self.activity_scores.get(st["name"], 0.0) / 5.0)
            dwell = st.get("wide_dwell", config.SMART_PATROL_WIDE_SECONDS) * (
                1.0 + hot * (config.SMART_PATROL_HOTSPOT_DWELL_MULTIPLIER - 1.0))
            done = elapsed >= dwell and self.results_in_epoch >= st.get("min_results", config.PATROL_MIN_RESULTS_WIDE)
            if ((self.candidates_pending or (self._motion_recent(now, 3.0) and elapsed < dwell + config.MOTION_CUE_MAX_EXTRA))
                    and elapsed < config.PATROL_OBSERVE_MAX_WAIT):
                done = False   # something might be there: keep looking until it is confirmed or fades
            if done or elapsed >= config.PATROL_OBSERVE_MAX_WAIT:
                # Zoom with a reason: the water-hole, something the AI is still
                # verifying, or a place animals were seen recently.  Zooming into
                # every bush only wasted time and focus.
                worth_zoom = (config.PATROL_ZOOM_EVERY_STATION or st["name"] == self.water_name
                              or self.candidates_pending > 0 or hot >= config.PATROL_ZOOM_HOTSPOT_MIN)
                if worth_zoom and st["zoom"] > zoom + config.PTZ_ZOOM_TOLERANCE:
                    self._prepare_focus()
                    self._close_done = False
                    if st["name"] == self.water_name and self._preset(st) and self._water_wide():
                        # glide in slowly, then the preset fixes the last pixel
                        self._zoom_to(st["zoom"])
                        self._finish_preset = st
                    elif not (st["name"] == self.water_name and self._goto_station_preset(st, pose=pose)):
                        self._zoom_to(min(st["zoom"], config.ZOOM_POSITION_MAX - config.ZOOM_POSITION_MARGIN))
                    self._expected_zoom = self._zoom_goal
                    self._phase("zoom")
                    return True
                self._next()
                return True
            return False

        if self.phase == "zoom":
            reached = self._zoom_reached(zoom, elapsed)
            if reached is not None:
                if reached:
                    logger.info("Patrol zoomed on %s: zoom=%.3f", st["name"], zoom)
                    if self._finish_preset is not None:
                        self._goto_station_preset(self._finish_preset, pose=pose)
                        self._finish_preset = None
                    self._glide = None
                    self._expected_zoom = self._zoom_goal
                    self._zoom_goal = None
                    self._phase("settle_zoom")
                    if self.focus is not None and config.FOCUS_REFOCUS_AFTER_ZOOM:
                        self.focus.request("zoom")
                return True
            res = self.motion.step(pose)
            if res in (ARRIVED, STALLED):
                if res == STALLED:
                    logger.warning("Patrol zoom-in stalled at %s (zoom=%.3f); observing here", st["name"], zoom)
                else:
                    logger.info("Patrol zoomed on %s: zoom=%.3f", st["name"], zoom)
                self._phase("settle_zoom")
                if self.focus is not None and config.FOCUS_REFOCUS_AFTER_ZOOM:
                    self.focus.request("zoom")
            return True

        if self.phase == "settle_zoom":
            if elapsed >= config.SETTLE_AFTER_ZOOM_SECONDS and not (self.focus is not None and self.focus.busy):
                self._phase("observe_zoom")
            elif elapsed >= config.SETTLE_AFTER_ZOOM_SECONDS + config.FOCUS_HUNT_MAX_SECONDS:
                self._phase("observe_zoom")
            return False

        if self.phase == "observe_zoom":
            hot = min(1.0, self.activity_scores.get(st["name"], 0.0) / 5.0)
            dwell = st["dwell"] * (1.0 + hot * (config.SMART_PATROL_HOTSPOT_DWELL_MULTIPLIER - 1.0))
            if (st["name"] == self.water_name and self.presets and not self._close_done
                    and elapsed >= dwell / 2.0 and config.WATER_HOLE_CLOSE_ZOOM > zoom + 0.02):
                # second half of the water-hole watch: closer, for small birds
                # (the water is centred in the preset, so it stays centred)
                self._close_done = True
                self._prepare_focus()
                self._zoom_to(config.WATER_HOLE_CLOSE_ZOOM)
                self._phase("zoom")
                return True
            if self._close_done and st["name"] == self.water_name:
                dwell = dwell / 2.0           # the first half was spent at the preset zoom
            done = elapsed >= dwell and self.results_in_epoch >= config.PATROL_MIN_RESULTS_ZOOM
            if self.candidates_pending and elapsed < config.PATROL_OBSERVE_MAX_WAIT:
                done = False
            if done or elapsed >= max(dwell, config.PATROL_OBSERVE_MAX_WAIT):
                self._next()
                return True
            return False

        # Unknown phase: recover.
        logger.error("Patrol in unknown phase %r; restarting", self.phase)
        self.reset()
        return False
