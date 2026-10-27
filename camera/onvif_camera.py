"""
ONVIF camera control, built on onvif-zeep-async.

Why this library and not plain onvif-zeep: the onvif-zeep wheel has a
known packaging defect where its bundled WSDL files are sometimes simply
missing from the install (confirmed on this deployment). onvif-zeep-async
is the actively maintained fork and explicitly guarantees its WSDL files
are bundled correctly.

That library's internals (aiohttp connectors) require an actively
RUNNING event loop even to construct the ONVIFCamera object itself, not
just to call its methods. So everything - including construction - is
done inside one async function, driven by a single persistent event loop
this class owns. Everything else in this project (main.py, ptz_tracker.py,
tools/) keeps calling this class's plain synchronous methods
and does not need to know any of this happened.

Requires: pip install onvif-zeep-async httpx
(uninstall onvif-zeep first if it's still installed - both packages use
the same "onvif" import name and must not coexist.)
"""

import asyncio
import logging
import math
import threading
import time
from datetime import timedelta

import config
from utils.privacy import min_zoom_for_pan, pan_high_for_zoom


logger = logging.getLogger("wildlife_tracker.camera")


def SimpleNamespace_pose(pose):
    """Wrap an already-read (pan, tilt, zoom) tuple in the status shape the guards expect."""
    from types import SimpleNamespace
    pan, tilt, zoom = pose
    return SimpleNamespace(PanTilt=SimpleNamespace(x=pan, y=tilt), Zoom=SimpleNamespace(x=zoom))


def _is_mirror(pan, ref):
    """True if ``pan`` looks like the sign-flipped GetStatus glitch of ``ref``.

    The sign changed while the magnitude stayed similar.  The magnitude
    tolerance is wide because the camera may have travelled a lot since
    ``ref`` was read (AbsoluteMove is fast).  A genuine move across pan 0 is
    not affected: there the two readings are close to each other.
    """
    return (pan < 0) != (ref < 0) and abs(abs(pan) - abs(ref)) <= 0.30 and abs(pan - ref) >= 0.16


class OnvifPTZCamera:
    """Thin sync wrapper around the async onvif-zeep-async client.

    Coordinate convention (v3.1): every pan value this class accepts or returns
    is in the camera's *reported* (GetStatus) frame - the frame ALLOWED_PAN_RANGE
    and WATER_HOLE_POSITION were measured in.  Cameras mounted upside-down with
    EFlip=ON (this SUNBA unit) execute pan commands mirrored; ``pan_command_sign``
    converts at the very last moment, when a request is sent.
    """

    pan_command_sign = 1
    _last_good_pan = None
    _mirror_since = None
    _mirror_value = None

    def __init__(self, ip, port, username, password):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        # v3: several threads (control loop, focus task) talk to the camera;
        # an asyncio loop must never be driven by two threads at once.
        self._run_lock = threading.RLock()
        self._last_status_warning = 0.0
        # Construction itself must happen while the loop is running, so it
        # is done inside _async_init rather than here directly.
        self._run(self._async_init(ip, port, username, password))

    async def _async_init(self, ip, port, username, password):
        from onvif import ONVIFCamera  # imported lazily so tests run without the library
        self.cam = ONVIFCamera(ip, port, username, password)
        await self.cam.update_xaddrs()

        self.media = await self.cam.create_media_service()
        self.ptz = await self.cam.create_ptz_service()

        profiles = await self.media.GetProfiles()
        if not profiles:
            raise RuntimeError("Camera returned no media profiles")
        self.profile = profiles[0]
        self.profile_token = self.profile.token

        config_options = await self.ptz.GetConfigurationOptions(
            {"ConfigurationToken": self.profile.PTZConfiguration.token}
        )
        spaces = config_options.Spaces
        self._pan_tilt_space = spaces.ContinuousPanTiltVelocitySpace[0].URI
        self._zoom_space = spaces.ContinuousZoomVelocitySpace[0].URI
        # Absolute spaces enable reliable station-to-station patrol (pan left/center/right).
        try:
            self._abs_pan_tilt_space = spaces.AbsolutePanTiltPositionSpace[0].URI
        except (AttributeError, IndexError, TypeError):
            self._abs_pan_tilt_space = None
        try:
            self._abs_zoom_space = spaces.AbsoluteZoomPositionSpace[0].URI
        except (AttributeError, IndexError, TypeError):
            self._abs_zoom_space = None
        try:
            self._rel_pan_tilt_space = spaces.RelativePanTiltTranslationSpace[0].URI
        except (AttributeError, IndexError, TypeError):
            self._rel_pan_tilt_space = None
        # Cameras with an empty PanTiltSpeedSpace (this SUNBA) must not be sent a
        # PanTilt speed in AbsoluteMove.
        self._pt_speed_supported = bool(getattr(spaces, "PanTiltSpeedSpace", None))
        self._zoom_speed_supported = bool(getattr(spaces, "ZoomSpeedSpace", None))
        try:
            modes = self._read_flip_modes(await self.ptz.GetConfiguration(
                {"PTZConfigurationToken": self.profile.PTZConfiguration.token}))
        except Exception as exc:  # camera-specific, never fatal
            logger.info("PTZ GetConfiguration unavailable (%s); assuming no flip", exc)
            modes = {}
        self.pan_command_sign = self._resolve_pan_sign(modes)
        # Imaging service (focus).  Optional: many cameras do not expose it.
        self._imaging = None
        self._video_source_token = None
        try:
            self._video_source_token = self.profile.VideoSourceConfiguration.SourceToken
            self._imaging = await self.cam.create_imaging_service()
        except Exception as exc:  # camera-specific, never fatal
            logger.info("Imaging (focus) service unavailable: %s", exc)
            self._imaging = None
        logger.info(
            "PTZ spaces: continuous OK; absolute_pan_tilt=%s absolute_zoom=%s relative=%s; "
            "pan_command_sign=%+d",
            bool(self._abs_pan_tilt_space),
            bool(self._abs_zoom_space),
            bool(self._rel_pan_tilt_space),
            self.pan_command_sign,
        )

    @staticmethod
    def _read_flip_modes(ptz_config):
        """Return {'EFlip': 'ON', 'Reverse': 'OFF', ...} from PTControlDirection."""
        modes = {}
        ext = getattr(ptz_config, "Extension", None)
        elements = list(getattr(ext, "_value_1", None) or [])
        direction = getattr(ext, "PTControlDirection", None)
        for name in ("EFlip", "Reverse"):
            mode = getattr(getattr(direction, name, None), "Mode", None)
            if mode:
                modes[name] = str(mode).upper()
        for el in elements:   # vendor XML the WSDL did not map
            try:
                for child in el.iter():
                    tag = str(child.tag).split("}")[-1]
                    if tag in ("EFlip", "Reverse"):
                        mode = child.findtext("{*}Mode") or ""
                        if mode:
                            modes[tag] = mode.strip().upper()
            except AttributeError:
                continue
        return modes

    @staticmethod
    def _resolve_pan_sign(modes):
        """Explicit config > the camera's own EFlip/Reverse setting > value measured by ptz_selftest.

        The camera's live setting wins over the file: on 17/10/2026 a corrupted
        file (+1) sent the camera to the hotel side while the camera itself said
        EFlip=ON (-1).
        """
        forced = str(getattr(config, "PTZ_PAN_COMMAND_SIGN", "auto")).strip().lower()
        if forced in ("1", "+1", "-1"):
            return int(forced)
        measured = None
        try:
            import json
            with open(config.PTZ_CAPABILITIES_FILE, "r", encoding="utf-8") as f:
                measured = json.load(f).get("pan_command_sign")
        except FileNotFoundError:
            pass
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("Measured pan sign unreadable in %s (%s); relying on the camera's own setting",
                           config.PTZ_CAPABILITIES_FILE, exc)
        if modes.get("EFlip") or modes.get("Reverse"):
            flipped = (modes.get("EFlip") == "ON") != (modes.get("Reverse") == "ON")
            sign = -1 if flipped else 1
            if measured in (1, -1) and measured != sign:
                logger.error("Pan sign in %s (%+d) contradicts the camera's PTControlDirection %s (%+d); "
                             "using the camera's setting. Re-run tools/ptz_selftest.py.",
                             config.PTZ_CAPABILITIES_FILE, measured, modes, sign)
            logger.info("PTControlDirection %s -> pan commands %s", modes, "MIRRORED" if flipped else "normal")
            return sign
        if measured in (1, -1):
            logger.info("Pan command sign %+d (measured by ptz_selftest; camera reports no direction setting)",
                        measured)
            return int(measured)
        logger.info("No PTControlDirection and no measurement: assuming normal pan commands")
        return 1

    def _run(self, coro):
        """Run one async call to completion on our persistent event loop (thread-safe)."""
        lock = self.__dict__.setdefault("_run_lock", threading.RLock())
        with lock:
            # 19/10 17:02: one ONVIF reply never came and the whole control loop froze
            # (camera kept "locked" on a buffalo it never approached).  Never wait forever.
            try:
                return self._loop.run_until_complete(asyncio.wait_for(coro, config.CAMERA_CALL_TIMEOUT))
            except asyncio.TimeoutError:
                logger.warning("Camera did not answer within %.0f s - command abandoned", config.CAMERA_CALL_TIMEOUT)
                raise

    def get_stream_uri(self):
        """Return the RTSP stream URI for this camera's first media profile."""
        async def _call():
            request = self.media.create_type("GetStreamUri")
            request.ProfileToken = self.profile_token
            request.StreamSetup = {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            }
            result = await self.media.GetStreamUri(request)
            return result.Uri

        return self._run(_call())

    def move(self, pan_speed=0.0, tilt_speed=0.0, zoom_speed=0.0, prevalidated=False,
             travel_mode=False, pulse_seconds=None):
        """Continuous PTZ move. Speeds are normalized, typically -1.0..1.0.

        ``prevalidated=True`` is only for the internal tracker after it has
        already obtained and checked one PTZ status. This avoids duplicate
        ONVIF status calls on every frame while retaining the default fail-safe
        guards for external callers.

        ``travel_mode=True`` uses a slightly longer pan pulse so the smart
        patrol can actually sweep left/right across stations without stalling
        on short pulses when inference FPS is modest.
        """
        # operator-takeover detection: the pulse plus the camera's coasting
        self.last_command_time = time.monotonic() + float(pulse_seconds or 0.5) + 1.0
        bounded = config.ALLOWED_PAN_RANGE is not None
        if bounded and not prevalidated:
            pan_speed = self.guard_pan_speed(pan_speed)
            zoom_speed = self.guard_zoom_speed(zoom_speed)
            if pan_speed == 0.0 and tilt_speed == 0.0 and zoom_speed == 0.0:
                self.stop()
                return

        async def _call():
            request = self.ptz.create_type("ContinuousMove")
            request.ProfileToken = self.profile_token
            request.Velocity = {
                "PanTilt": {"x": pan_speed * self.pan_command_sign, "y": tilt_speed,
                            "space": self._pan_tilt_space},
                "Zoom": {"x": zoom_speed, "space": self._zoom_space},
            }
            if bounded:
                # float(): a numpy.float32 pulse (optical-flow box) made timedelta raise for every
                # pulse >= 0.6 s - 640 failed tracking moves 21-23/10, each one stopping the camera
                request.Timeout = timedelta(seconds=float(max(config.BOUNDARY_MOVE_TIMEOUT_SECONDS,
                                                                (pulse_seconds or 0.0) + 0.6)))
            try:
                await self.ptz.ContinuousMove(request)
                if bounded:
                    # Longer pan pulse while travelling between scan stations so
                    # the camera visibly rotates left/right even at modest FPS.
                    if travel_mode and (pan_speed != 0.0 or tilt_speed != 0.0):
                        pt_duration = getattr(
                            config, "SMART_PATROL_TRAVEL_PULSE_SECONDS",
                            config.BOUNDARY_PAN_PULSE_SECONDS * 1.4,
                        )
                    else:
                        pt_duration = config.BOUNDARY_PAN_PULSE_SECONDS
                    if pulse_seconds is not None:
                        pt_duration = float(pulse_seconds)
                    zoom_duration = config.ZOOM_PULSE_SECONDS
                    moving_pt = pan_speed != 0.0 or tilt_speed != 0.0
                    if moving_pt:
                        await asyncio.sleep(pt_duration)
                        await self.ptz.Stop({"ProfileToken": self.profile_token,
                                             "PanTilt": True, "Zoom": False})
                    if zoom_speed != 0.0:
                        await asyncio.sleep(max(0.0, zoom_duration -
                                                (pt_duration if moving_pt else 0.0)))
            finally:
                if bounded:
                    await self.ptz.Stop({"ProfileToken": self.profile_token,
                                         "PanTilt": True, "Zoom": True})

        self._run(_call())

    @staticmethod
    def pan_limits(zoom=None):
        """(low, high, margin) of the wildlife arc at this zoom.  27/10: the high end reaches
        past ALLOWED_PAN_RANGE while zoomed in (utils.privacy.pan_high_for_zoom - the lodge
        deck stays out of the picture), and the edge margin shrinks with the zoom like the
        camera's own pan rate does."""
        low, high = config.ALLOWED_PAN_RANGE
        margin = config.PAN_BOUNDARY_MARGIN
        try:
            z = float(zoom)
        except (TypeError, ValueError):
            return low, high, margin
        if not math.isfinite(z):
            return low, high, margin
        wide = 1.0 + config.PTZ_ZOOM_MAG_PER_UNIT * config.PATROL_SWEEP_ZOOM
        m = 1.0 + config.PTZ_ZOOM_MAG_PER_UNIT * max(0.0, z)
        margin = max(config.PAN_BOUNDARY_MARGIN_MIN, margin * min(1.0, wide / m))
        return low, pan_high_for_zoom(z), margin

    def safe_zoom(self, zoom, pan=None):
        """``zoom``, or the widest zoom that keeps the lodge deck out of the picture at this
        pan (27/10: far left, past the old arc end, the camera must not widen)."""
        if pan is None:
            pan = getattr(self, "_last_good_pan", None)
        if pan is None:
            return zoom
        floor = min_zoom_for_pan(pan)
        if zoom < floor:
            logger.info("Zoom %.3f refused at pan %.4f (the lodge deck would show): %.3f instead",
                        zoom, pan, floor)
            return floor
        return zoom

    def guard_pan_speed(self, speed, status=None, margin=None, max_speed=None):
        """Allow only inward pan near an edge; fail closed on invalid status."""
        limits = config.ALLOWED_PAN_RANGE
        if limits is None or speed == 0.0:
            return speed
        try:
            status = status if status is not None else self.get_status()
            zoom = getattr(getattr(status, "Zoom", None), "x", None)
            low, high, auto_margin = self.pan_limits(zoom)
            margin = auto_margin if margin is None else float(margin)
            if not (all(math.isfinite(v) for v in (low, high, margin, speed))
                    and -1.0 <= low < high <= 1.0
                    and 0 <= margin < (high - low) / 2):
                raise ValueError("Invalid pan limits")
            pan = float(status.PanTilt.x)
            if not math.isfinite(pan):
                raise ValueError("Non-finite pan position")
            tolerance = config.PAN_RECOVERY_TOLERANCE
            if not low <= pan <= high:
                if low - tolerance <= pan < low and speed > 0:
                    return min(speed, config.PAN_RECOVERY_SPEED)
                if high < pan <= high + tolerance and speed < 0:
                    return max(speed, -config.PAN_RECOVERY_SPEED)
                logger.warning("Pan blocked: position %.6f outside [%.6f, %.6f]; "
                               "only small inward recovery permitted", pan, low, high)
                return 0.0
            if ((speed < 0 and pan <= low + margin)
                    or (speed > 0 and pan >= high - margin)):
                return 0.0
            cap = config.BOUNDARY_PAN_MAX_SPEED if max_speed is None else float(max_speed)
            return max(-cap, min(cap, speed))
        except (AttributeError, TypeError, ValueError):
            logger.warning("Pan blocked: position unavailable or outside wildlife limits")
            return 0.0

    def guard_zoom_speed(self, speed, status=None):
        """Stop at lens limits or when position feedback is unavailable."""
        if speed == 0.0:
            return 0.0
        try:
            status = status if status is not None else self.get_status()
            position = float(status.Zoom.x)
            if not math.isfinite(position) or not 0.0 <= position <= 1.0 or not math.isfinite(speed):
                return 0.0
            if speed > 0 and position >= config.ZOOM_POSITION_MAX - config.ZOOM_POSITION_MARGIN:
                return 0.0
            if speed < 0 and position <= config.ZOOM_POSITION_MIN + config.ZOOM_POSITION_MARGIN:
                return 0.0
            if speed < 0 and float(status.PanTilt.x) > config.ALLOWED_PAN_RANGE[1]:
                return 0.0              # past the old arc end only zoomed in (27/10, lodge deck)
            return max(-config.MAX_ZOOM_SPEED, min(config.MAX_ZOOM_SPEED, speed))
        except (AttributeError, TypeError, ValueError):
            logger.warning("Zoom blocked: lens position unavailable")
            return 0.0

    def stop(self):
        async def _call():
            request = self.ptz.create_type("Stop")
            request.ProfileToken = self.profile_token
            request.PanTilt = True
            request.Zoom = True
            await self.ptz.Stop(request)

        self._run(_call())

    def _read_position(self):
        async def _call():
            result = await self.ptz.GetStatus({"ProfileToken": self.profile_token})
            return result.Position

        return self._run(_call())

    def _looks_mirrored(self, pan):
        """Is ``pan`` the sign-flipped image of a pose inside the wildlife arc?"""
        limits = config.ALLOWED_PAN_RANGE
        if limits is None:
            ref = self._last_good_pan
            return ref is not None and _is_mirror(pan, ref)
        low, high = limits
        tol = config.PTZ_POSE_MIRROR_TOLERANCE

        def inside(p):
            return low - tol <= p <= high + tol
        return not inside(pan) and inside(-pan)

    def _filter_pan_glitch(self, position):
        """Correct the sign-flipped pan readings this SUNBA camera returns.

        Measured 2026-10-16: while the camera TRAVELS every GetStatus reply
        carries -pan; when it is still ~5 % do.  The magnitude is always right.
        Before v3.1 a flipped reading looked like "camera outside the arc": the
        patrol stopped ("status unavailable") and the pan guard blocked moves.

        The wildlife arc lies entirely on one side of 0, so a reading outside the
        arc whose mirror image is inside it gets its sign corrected.  It is
        believed as a real position only if it stays unchanged (camera still)
        for PTZ_POSE_GLITCH_CONFIRM_SECONDS - then the normal guards treat the
        camera as outside the arc and refuse to move it.
        """
        pan = float(position.PanTilt.x)
        if not self._looks_mirrored(pan):
            self._last_good_pan = pan
            self._mirror_since = None
            return position
        now = time.monotonic()
        if self._mirror_since is None or abs(pan - self._mirror_value) > 0.002:
            self._mirror_since, self._mirror_value = now, pan   # new or still moving
        # With a wildlife arc, a reading in its mirror image is ALWAYS treated as
        # the glitch: on 17/10 the idle camera reported -pan for >10 s at a time.
        # (Patrol moves are bounded and its AbsoluteMoves land in the true arc, so
        # even a camera hand-moved to the mirror side is driven back correctly.)
        if config.ALLOWED_PAN_RANGE is None and now - self._mirror_since >= config.PTZ_POSE_GLITCH_CONFIRM_SECONDS:
            if self._last_good_pan is None or self._last_good_pan != pan:
                logger.warning("PTZ pan %.4f unchanged for %.0fs - accepted as the real position",
                               pan, now - self._mirror_since)
            self._last_good_pan = pan
            return position
        position.PanTilt.x = -pan
        self._last_good_pan = -pan
        return position

    def get_status(self):
        """Return current pan/tilt/zoom position, if the camera reports it."""
        try:
            return self._filter_pan_glitch(self._read_position())
        except Exception as exc:  # camera-specific quirk, not fatal to tracking
            now = time.monotonic()
            if now - self.__dict__.get("_last_status_warning", 0.0) >= 10.0:   # do not spam at 8 Hz
                self.__dict__["_last_status_warning"] = now
                logger.warning("Could not read PTZ status: %s", exc)
            return None

    def goto_position(self, pan, tilt, zoom=None):
        """Absolute move to a normalized PTZ position inside the wildlife arc.

        This is the reliable way to sweep left/center/right on cameras where
        short ContinuousMove pan pulses are weak or ignored. Zoom may be
        omitted (None) to leave the lens alone during pan travel.
        """
        self.last_command_time = time.monotonic() + 3.5   # the camera keeps moving ~3 s (operator detection)
        low, high = config.ALLOWED_PAN_RANGE
        if zoom is not None:
            zoom = max(
                config.ZOOM_POSITION_MIN,
                min(config.ZOOM_POSITION_MAX, float(zoom)),
            )
            high = pan_high_for_zoom(zoom)      # zoomed in it may go a little further (27/10)
        pan = max(low, min(high, float(pan)))
        tilt = max(-1.0, min(1.0, float(tilt)))

        if not self._abs_pan_tilt_space:
            # Fallback: continuous pulses toward the target.
            logger.warning("Absolute PTZ not available; using continuous pan toward target")
            status = self.get_status()
            if status is None:
                return False
            cur = float(status.PanTilt.x)
            dx = pan - cur
            if abs(dx) < 0.008:
                return True
            speed = math.copysign(min(config.SEARCH_PAN_SPEED, max(0.12, abs(dx))), dx)
            self.move(pan_speed=speed, travel_mode=True)
            return True

        async def _call():
            request = self.ptz.create_type("AbsoluteMove")
            request.ProfileToken = self.profile_token
            position = {
                "PanTilt": {"x": pan * self.pan_command_sign, "y": tilt,
                            "space": self._abs_pan_tilt_space},
            }
            if zoom is not None and self._abs_zoom_space:
                position["Zoom"] = {"x": zoom, "space": self._abs_zoom_space}
            request.Position = position
            # Moderate speed so the move is visible but controlled - but only on
            # axes the camera declares a speed space for (an unsupported PanTilt
            # speed can make the whole request fail).
            speed = {}
            if getattr(self, "_pt_speed_supported", False):
                speed["PanTilt"] = {"x": 0.35, "y": 0.20}
            if zoom is not None and self._abs_zoom_space and getattr(self, "_zoom_speed_supported", False):
                speed["Zoom"] = {"x": 0.5}
            if speed:
                request.Speed = speed
            await self.ptz.AbsoluteMove(request)
            logger.info("AbsoluteMove → pan=%.4f tilt=%.3f zoom=%s", pan, tilt, zoom)

        try:
            self._run(_call())
            return True
        except Exception as exc:
            logger.warning("AbsoluteMove failed (%s); falling back to continuous", exc)
            status = self.get_status()
            if status is None:
                return False
            cur = float(status.PanTilt.x)
            dx = pan - cur
            speed = math.copysign(min(config.SEARCH_PAN_SPEED, max(0.12, abs(dx) or 0.15)), dx or -1)
            self.move(pan_speed=speed, travel_mode=True)
            return True

    def relative_pan(self, delta_pan, delta_tilt=0.0):
        """Nudge pan/tilt by a relative amount (clamped to wildlife arc)."""
        self.last_command_time = time.monotonic()   # operator-takeover detection
        status = self.get_status()
        if status is None:
            return False
        cur_pan = float(status.PanTilt.x)
        cur_tilt = float(status.PanTilt.y)
        low, high = config.ALLOWED_PAN_RANGE
        target_pan = max(low, min(high, cur_pan + float(delta_pan)))
        target_tilt = max(-1.0, min(1.0, cur_tilt + float(delta_tilt)))
        return self.goto_position(target_pan, target_tilt, zoom=None)


    # --- v3: pose helpers, non-blocking closed-loop motion, focus ------------

    def get_pose(self):
        """Return (pan, tilt, zoom) floats, or None if the camera does not answer."""
        status = self.get_status()
        try:
            pan = float(status.PanTilt.x)
            tilt = float(status.PanTilt.y)
            zoom = float(status.Zoom.x)
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (pan, tilt, zoom)):
            return None
        return pan, tilt, zoom

    def velocity_move(self, pan_speed=0.0, tilt_speed=0.0, zoom_speed=0.0,
                      pose=None, edge_margin=None, timeout=None):
        """NON-blocking ContinuousMove with a device-side auto-stop timeout.

        Unlike :meth:`move` this returns immediately; the camera keeps moving
        until ``timeout`` seconds pass (default PTZ_CONTINUOUS_TIMEOUT) unless
        refreshed or stopped.  Every command still passes the hard pan guards:
        ``edge_margin`` only controls how close to the edge a *patrol* move may
        go - the arc itself (ALLOWED_PAN_RANGE) is never exceeded.
        Returns the (pan, tilt, zoom) speeds actually sent.
        """
        self.last_command_time = time.monotonic() + float(timeout or config.PTZ_CONTINUOUS_TIMEOUT)   # runs >= 1 s
        status = None
        if pose is not None:
            status = SimpleNamespace_pose(pose)
        pan_speed = self.guard_pan_speed(pan_speed, status=status, margin=edge_margin)
        zoom_speed = self.guard_zoom_speed(zoom_speed, status=status)
        if pan_speed == 0.0 and tilt_speed == 0.0 and zoom_speed == 0.0:
            self.stop()
            return 0.0, 0.0, 0.0
        timeout = config.PTZ_CONTINUOUS_TIMEOUT if timeout is None else timeout
        # Measured: this camera runs Timeouts in WHOLE seconds and treats anything
        # under 1 s as ~2 s (0.5 s moved as far as 2.1 s).  Never send a fraction.
        timeout = max(1, int(round(float(timeout))))

        async def _call():
            request = self.ptz.create_type("ContinuousMove")
            request.ProfileToken = self.profile_token
            request.Velocity = {
                "PanTilt": {"x": pan_speed * self.pan_command_sign, "y": tilt_speed,
                            "space": self._pan_tilt_space},
                "Zoom": {"x": zoom_speed, "space": self._zoom_space},
            }
            request.Timeout = timedelta(seconds=float(timeout))
            await self.ptz.ContinuousMove(request)

        try:
            self._run(_call())
        except Exception as exc:
            logger.warning("ContinuousMove failed: %s", exc)
            return 0.0, 0.0, 0.0
        return pan_speed, tilt_speed, zoom_speed

    # --- focus (ONVIF Imaging) ---

    @property
    def focus_available(self):
        return getattr(self, "_imaging", None) is not None and self._video_source_token is not None

    def get_focus_info(self):
        """Return a dict describing the focus state/options, or None."""
        if not self.focus_available:
            return None

        async def _call():
            settings = await self._imaging.GetImagingSettings({"VideoSourceToken": self._video_source_token})
            info = {"auto_focus_mode": None, "default_speed": None}
            focus = getattr(settings, "Focus", None)
            if focus is not None:
                info["auto_focus_mode"] = getattr(focus, "AutoFocusMode", None)
                info["default_speed"] = getattr(focus, "DefaultSpeed", None)
            return info

        try:
            return self._run(_call())
        except Exception as exc:
            logger.info("GetImagingSettings failed: %s", exc)
            return None

    def set_autofocus(self, enabled=True):
        """Switch the lens between continuous autofocus (AUTO) and MANUAL."""
        if not self.focus_available:
            return False

        async def _call():
            request = self._imaging.create_type("SetImagingSettings")
            request.VideoSourceToken = self._video_source_token
            request.ImagingSettings = {"Focus": {"AutoFocusMode": "AUTO" if enabled else "MANUAL"}}
            request.ForcePersistence = False
            await self._imaging.SetImagingSettings(request)

        try:
            self._run(_call())
            logger.info("Focus mode set to %s", "AUTO" if enabled else "MANUAL")
            return True
        except Exception as exc:
            logger.warning("Could not set focus mode: %s", exc)
            return False

    def focus_nudge(self, speed, seconds=None):
        """Move focus by a short pulse (manual focus). Negative = near, positive = far
        (direction convention varies by camera - the focus tuner learns it)."""
        if not self.focus_available:
            return False
        seconds = config.FOCUS_NUDGE_SECONDS if seconds is None else seconds

        async def _call():
            request = self._imaging.create_type("Move")
            request.VideoSourceToken = self._video_source_token
            request.Focus = {"Continuous": {"Speed": float(speed)}}
            await self._imaging.Move(request)
            await asyncio.sleep(float(seconds))
            await self._imaging.Stop({"VideoSourceToken": self._video_source_token})

        try:
            self._run(_call())
            return True
        except Exception as exc:
            logger.info("Focus move failed: %s", exc)
            try:
                self._run(self._imaging.Stop({"VideoSourceToken": self._video_source_token}))
            except Exception:
                pass
            return False

    def get_focus_position(self):
        """Lens focus position (0..1) from Imaging GetStatus, or None."""
        if not self.focus_available:
            return None

        async def _call():
            status = await self._imaging.GetStatus({"VideoSourceToken": self._video_source_token})
            return float(status.FocusStatus20.Position)

        try:
            return self._run(_call())
        except Exception as exc:
            logger.debug("Focus GetStatus failed: %s", exc)
            return None

    def focus_absolute(self, position, speed=1.0):
        """Drive the focus to an absolute position (the lens must be in MANUAL)."""
        if not self.focus_available:
            return False

        async def _call():
            request = self._imaging.create_type("Move")
            request.VideoSourceToken = self._video_source_token
            request.Focus = {"Absolute": {"Position": max(0.0, min(1.0, float(position))),
                                          "Speed": float(speed)}}
            await self._imaging.Move(request)

        try:
            self._run(_call())
            return True
        except Exception as exc:
            logger.info("Absolute focus move failed: %s", exc)
            return False

    def zoom_only(self, zoom):
        """AbsoluteMove of the ZOOM axis only.

        Pan/tilt are not part of the request, so the camera keeps its current
        tilt - which matters because this camera cannot report (or be sent) tilts
        above -1.0, where the plain and the water-hole actually are.
        """
        self.last_command_time = time.monotonic() + 3.5   # the camera keeps moving ~3 s (operator detection)
        if not self._abs_zoom_space:
            return False
        zoom = self.safe_zoom(max(config.ZOOM_POSITION_MIN, min(config.ZOOM_POSITION_MAX, float(zoom))))

        async def _call():
            request = self.ptz.create_type("AbsoluteMove")
            request.ProfileToken = self.profile_token
            request.Position = {"Zoom": {"x": zoom, "space": self._abs_zoom_space}}
            await self.ptz.AbsoluteMove(request)

        try:
            self._run(_call())
            return True
        except Exception as exc:
            logger.warning("Zoom-only move failed: %s", exc)
            return False

    def goto_preset_safe(self, token, expected_pan, pose=None):
        """GotoPreset for a station preset - only when the camera is already near it.

        Presets store the real motor position (including tilt above -1.0).  The
        camera chooses its own route, so presets are only used for the final
        short correction: the patrol first glides to the station's pan.
        """
        self.last_command_time = time.monotonic() + 3.5   # the camera keeps moving ~3 s (operator detection)
        pose = pose or self.get_pose()
        low, high = config.ALLOWED_PAN_RANGE
        if (pose is None or not low - 0.01 <= pose[0] <= high + 0.01
                or abs(pose[0] - float(expected_pan)) > config.PRESET_MAX_APPROACH_PAN):
            logger.warning("Preset %s refused: camera at %s, preset pan %.4f (too far - safety)",
                           token, None if pose is None else round(pose[0], 4), expected_pan)
            return False

        async def _call():
            request = self.ptz.create_type("GotoPreset")
            request.ProfileToken = self.profile_token
            request.PresetToken = str(token)
            await self.ptz.GotoPreset(request)

        try:
            self._run(_call())
            return True
        except Exception as exc:
            logger.warning("GotoPreset %s failed: %s", token, exc)
            return False

    # --- Presets (used for patrol waypoints and the no-go zone helper) ---

    def get_presets(self):
        async def _call():
            return await self.ptz.GetPresets({"ProfileToken": self.profile_token})

        return self._run(_call())

    def set_preset(self, preset_name, token=None):
        """Save the camera's CURRENT position as a preset (``token`` overwrites one). Returns its token."""
        async def _call():
            request = self.ptz.create_type("SetPreset")
            request.ProfileToken = self.profile_token
            request.PresetName = preset_name
            if token is not None:
                request.PresetToken = str(token)
            response = await self.ptz.SetPreset(request)
            return getattr(response, "PresetToken", response)

        return self._run(_call())

    def goto_preset(self, preset_token):
        if config.ALLOWED_PAN_RANGE is not None:
            self.stop()
            raise ValueError("Preset travel blocked: camera may route through the hotel")
        async def _call():
            request = self.ptz.create_type("GotoPreset")
            request.ProfileToken = self.profile_token
            request.PresetToken = preset_token
            await self.ptz.GotoPreset(request)

        self._run(_call())

    def goto_preset_by_name(self, name):
        if config.ALLOWED_PAN_RANGE is not None:
            self.stop()
            logger.warning("Preset travel blocked while wildlife pan limits are enabled")
            return False
        for preset in self.get_presets():
            if getattr(preset, "Name", None) == name:
                self.goto_preset(preset.token)
                return True
        logger.warning("No saved preset named '%s' on the camera", name)
        return False

    def close(self):
        """Cleanly shut down the underlying aiohttp session (avoids 'unclosed client session' warnings)."""
        async def _call():
            close = getattr(self.cam, "close", None)
            if close is not None:
                await close()

        try:
            self._run(_call())
        except Exception as exc:
            logger.debug("Error closing camera session: %s", exc)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
