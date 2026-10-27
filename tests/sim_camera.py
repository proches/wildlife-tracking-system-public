"""A physics-ish fake PTZ camera for patrol tests (no network, no onvif library)."""
import asyncio
from types import SimpleNamespace

import config
from camera.onvif_camera import OnvifPTZCamera


def measured_pan_rate(speed):
    """Pan rate (pose units/s, wide zoom) for a ContinuousMove speed - the SUNBA's
    measured STEP levels (17/10/2026): nothing / slow 0.6 deg/s / ~7 / ~12 / ~16 deg/s."""
    a = abs(speed)
    if a < 0.12:
        r = 0.0
    elif a < 0.223:
        r = 0.0034
    elif a < 0.34:
        r = 0.040
    elif a < 0.45:
        r = 0.068
    else:
        r = 0.090
    return r if speed >= 0 else -r


class SimCamera(OnvifPTZCamera):
    """Subclasses the real camera so the REAL pan/zoom guards are exercised.

    absolute_works=False reproduces the field bug: AbsoluteMove returns OK but
    the lens never moves.
    """

    PAN_RATE = 0.056 / 0.30   # measured: gliding at speed 0.30 for 1 s moves ~0.056
    ZOOM_RATE = 0.20
    TILT_RATE = 0.30

    def __init__(self, pan=0.476, tilt=-1.0, zoom=0.011, absolute_works=True, clock=None):
        self._loop = asyncio.new_event_loop()
        self.pan, self.tilt, self.zoom = pan, tilt, zoom
        self.absolute_works = absolute_works
        self.vel = (0.0, 0.0, 0.0)
        self.vel_until = -1.0
        self.abs_target = None
        self.t = 0.0
        self.log = []
        self._imaging = None

    # -- interface used by the code under test
    def get_status(self):
        return SimpleNamespace(PanTilt=SimpleNamespace(x=self.pan, y=self.tilt), Zoom=SimpleNamespace(x=self.zoom))

    def get_pose(self):
        return self.pan, self.tilt, self.zoom

    def stop(self):
        self.vel = (0.0, 0.0, 0.0)
        self.abs_target = None

    def goto_position(self, pan, tilt, zoom=None):
        low, high = config.ALLOWED_PAN_RANGE
        pan = max(low, min(high, pan))
        self.log.append(("abs", pan, tilt, zoom))
        if self.absolute_works:
            self.abs_target = (pan, tilt, zoom)
        return True

    def velocity_move(self, pan_speed=0.0, tilt_speed=0.0, zoom_speed=0.0, pose=None,
                      edge_margin=None, timeout=None):
        from camera.onvif_camera import SimpleNamespace_pose
        status = SimpleNamespace_pose(pose) if pose is not None else None
        pan_speed = self.guard_pan_speed(pan_speed, status=status, margin=edge_margin)
        zoom_speed = self.guard_zoom_speed(zoom_speed, status=status)
        self.log.append(("vel", pan_speed, tilt_speed, zoom_speed))
        self.abs_target = None
        self.vel = (pan_speed, tilt_speed, zoom_speed)
        self.vel_until = self.t + (timeout or config.PTZ_CONTINUOUS_TIMEOUT)
        return pan_speed, tilt_speed, zoom_speed

    # -- simulation
    def advance(self, dt):
        self.t += dt
        if self.abs_target is not None:
            tp, tt, tz = self.abs_target
            for attr, tgt, rate in (("pan", tp, self.PAN_RATE * 0.35 / 0.3), ("tilt", tt, self.TILT_RATE),
                                    ("zoom", tz, self.ZOOM_RATE * 0.25 / 0.2)):
                if tgt is None:
                    continue
                cur = getattr(self, attr)
                step = max(-rate * dt, min(rate * dt, tgt - cur))
                setattr(self, attr, cur + step)
            return
        if self.t > self.vel_until:
            self.vel = (0.0, 0.0, 0.0)
        p, ti, z = self.vel
        self.pan += measured_pan_rate(p) * dt
        self.tilt = max(-1.0, min(1.0, self.tilt + ti * self.TILT_RATE * dt))
        self.zoom = max(0.0, min(1.0, self.zoom + z * self.ZOOM_RATE * dt))

    def zoom_only(self, zoom):
        """AbsoluteMove of the zoom axis only (pan/tilt untouched), applied at once."""
        zoom = self.safe_zoom(max(config.ZOOM_POSITION_MIN, min(config.ZOOM_POSITION_MAX, float(zoom))), pan=self.pan)
        self.log.append(("zoomto", zoom))
        self.zoom = zoom
        return True

    def move(self, pan_speed=0.0, tilt_speed=0.0, zoom_speed=0.0, prevalidated=False,
             travel_mode=False, pulse_seconds=None):
        """Blocking pulse used by live tracking (integrates the pulse instantly)."""
        pan_speed = self.guard_pan_speed(pan_speed) if not prevalidated else pan_speed
        dur = pulse_seconds if pulse_seconds is not None else config.BOUNDARY_PAN_PULSE_SECONDS
        self.log.append(("pulse", pan_speed, tilt_speed, zoom_speed, dur))
        self.pan += measured_pan_rate(pan_speed) * dur
        self.tilt = max(-1.0, min(1.0, self.tilt + tilt_speed * self.TILT_RATE * dur))
        self.zoom = max(0.0, min(1.0, self.zoom + zoom_speed * self.ZOOM_RATE * config.ZOOM_PULSE_SECONDS))
