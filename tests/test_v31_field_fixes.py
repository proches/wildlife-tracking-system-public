"""v3.1 regressions for faults measured on the real SUNBA camera (2026-10-16).

Unlike tests/sim_camera.py, the fake here sits BELOW OnvifPTZCamera (it fakes
the ONVIF PTZ service), so the real velocity_move / goto_position / get_status
code is exercised - that is where the field bugs were.
"""
import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from lxml import etree

import config
from camera.focus import FocusController, sharpness
from camera.onvif_camera import OnvifPTZCamera
from detection.validator import is_verified, species_label
from tracking.motion import MotionController
from tracking.multi_object_tracker import MultiObjectTracker
from tracking.observation_patrol import OBSERVE_PHASES, ObservationPatrol
from tests.sim_camera import measured_pan_rate


class FakeSunbaPTZ:
    """ONVIF PTZ service with the measured quirks of the SUNBA camera.

    * EFlip=ON: pan commands are executed mirrored (velocity +x lowers the
      reported pan, AbsoluteMove x lands at reported -x).
    * every ``glitch_every``-th GetStatus reports -pan.
    * AbsoluteMove with a PanTilt speed is rejected (empty PanTiltSpeedSpace).
    """

    PAN_RATE, ZOOM_RATE, ABS_PAN_RATE, ABS_ZOOM_RATE = None, 0.25, 0.30, 0.20   # None = measured step levels

    def __init__(self, pan=0.7275, zoom=0.145, glitch_every=0, mirrored=True):
        self.pan, self.tilt, self.zoom = pan, -1.0, zoom
        self.mirror = -1.0 if mirrored else 1.0
        self.glitch_every = glitch_every
        self.reads = 0
        self.vel = (0.0, 0.0, 0.0)
        self.until = 0.0
        self.abs = None
        self.t = 0.0
        self.min_pan = self.max_pan = pan
        self.coast = 0.0
        self.coast_until = -1.0
        self.abs_jumps = []
        self.presets = {}            # token -> (pan, zoom, hidden tilt above -1.0)
        self.preset_calls = []
        self.hidden_tilt = 0.0       # the tilt the camera cannot report
        self.zoom_only_calls = 0

    @staticmethod
    def create_type(_name):
        return SimpleNamespace()

    async def GetStatus(self, _req):
        self.reads += 1
        pan = self.pan
        if self.glitch_every and self.reads % self.glitch_every == 0:
            pan = -pan
        return SimpleNamespace(Position=SimpleNamespace(
            PanTilt=SimpleNamespace(x=pan, y=self.tilt), Zoom=SimpleNamespace(x=self.zoom)))

    async def ContinuousMove(self, req):
        v = req.Velocity
        self.raw_x = v["PanTilt"]["x"]
        self.abs = None
        self.vel = (self.mirror * v["PanTilt"]["x"], v["PanTilt"]["y"], v["Zoom"]["x"])
        timeout = getattr(req, "Timeout", None)
        self.until = self.t + (timeout.total_seconds() if timeout else 1e9)

    async def AbsoluteMove(self, req):
        speed = getattr(req, "Speed", None) or {}
        if "PanTilt" in speed:
            raise RuntimeError("ter:InvalidArgVal - PanTilt speed not supported")
        pos = req.Position
        if "PanTilt" not in pos:                       # zoom-only move keeps pan AND hidden tilt
            self.abs = (self.pan, self.tilt, pos["Zoom"]["x"])
            self.vel = (0.0, 0.0, 0.0)
            self.zoom_only_calls += 1
            return
        self.hidden_tilt = 0.0                         # pan/tilt AbsoluteMove resets the raised tilt
        zoom = pos.get("Zoom", {}).get("x") if isinstance(pos.get("Zoom"), dict) else None
        self.abs = (self.mirror * pos["PanTilt"]["x"], pos["PanTilt"]["y"], zoom)
        self.abs_jumps.append(abs(self.abs[0] - self.pan))
        self.vel = (0.0, 0.0, 0.0)

    async def GotoPreset(self, req):
        pan, zoom, hidden = self.presets[str(req.PresetToken)]
        self.abs = (pan, self.tilt, zoom)
        self.vel = (0.0, 0.0, 0.0)
        self.hidden_tilt = hidden
        self.preset_calls.append(str(req.PresetToken))

    async def Stop(self, _req):
        if self.coast and self.vel[0]:
            self.coast_until = self.t + self.coast      # like the real camera: it slides on
            self.until = self.coast_until               # ...and then stops
            self.vel = (self.vel[0], 0.0, 0.0)
        else:
            self.vel = (0.0, 0.0, 0.0)
        self.abs = None

    def advance(self, dt):
        self.t += dt
        if self.abs is not None:
            for attr, tgt, rate in (("pan", self.abs[0], self.ABS_PAN_RATE), ("tilt", self.abs[1], 0.3),
                                    ("zoom", self.abs[2], self.ABS_ZOOM_RATE)):
                if tgt is not None:
                    cur = getattr(self, attr)
                    setattr(self, attr, cur + max(-rate * dt, min(rate * dt, tgt - cur)))
        else:
            moving_dt = max(0.0, min(dt, max(self.until, self.coast_until) - (self.t - dt)))
            p, ti, z = self.vel
            self.pan += (measured_pan_rate(p) if self.PAN_RATE is None else p * self.PAN_RATE) * moving_dt
            if self.t >= self.until and self.t >= self.coast_until:
                self.vel = (0.0, 0.0, 0.0)
            self.tilt = max(-1.0, min(1.0, self.tilt + ti * 0.3 * dt))
            self.zoom = max(0.0, min(1.0, self.zoom + z * self.ZOOM_RATE * dt))
        self.min_pan, self.max_pan = min(self.min_pan, self.pan), max(self.max_pan, self.pan)


def make_camera(device, sign=-1):
    cam = OnvifPTZCamera.__new__(OnvifPTZCamera)
    cam._loop = asyncio.new_event_loop()
    cam.profile_token = "p"
    cam._pan_tilt_space = "vel"
    cam._zoom_space = "zvel"
    cam._abs_pan_tilt_space = "abs"
    cam._abs_zoom_space = "zabs"
    cam._pt_speed_supported = False
    cam._zoom_speed_supported = True
    cam._imaging = None
    cam._video_source_token = None
    cam.pan_command_sign = sign
    cam.ptz = device
    return cam


class TempCapsMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        caps = os.path.join(self.tmp.name, "caps.json")
        p = patch.object(config, "PTZ_CAPABILITIES_FILE", caps)
        p.start()
        self.addCleanup(p.stop)
        self.caps = caps


class PanSignTests(TempCapsMixin, unittest.TestCase):
    def test_commands_are_mirrored_for_eflip_camera(self):
        dev = FakeSunbaPTZ(pan=0.6)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        cam.velocity_move(0.2, 0.0, 0.0, pose=(0.6, -1.0, 0.1))
        self.assertLess(dev.raw_x, 0, "device must receive a mirrored (negative) command...")
        dev.advance(0.5)
        self.assertGreater(dev.pan, 0.6, "...so a +pan request raises the REPORTED pan")
        cam.velocity_move(-0.2, 0.0, 0.0, pose=(dev.pan, -1.0, 0.1))
        for _ in range(9):
            dev.advance(0.1)
        self.assertLess(dev.pan, 0.6)
        cam.goto_position(0.5, -1.0, zoom=0.1)
        for _ in range(40):
            dev.advance(0.1)
        self.assertAlmostEqual(dev.pan, 0.5, places=3)
        self.assertAlmostEqual(dev.zoom, 0.1, places=3)

    def test_absolute_move_omits_unsupported_pan_tilt_speed(self):
        dev = FakeSunbaPTZ(pan=0.6)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        self.assertTrue(cam.goto_position(0.55, -1.0))
        self.assertIsNotNone(dev.abs)   # accepted, not rejected for the speed

    def test_flip_mode_parsing_and_sign_resolution(self):
        el = etree.fromstring(
            b'<tt:PTControlDirection xmlns:tt="http://www.onvif.org/ver10/schema">'
            b'<tt:EFlip><tt:Mode>ON</tt:Mode></tt:EFlip></tt:PTControlDirection>')
        cfg = SimpleNamespace(Extension=SimpleNamespace(_value_1=[el], PTControlDirection=None))
        modes = OnvifPTZCamera._read_flip_modes(cfg)
        self.assertEqual(modes, {"EFlip": "ON"})
        with patch.object(config, "PTZ_PAN_COMMAND_SIGN", "auto"):
            self.assertEqual(OnvifPTZCamera._resolve_pan_sign(modes), -1)
            self.assertEqual(OnvifPTZCamera._resolve_pan_sign({"EFlip": "OFF"}), 1)
            self.assertEqual(OnvifPTZCamera._resolve_pan_sign({"EFlip": "ON", "Reverse": "ON"}), 1)
            with open(self.caps, "w") as f:
                json.dump({"pan_command_sign": 1}, f)
            self.assertEqual(OnvifPTZCamera._resolve_pan_sign(modes), -1,
                             "the camera's live EFlip setting beats a (possibly corrupted) file")
            self.assertEqual(OnvifPTZCamera._resolve_pan_sign({}), 1, "file used when the camera says nothing")
        with patch.object(config, "PTZ_PAN_COMMAND_SIGN", "-1"):
            self.assertEqual(OnvifPTZCamera._resolve_pan_sign({}), -1)

    def test_old_verdict_taken_with_other_sign_is_ignored(self):
        with open(self.caps, "w") as f:
            json.dump({"absolute_move_works": False}, f)        # v3 file, no sign recorded
        with patch.object(config, "PTZ_MOVE_MODE", "auto"):
            mirrored = MotionController(SimpleNamespace(pan_command_sign=-1))
            normal = MotionController(SimpleNamespace(pan_command_sign=1))
        self.assertIsNone(mirrored.absolute_ok)
        self.assertFalse(normal.absolute_ok)


class PoseGlitchTests(unittest.TestCase):
    def test_sign_flipped_reads_are_filtered(self):
        dev = FakeSunbaPTZ(pan=0.7275, glitch_every=3)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        pans = [cam.get_pose()[0] for _ in range(60)]
        self.assertTrue(all(p > 0 for p in pans), pans)

    def test_mirror_of_arc_always_corrected_other_values_real(self):
        dev = FakeSunbaPTZ(pan=0.7275)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        t = [100.0]
        with patch("camera.onvif_camera.time.monotonic", side_effect=lambda: t[0]):
            cam.get_pose()
            dev.pan = -0.7275          # really there (e.g. moved by hand in the web UI)
            self.assertAlmostEqual(cam.get_pose()[0], 0.7275, msg="first: treated as glitch")
            t[0] += config.PTZ_POSE_GLITCH_CONFIRM_SECONDS + 5
            # 17/10: the idle camera reported -pan for >10 s - the mirror of the
            # arc is always corrected
            self.assertAlmostEqual(cam.get_pose()[0], 0.7275, msg="still corrected")
            dev.pan = -0.05                                   # not a mirror of the arc: real
            self.assertAlmostEqual(cam.get_pose()[0], -0.05)

    def test_measured_travel_sequence(self):
        """Raw GetStatus pans recorded during a real 0.7275 -> 0.30 AbsoluteMove."""
        dev = FakeSunbaPTZ(pan=0.7275)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        cam.get_pose()
        for raw in (-0.3486, -0.3002, 0.2999, 0.2999, -0.2999, 0.2999, -0.4566, -0.6784, -0.7275, 0.7275):
            dev.pan = raw
            self.assertAlmostEqual(cam.get_pose()[0], abs(raw))

    def test_burst_of_flipped_reads_while_travelling_is_corrected(self):
        dev = FakeSunbaPTZ(pan=0.70)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        cam.get_pose()
        t = [0.0]
        with patch("camera.onvif_camera.time.monotonic", side_effect=lambda: t[0]):
            for i in range(40):                     # a 10 s burst while travelling
                true_pan = 0.70 - 0.01 * i
                dev.pan = -true_pan
                t[0] += 0.25
                self.assertAlmostEqual(cam.get_pose()[0], true_pan)


class PatrolOnMirroredCameraTests(TempCapsMixin, unittest.TestCase):
    def _run(self, mode, glitch_every, seconds=900):
        dev = FakeSunbaPTZ(pan=0.7275, zoom=0.145, glitch_every=glitch_every)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        now = [0.0]
        with patch.object(config, "PTZ_MOVE_MODE", mode):
            patrol = ObservationPatrol(cam, clock=lambda: now[0])
        visited, zooms, phases = set(), [], []
        for _ in range(int(seconds / 0.125)):
            now[0] += 0.125
            dev.advance(0.125)
            patrol.tick(pose=cam.get_pose())
            phases.append(patrol.phase)
            if patrol.phase in OBSERVE_PHASES:
                patrol.note_result(patrol.observe_epoch)
                visited.add(patrol.station["name"])
                if patrol.station["name"] == "water_hole":
                    zooms.append(dev.zoom)
        return dev, patrol, visited, zooms, phases

    def _check(self, dev, patrol, visited, zooms):
        low, high = config.ALLOWED_PAN_RANGE
        self.assertGreaterEqual(dev.min_pan, low - config.PAN_RECOVERY_TOLERANCE)
        self.assertLessEqual(dev.max_pan, high + config.PAN_RECOVERY_TOLERANCE)
        self.assertEqual(visited, {s["name"] for s in patrol.stations})
        self.assertTrue(zooms and max(zooms) >= config.WATER_HOLE_POSITION["zoom"] - 0.01)

    def test_absolute_patrol_covers_arc_with_glitchy_status(self):
        self._check(*self._run("auto", glitch_every=20)[:4])

    def test_continuous_patrol_covers_arc_with_glitchy_status(self):
        self._check(*self._run("continuous", glitch_every=20)[:4])

    def test_start_at_water_hole_does_not_zoom_out_first(self):
        dev, patrol, _, _, phases = self._run("auto", glitch_every=0, seconds=20)
        self.assertEqual(phases[0], "zoom")
        self.assertNotIn("widen", phases[:40])


class CinematicMotionTests(TempCapsMixin, unittest.TestCase):
    def test_glides_slowly_never_crosses_edge_and_lands_precisely(self):
        dev = FakeSunbaPTZ(pan=0.30, zoom=0.013, glitch_every=17)
        dev.coast = 0.11                           # slides ~0.005 after Stop (measured)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        now = [0.0]
        with patch.object(config, "PTZ_MOVE_MODE", "auto"):
            mc = MotionController(cam, clock=lambda: now[0])
        mc.absolute_ok = True
        water = config.WATER_HOLE_POSITION["pan"]
        mc.set_target(pan=water, tilt=-1.0)
        speeds, res = [], None
        for _ in range(int(240 / 0.125)):      # slow cinematic glide: 0.43 takes ~2 min
            now[0] += 0.125
            prev = dev.pan
            dev.advance(0.125)
            speeds.append(abs(dev.pan - prev) / 0.125)
            res = mc.step(cam.get_pose())
            if res == "arrived":
                break
        self.assertEqual(res, "arrived")
        for _ in range(40):
            dev.advance(0.125)
        self.assertLessEqual(dev.max_pan, config.ALLOWED_PAN_RANGE[1])
        self.assertLess(abs(dev.pan - water), config.PTZ_CINEMATIC_FINAL_TOL + 0.001)
        # next to the edge the glide lands deliberately short; the finish is small
        limit = config.PTZ_CINEMATIC_EDGE_SHORTFALL + config.PTZ_CINEMATIC_GLIDE_WINDOW + 0.002
        self.assertTrue(all(j < limit for j in dev.abs_jumps), dev.abs_jumps)
        self.assertGreater(len(speeds), 400, "a 0.43 pan must take ~2 min (slow cinematic), not jump")


    def test_mid_arc_target_has_no_visible_finishing_jump(self):
        dev = FakeSunbaPTZ(pan=0.30, zoom=0.013)
        dev.coast = 0.11
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        now = [0.0]
        with patch.object(config, "PTZ_MOVE_MODE", "auto"):
            mc = MotionController(cam, clock=lambda: now[0])
        mc.absolute_ok = True
        mc.set_target(pan=0.55, tilt=-1.0)
        for _ in range(1200):
            now[0] += 0.125
            dev.advance(0.125)
            if mc.step(cam.get_pose()) == "arrived":
                break
        self.assertTrue(all(j < 0.006 for j in dev.abs_jumps), f"visible jump: {dev.abs_jumps}")
        self.assertLess(abs(dev.pan - 0.55), 0.004)


    def test_frozen_control_loop_cannot_push_camera_out_of_the_arc(self):
        """17/10 field case: the AI froze the PTZ loop ~3 s while gliding to the
        water-hole and the camera ended at 0.749 (arc edge 0.735)."""
        high = config.ALLOWED_PAN_RANGE[1]
        for freeze_at in (0.66, 0.68, 0.69, 0.70, 0.71):
            dev = FakeSunbaPTZ(pan=0.55, zoom=0.013)
            dev.coast = 0.11
            cam = make_camera(dev)
            self.addCleanup(cam._loop.close)
            now = [0.0]
            with patch.object(config, "PTZ_MOVE_MODE", "auto"):
                mc = MotionController(cam, clock=lambda: now[0])
            mc.absolute_ok = True
            mc.set_target(pan=config.WATER_HOLE_POSITION["pan"], tilt=-1.0)
            frozen = 0
            for _ in range(600):
                now[0] += 0.125
                dev.advance(0.125)
                if dev.pan >= freeze_at and frozen < 24:      # 3 s with no control at all
                    frozen += 1
                    continue
                if mc.step(cam.get_pose()) == "arrived":
                    break
            for _ in range(40):
                dev.advance(0.125)
            self.assertLessEqual(dev.max_pan, high, f"freeze at {freeze_at}: reached {dev.max_pan:.4f}")


    def test_wrong_pan_sign_triggers_safety_stop(self):
        """If commands ever move the camera AWAY from the target, stop at once."""
        dev = FakeSunbaPTZ(pan=0.50, zoom=0.013)
        cam = make_camera(dev, sign=+1)            # WRONG sign for this mirrored camera
        self.addCleanup(cam._loop.close)
        now = [0.0]
        with patch.object(config, "PTZ_MOVE_MODE", "auto"):
            mc = MotionController(cam, clock=lambda: now[0])
        mc.absolute_ok = True
        mc.set_target(pan=0.65, tilt=-1.0)
        results = []
        for _ in range(200):
            now[0] += 0.125
            dev.advance(0.125)
            results.append(mc.step(cam.get_pose()))
        for _ in range(16):
            dev.advance(0.125)
        self.assertIsNotNone(mc.fault)
        self.assertEqual(results[-1], "stalled")
        self.assertGreater(dev.pan, 0.50 - 0.015 - 0.06, "stopped within a second of going the wrong way")


class SmartZoomTests(TempCapsMixin, unittest.TestCase):
    def _patrol_at_scan_station(self, candidates):
        dev = FakeSunbaPTZ(pan=0.6245, zoom=0.013)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        phases = []
        for _ in range(int(120 / 0.125)):
            now[0] += 0.125
            dev.advance(0.125)
            patrol.candidates_pending = candidates
            patrol.tick(pose=cam.get_pose())
            if patrol.phase in OBSERVE_PHASES:
                patrol.note_result(patrol.observe_epoch)
            phases.append((patrol.station["name"], patrol.phase))
            if len({n for n, _ in phases}) > 2:
                break
        return phases

    def test_plain_scan_station_is_not_zoomed_into_bushes(self):
        phases = self._patrol_at_scan_station(candidates=0)
        self.assertNotIn(("scan_2", "observe_zoom"), phases)
        self.assertIn(("scan_2", "observe_wide"), phases)

    def test_station_with_a_candidate_is_zoomed(self):
        phases = self._patrol_at_scan_station(candidates=1)
        self.assertIn(("scan_2", "zoom"), phases)


class HoldPositionTests(TempCapsMixin, unittest.TestCase):
    def test_camera_moved_by_itself_during_water_watch_is_re_aimed(self):
        w = config.WATER_HOLE_POSITION
        dev = FakeSunbaPTZ(pan=w["pan"], zoom=w["zoom"])
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        moved = False
        for _ in range(int(120 / 0.125)):
            now[0] += 0.125
            dev.advance(0.125)
            patrol.tick(pose=cam.get_pose())
            if patrol.phase in OBSERVE_PHASES:
                patrol.note_result(patrol.observe_epoch)
            if patrol.phase == "observe_zoom" and not moved:
                dev.pan, dev.zoom = w["pan"] + 0.0021, 0.176     # the camera's own park action
                moved = True
            if moved and patrol.phase == "observe_zoom" and abs(dev.zoom - w["zoom"]) < 0.01:
                break
        self.assertTrue(moved)
        self.assertLess(abs(dev.zoom - w["zoom"]), 0.01, "zoom restored to the water-hole")


class PresetPatrolTests(TempCapsMixin, unittest.TestCase):
    """17/10: the plain and the water-hole are above tilt -1.0 - only presets reach them."""

    def _setup(self):
        dev = FakeSunbaPTZ(pan=0.55, zoom=0.03)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        presets = {"water_hole": {"token": "1", "pan": 0.723, "zoom": 0.2193}}
        dev.presets["1"] = (0.723, 0.2193, 0.9)
        for i, (name, pan) in enumerate((("scan_2", 0.6244), ("scan_3", 0.5559), ("scan_4", 0.4874),
                                         ("scan_5", 0.4189), ("scan_6", 0.3503), ("scan_7", 0.2818)), start=2):
            presets[name] = {"token": str(i), "pan": pan, "zoom": 0.03}
            dev.presets[str(i)] = (pan, 0.03, 0.5)
        path = os.path.join(self.tmp.name, "presets.json")
        with open(path, "w") as f:
            json.dump(presets, f)
        p = patch.object(config, "PTZ_PRESETS_FILE", path)
        p.start()
        self.addCleanup(p.stop)
        return dev, cam

    def test_every_station_is_framed_by_its_preset_and_water_gets_close_zoom(self):
        dev, cam = self._setup()
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        observed = {}
        max_water_zoom = 0.0
        for _ in range(int(2400 / 0.125)):
            now[0] += 0.125
            dev.advance(0.125)
            patrol.tick(pose=cam.get_pose())
            if patrol.phase in OBSERVE_PHASES:
                patrol.note_result(patrol.observe_epoch)
                observed.setdefault(patrol.station["name"], set()).add(round(dev.hidden_tilt, 2))
                if patrol.station["name"] == "water_hole":
                    max_water_zoom = max(max_water_zoom, dev.zoom)
            if len(observed) == 7 and max_water_zoom > 0.29:
                break
        self.assertEqual(set(observed), {"water_hole", "scan_2", "scan_3", "scan_4", "scan_5", "scan_6", "scan_7"})
        # every station was observed with its raised (preset) tilt, never at the -1.0 floor
        self.assertTrue(all(0.0 not in tilts for tilts in observed.values()), observed)
        self.assertGreater(max_water_zoom, config.WATER_HOLE_CLOSE_ZOOM - 0.01)
        self.assertIn("1", dev.preset_calls)

    def test_water_hole_zoom_glides_slowly_onto_the_centred_preset(self):
        """18/10: zoom in/out was too fast - glide via the water_wide preset."""
        dev, cam = self._setup()
        with open(config.PTZ_PRESETS_FILE) as f:
            presets = json.load(f)
        presets["water_wide"] = {"token": "8", "pan": 0.723, "zoom": 0.03}
        dev.presets["8"] = (0.723, 0.03, 0.9)
        with open(config.PTZ_PRESETS_FILE, "w") as f:
            json.dump(presets, f)
        dev.pan = 0.64
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        trace = []
        for _ in range(int(2400 / 0.125)):
            now[0] += 0.125
            dev.advance(0.125)
            patrol.tick(pose=cam.get_pose())
            if patrol.phase in OBSERVE_PHASES:
                patrol.note_result(patrol.observe_epoch)
            trace.append((now[0], patrol.station["name"], patrol.phase, dev.zoom, dev.hidden_tilt))
            if (patrol.station["name"] != "water_hole" and patrol.phase == "travel"
                    and any(p == "observe_zoom" for _, _, p, _, _ in trace)):
                break
        self.assertIn("8", dev.preset_calls)                 # wide framing of the water first
        self.assertIn("1", dev.preset_calls)                 # the centred preset finishes the zoom
        self.assertLess(dev.preset_calls.index("8"), dev.preset_calls.index("1"))
        t_start = next(t for t, n, p, z, _ in trace if n == "water_hole" and p == "zoom")
        t_in = next(t for t, n, p, z, _ in trace if t > t_start and z >= 0.21)
        self.assertGreater(t_in - t_start, 4.5)               # ~6 s, not a ~1 s jump
        zooms = [(t, z) for t, n, p, z, _ in trace if t_start <= t <= t_in]
        for (t0, z0), (t1, z1) in zip(zooms, zooms[1:]):      # never a big magnification jump
            self.assertLess((1 + 95 * z1) / (1 + 95 * z0), 1.25)
        watch = [h for _, n, p, _, h in trace if n == "water_hole" and p == "observe_zoom"]
        self.assertTrue(watch and all(h == 0.9 for h in watch))
        # zoom-out before leaving is a glide as well
        t_obs = max(t for t, n, p, z, _ in trace if n == "water_hole" and p == "observe_zoom")
        t_w = next(t for t, n, p, z, _ in trace if t > t_obs and p == "widen")
        t_out = next(t for t, n, p, z, _ in trace if t > t_w and z <= config.PATROL_SWEEP_ZOOM + 0.005)
        self.assertGreater(t_out - t_w, 4.5)

    def test_two_rows_out_along_the_plain_back_along_the_low_row(self):
        """19/10: baboons rest on the near ground below the plain views."""
        dev, cam = self._setup()
        with open(config.PTZ_PRESETS_FILE) as f:
            presets = json.load(f)
        low = {"low_2": 0.6244, "low_3": 0.5559, "low_4": 0.4874, "low_5": 0.4189, "baboon_spot": 0.345, "low_7": 0.2818}
        for i, (name, pan) in enumerate(low.items(), start=20):
            presets[name] = {"token": str(i), "pan": pan, "zoom": 0.03, "tilt": -0.85}
            dev.presets[str(i)] = (pan, 0.03, -0.3)          # hidden tilt marker of the low row
        with open(config.PTZ_PRESETS_FILE, "w") as f:
            json.dump(presets, f)
        extra = {k: 5.0 for k in low}
        with patch.object(config, "PATROL_EXTRA_STATIONS", extra):
            now = [0.0]
            patrol = ObservationPatrol(cam, clock=lambda: now[0])
            seen = []
            for _ in range(int(3000 / 0.125)):
                now[0] += 0.125
                dev.advance(0.125)
                patrol.tick(pose=cam.get_pose())
                if patrol.phase in OBSERVE_PHASES:
                    patrol.note_result(patrol.observe_epoch)
                    name = patrol.station["name"]
                    if not seen or seen[-1][0] != name:
                        seen.append((name, round(dev.hidden_tilt, 2)))
                if len(seen) >= 15:
                    break
        names = [n for n, _ in seen]
        i7 = names.index("scan_7")
        self.assertEqual(names[i7 + 1:i7 + 7], ["low_7", "baboon_spot", "low_5", "low_4", "low_3", "low_2"])
        self.assertEqual(names[i7 + 7], "water_hole")
        self.assertTrue(all(t == -0.3 for n, t in seen if n in low))           # low views use their presets
        self.assertTrue(all(t != -0.3 for n, t in seen if n.startswith("scan")))

    def test_preset_refused_when_far_away(self):
        dev, cam = self._setup()
        self.assertFalse(cam.goto_preset_safe("1", 0.723, pose=(0.40, -1.0, 0.03)))
        self.assertTrue(cam.goto_preset_safe("1", 0.723, pose=(0.715, -1.0, 0.03)))


class SweepIntelligenceTests(TempCapsMixin, unittest.TestCase):
    """19/10: a buffalo seen while sweeping was found, locked - and lost 16 s later."""

    def _det(self, x, conf=0.6):
        return {"box": [x, 770, x + 50, 800], "class_id": 0, "det_confidence": conf,
                "species": "unidentified", "species_confidence": 0.2, "species_trusted": True,
                "source": "custom_yolo", "site_model_active": True}

    def test_camera_motion_is_compensated_between_results(self):
        t = MultiObjectTracker()
        t.update([self._det(253)], 1920, 1080, pose=(0.646, -1.0, 0.03))
        out = t.update([self._det(652)], 1920, 1080, pose=(0.6644, -1.0, 0.03))   # measured pair
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["track_hits"], 2)
        self.assertTrue(out[0]["track_confirmed"])

    def test_without_pose_a_big_jump_is_a_new_object(self):
        t = MultiObjectTracker()
        t.update([self._det(100)], 1920, 1080)
        out = t.update([self._det(900)], 1920, 1080)
        self.assertEqual(out[0]["track_hits"], 1)

    def test_sweep_pauses_for_a_candidate_once_then_continues(self):
        dev = FakeSunbaPTZ(pan=0.30, zoom=0.03)
        cam = make_camera(dev)
        self.addCleanup(cam._loop.close)
        now = [0.0]
        patrol = ObservationPatrol(cam, clock=lambda: now[0])
        phases = []
        for i in range(int(400 / 0.125)):
            now[0] += 0.125
            dev.advance(0.125)
            if patrol.phase == "travel" and 0.40 < dev.pan < 0.405 and not patrol._verify_sites:
                # a box the classifier names (main.worth_a_closer_look) - 21/10 contract
                patrol.candidates_pending, patrol.candidate_best, patrol.travel_candidate_best = 1, 0.45, 0.45
            elif patrol.phase == "verify" and now[0] - patrol.since > 3:
                patrol.candidates_pending, patrol.candidate_best, patrol.travel_candidate_best = 0, 0.0, 0.0  # rejected
            patrol.tick(pose=cam.get_pose())
            if patrol.phase in OBSERVE_PHASES:
                patrol.note_result(patrol.observe_epoch)
            phases.append((now[0], patrol.phase, dev.pan))
        verify = [(t, pan) for t, ph, pan in phases if ph == "verify"]
        self.assertTrue(verify)
        self.assertLess(verify[-1][0] - verify[0][0], config.PATROL_VERIFY_SECONDS + 0.5)
        self.assertLess(max(p for _, p in verify) - min(p for _, p in verify), 0.01)   # camera held still
        after = [ph for t, ph, _ in phases if t > verify[-1][0]]
        self.assertIn("travel", after)                                                # sweep resumed
        self.assertEqual(len(patrol._verify_sites), 1)


class OverlayTests(unittest.TestCase):
    """19/10: boxes hardly ever showed; candidates only in the local preview."""

    def test_candidates_drawn_in_preview_only(self):
        from main import draw_overlay
        tracker = SimpleNamespace(locked_track_id=None)
        cand = {"box": [800, 500, 900, 580], "class_id": 0, "det_confidence": 0.35,
                "species": "unidentified", "track_confirmed": False, "track_id": 7}
        stream = draw_overlay(np.zeros((1080, 1920, 3), np.uint8), [cand], tracker, "PATROL")
        preview = draw_overlay(np.zeros((1080, 1920, 3), np.uint8), [cand], tracker, "PATROL", show_candidates=True)
        self.assertEqual(int(stream[500:582, 798:902].sum()), 0)
        self.assertGreater(int(preview[500:582, 798:902].sum()), 0)

    def test_boxes_follow_the_camera(self):
        from tracking.multi_object_tracker import camera_shift, shift_box
        dx, k, _dy = camera_shift((0.646, -1.0, 0.03), (0.6644, -1.0, 0.03), 1920)
        box = shift_box([253, 766, 308, 796], dx, k, 1920, 1080)
        self.assertAlmostEqual(box[0], 620, delta=60)          # measured buffalo: 253 -> 652
        self.assertIsNone(camera_shift((0.6, -1, 0.03), (-0.6, -1, 0.03), 1920))   # glitched read


class FakeLens:
    """Focus lens + scene: blur grows with |focus - peak|."""

    def __init__(self, peak=0.196, start=0.196):
        rng = np.random.default_rng(1)
        self.scene = (rng.random((540, 960)) * 255).astype(np.uint8)
        self.scene = cv2.cvtColor(cv2.GaussianBlur(self.scene, (0, 0), 1.2), cv2.COLOR_GRAY2BGR)
        self.peak, self.pos, self.auto = peak, start, True
        self.focus_available = True

    def frame(self):
        sigma = 0.3 + abs(self.pos - self.peak) * 400
        return cv2.GaussianBlur(self.scene, (0, 0), sigma)

    def get_focus_info(self):
        return {}

    def set_autofocus(self, enabled=True):
        self.auto = enabled          # like the real lens: switching does NOT refocus
        return True

    def get_focus_position(self):
        return self.pos

    def focus_absolute(self, position, speed=1.0):
        if self.auto:
            return False
        self.pos = position
        return True


class FocusTests(unittest.TestCase):
    def _controller(self, lens):
        t = [0.0]

        def sleep(dt):
            t[0] += dt
        fc = FocusController(lens, lens.frame, pose_source=lambda: (0.7275, -1.0, 0.145),
                             clock=lambda: t[0], sleep=sleep)
        return fc, t

    def test_soft_lens_is_searched_and_locked_sharp(self):
        lens = FakeLens()
        fc, _ = self._controller(lens)
        fc._run("seed")                                 # learns the sharp reference
        lens.pos = 0.196 + 0.012                        # AF left it soft
        fc._run("zoom")
        self.assertLess(abs(lens.pos - 0.196), 0.004)
        self.assertTrue(fc.locked)
        self.assertFalse(lens.auto)
        fc.prepare_for_motion()                         # next zoom: AF back on
        self.assertTrue(lens.auto)
        self.assertFalse(fc.locked)

    def test_failed_search_puts_lens_back_where_it_was(self):
        lens = FakeLens()
        fc, _ = self._controller(lens)
        fc._run("seed")
        lens.scene = cv2.GaussianBlur(lens.scene, (0, 0), 6)   # scene itself went soft (haze)
        fc._run("zoom")
        self.assertAlmostEqual(lens.pos, 0.196, places=6)
        self.assertTrue(lens.auto)

    def test_frame_gate_rejects_blur_and_forgets_slowly(self):
        lens = FakeLens()
        fc, t = self._controller(lens)
        pose = (0.7275, -1.0, 0.145)
        self.assertTrue(fc.frame_is_sharp(lens.frame(), pose))
        lens.pos = 0.23
        self.assertFalse(fc.frame_is_sharp(lens.frame(), pose))
        t[0] += 3600                                    # an hour later the reference has faded
        self.assertTrue(fc.frame_is_sharp(lens.frame(), pose))

    def test_sharpness_ignores_none(self):
        self.assertEqual(sharpness(None), 0.0)


class ClosedSetClassifierTests(unittest.TestCase):
    def det(self, trusted, conf=0.5, species="zebra", sconf=0.83, box=(100, 100, 160, 150)):
        return {"box": list(box), "class_id": 0, "det_confidence": conf, "species": species,
                "species_confidence": sconf, "species_trusted": trusted}

    def test_untrusted_label_never_verifies(self):
        self.assertFalse(is_verified(self.det(False)))
        self.assertTrue(is_verified(self.det(True)))
        self.assertEqual(species_label(self.det(False)), "zebra?")
        self.assertEqual(species_label(self.det(True)), "zebra")
        self.assertEqual(species_label(self.det(False, species="unidentified")), "unidentified")

    def test_ai4g_zebra_on_a_stump_is_not_confirmed_instantly(self):
        trk = MultiObjectTracker()
        out = trk.update([self.det(False, conf=0.42)], 1920, 1080)
        self.assertFalse(out[0]["track_confirmed"])
        trk2 = MultiObjectTracker()
        out2 = trk2.update([self.det(True, conf=0.42)], 1920, 1080)
        self.assertTrue(out2[0]["track_confirmed"], "a trusted species is still believed at once")


if __name__ == "__main__":
    unittest.main()
