"""Boundary checks with fake ONVIF services; never connect to a camera."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import config
from camera.onvif_camera import OnvifPTZCamera
from tracking.ptz_tracker import PTZTracker


class CameraBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.camera = OnvifPTZCamera.__new__(OnvifPTZCamera)
        self.camera._loop = asyncio.new_event_loop()
        self.addCleanup(self.camera._loop.close)
        self.camera.profile_token = 'test'
        self.camera._pan_tilt_space = 'pan'
        self.camera._zoom_space = 'zoom'
        self.camera._abs_zoom_space = 'abszoom'
        self.camera.ptz = SimpleNamespace(
            create_type=Mock(side_effect=lambda name: SimpleNamespace()),
            ContinuousMove=AsyncMock(), Stop=AsyncMock(), GotoPreset=AsyncMock(), AbsoluteMove=AsyncMock())
        self.position(config.WILDLIFE_CENTER_PAN)

    def position(self, pan):
        self.camera.get_status = Mock(return_value=SimpleNamespace(
            PanTilt=SimpleNamespace(x=pan), Zoom=SimpleNamespace(x=0.012)))

    def test_edges_block_outward_but_allow_inward(self):
        low, high = config.ALLOWED_PAN_RANGE
        for pan in (low, low + config.PAN_BOUNDARY_MARGIN / 2):
            self.position(pan)
            self.assertEqual(self.camera.guard_pan_speed(-0.1), 0)
            self.assertEqual(self.camera.guard_pan_speed(0.1), 0.1)
        for pan in (high, high - config.PAN_BOUNDARY_MARGIN / 2):
            self.position(pan)
            self.assertEqual(self.camera.guard_pan_speed(0.1), 0)
            self.assertEqual(self.camera.guard_pan_speed(-0.1), -0.1)

    def test_center_allows_both_directions_and_caps_speed(self):
        self.assertEqual(self.camera.guard_pan_speed(0.5), config.BOUNDARY_PAN_MAX_SPEED)
        self.assertEqual(self.camera.guard_pan_speed(-0.5), -config.BOUNDARY_PAN_MAX_SPEED)

    def test_invalid_or_outside_position_blocks_both_directions(self):
        low, high = config.ALLOWED_PAN_RANGE
        for pan in (low - config.PAN_RECOVERY_TOLERANCE - 0.001, high + config.PAN_RECOVERY_TOLERANCE + 0.001, -1, 1, float('nan')):
            self.position(pan)
            for speed in (-0.1, 0.1):
                self.assertEqual(self.camera.guard_pan_speed(speed), 0)
        for status in (None, SimpleNamespace(PanTilt=None)):
            self.camera.get_status.return_value = status
            self.assertEqual(self.camera.guard_pan_speed(0.1), 0)

    def test_blocked_move_sends_stop_not_movement(self):
        self.position(config.ALLOWED_PAN_RANGE[0])
        self.camera.move(-0.1)
        self.camera.ptz.ContinuousMove.assert_not_awaited()
        self.camera.ptz.Stop.assert_awaited_once()

    def test_pan_pulse_has_device_timeout_and_explicit_stop(self):
        with patch('camera.onvif_camera.asyncio.sleep', new_callable=AsyncMock) as sleep:
            self.camera.move(0.5)
        request = self.camera.ptz.ContinuousMove.await_args.args[0]
        self.assertEqual(request.Timeout.total_seconds(), config.BOUNDARY_MOVE_TIMEOUT_SECONDS)
        self.assertEqual(request.Velocity['PanTilt']['x'], config.BOUNDARY_PAN_MAX_SPEED)
        sleep.assert_awaited_once_with(config.BOUNDARY_PAN_PULSE_SECONDS)
        self.assertEqual(self.camera.ptz.Stop.await_count, 2)

    def test_float32_pulse_from_optical_flow_is_sent(self):
        """21-23/10: every tracking pulse >= 0.6 s from the optical-flow box was a numpy.float32;
        timedelta() refused it, 640 moves failed and each one stopped the camera (jerky following)."""
        import numpy as np
        with patch('camera.onvif_camera.asyncio.sleep', new_callable=AsyncMock):
            self.camera.move(0.3, 0.25, 0.0, prevalidated=True, pulse_seconds=np.float32(0.8))
        request = self.camera.ptz.ContinuousMove.await_args.args[0]
        self.assertAlmostEqual(request.Timeout.total_seconds(), 1.4, places=3)

    def test_move_error_still_attempts_stop(self):
        self.camera.ptz.ContinuousMove.side_effect = RuntimeError('connection lost')
        with self.assertRaises(RuntimeError):
            self.camera.move(0.1)
        self.camera.ptz.Stop.assert_awaited_once()

    def test_presets_cannot_bypass_boundary(self):
        with self.assertRaises(ValueError):
            self.camera.goto_preset('201')
        self.assertFalse(self.camera.goto_preset_by_name('Wildlife Left'))
        self.camera.ptz.GotoPreset.assert_not_awaited()

    def test_easing_does_not_leave_outward_momentum(self):
        self.position(config.ALLOWED_PAN_RANGE[1])
        tracker = PTZTracker(self.camera)
        tracker._pan_speed = 0.15
        tracker._ease_toward(0, 0, 0)
        self.assertEqual(tracker._pan_speed, 0)
        self.camera.ptz.ContinuousMove.assert_not_awaited()

    def test_search_reverses_into_allowed_arc(self):
        self.position(config.ALLOWED_PAN_RANGE[1])
        tracker = PTZTracker(self.camera)
        with patch('camera.onvif_camera.asyncio.sleep', new_callable=AsyncMock):
            tracker._sweep_search()
        request = self.camera.ptz.ContinuousMove.await_args.args[0]
        self.assertLess(request.Velocity['PanTilt']['x'], 0)
        self.assertEqual(tracker._search_direction, -1)

    def test_small_overshoot_allows_only_slow_inward_recovery(self):
        low, high = config.ALLOWED_PAN_RANGE
        self.position(high + 0.002111137)
        self.assertEqual(self.camera.guard_pan_speed(0.3), 0)
        self.assertEqual(self.camera.guard_pan_speed(-0.3), -config.PAN_RECOVERY_SPEED)
        self.position(low - 0.002)
        self.assertEqual(self.camera.guard_pan_speed(-0.3), 0)
        self.assertEqual(self.camera.guard_pan_speed(0.3), config.PAN_RECOVERY_SPEED)

    def test_zoom_limits_and_missing_position(self):
        self.camera.get_status.return_value.Zoom.x = config.ZOOM_POSITION_MAX
        self.assertEqual(self.camera.guard_zoom_speed(0.1), 0)
        self.assertEqual(self.camera.guard_zoom_speed(-0.1), -0.1)
        self.camera.get_status.return_value.Zoom.x = 0
        self.assertEqual(self.camera.guard_zoom_speed(-0.1), 0)
        self.camera.get_status.return_value = None
        self.assertEqual(self.camera.guard_zoom_speed(0.1), 0)

    def test_zoom_outlasts_pan_but_always_stops(self):
        with patch('camera.onvif_camera.asyncio.sleep', new_callable=AsyncMock) as sleep:
            self.camera.move(0.1, 0, 0.1)
        self.assertAlmostEqual(sum(c.args[0] for c in sleep.await_args_list),
                               config.ZOOM_PULSE_SECONDS)
        stops = self.camera.ptz.Stop.await_args_list
        self.assertFalse(stops[0].args[0]['Zoom'])
        self.assertTrue(stops[-1].args[0]['Zoom'])

    def test_zoom_only_pulse_is_bounded(self):
        with patch('camera.onvif_camera.asyncio.sleep', new_callable=AsyncMock) as sleep:
            self.camera.move(0, 0, 0.1)
        sleep.assert_awaited_once_with(config.ZOOM_PULSE_SECONDS)
        self.camera.ptz.Stop.assert_awaited_once()

    def test_small_centered_target_zooms_but_offcenter_does_not(self):
        tracker = PTZTracker(self.camera)
        animal = {'class_id': 0, 'species': 'bird', 'det_confidence': 0.9,
                  'box': [490, 490, 510, 510]}
        tracker.locked_box = animal['box']
        tracker.pose_hint = (config.WILDLIFE_CENTER_PAN, -1.0, 0.012)
        with patch('camera.onvif_camera.asyncio.sleep', new_callable=AsyncMock):
            tracker.update([animal], 1000, 1000)
        # 21/10: tracking zoom is one bounded AbsoluteMove step (a continuous pulse coasted 26x -> 42x)
        request = self.camera.ptz.AbsoluteMove.await_args.args[0]
        self.assertGreater(request.Position['Zoom']['x'], 0.012)
        self.assertLessEqual(request.Position['Zoom']['x'], config.TRACK_MAX_ZOOM)
        animal['box'] = [700, 490, 720, 510]
        with patch('camera.onvif_camera.asyncio.sleep', new_callable=AsyncMock):
            tracker.update([animal], 1000, 1000)
        request = self.camera.ptz.ContinuousMove.await_args.args[0]
        self.assertEqual(request.Velocity['Zoom']['x'], 0)
        self.assertEqual(self.camera.ptz.AbsoluteMove.await_count, 1, "off-centre: no zoom step")

    def test_lost_target_stops_zoom_immediately(self):
        tracker = PTZTracker(self.camera)
        tracker._zoom_speed = 0.2
        tracker.update([], 1000, 1000)
        self.assertEqual(tracker._zoom_speed, 0)
        self.camera.ptz.Stop.assert_awaited_once()
        self.camera.ptz.ContinuousMove.assert_not_awaited()

    def test_search_widens_view(self):
        self.camera.get_status.return_value.Zoom.x = 0.15
        tracker = PTZTracker(self.camera)
        with patch('camera.onvif_camera.asyncio.sleep', new_callable=AsyncMock):
            tracker._sweep_search()
        request = self.camera.ptz.ContinuousMove.await_args.args[0]
        self.assertLess(request.Velocity['Zoom']['x'], 0)

    def test_zoom_reversal_never_sends_residual_zoom_in(self):
        tracker = PTZTracker(self.camera)
        tracker._zoom_speed = 0.3
        with patch('camera.onvif_camera.asyncio.sleep', new_callable=AsyncMock):
            tracker._ease_toward(0, 0, -0.08)
        request = self.camera.ptz.ContinuousMove.await_args.args[0]
        self.assertLess(request.Velocity['Zoom']['x'], 0)


if __name__ == '__main__':
    unittest.main()
