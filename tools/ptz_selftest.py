"""Find out - in about a minute - what this camera's PTZ actually does.

    python tools/ptz_selftest.py            # pan direction, absolute move, zoom, focus
    python tools/ptz_selftest.py --no-zoom  # skip zoom tests

Order matters for safety:

1. Zoom out (sign-independent), then a short, slow ContinuousMove toward the
   middle of the arc tells us the PAN DIRECTION (an EFlip / upside-down camera executes pan commands mirrored).
   Before v3.1 this was never checked, and an AbsoluteMove to a "safe" pan
   could swing the camera to the mirror position - the hotel side.
2. Only with a MEASURED direction is an AbsoluteMove ever tried.
3. Focus service.
4. The camera is put back where it started (absolute zoom is verified there).

Every pose is read through the glitch filter, and the test aborts (stop) if the
camera ever reports a pan outside ALLOWED_PAN_RANGE.  The verdict is saved to
logs/ptz_capabilities.json (pan_command_sign, absolute_move_works, ...); main.py
uses it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from camera.onvif_camera import OnvifPTZCamera


class Abort(Exception):
    pass


def settle(cam, seconds):
    low, high = config.ALLOWED_PAN_RANGE
    t0 = time.time()
    last = None
    while time.time() - t0 < seconds:
        time.sleep(0.3)
        last = cam.get_pose()
        if last is not None and not (low - 0.01 <= last[0] <= high + 0.01):
            cam.stop()
            raise Abort(f"camera reports pan {last[0]:.4f} outside the wildlife arc - stopped")
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zoom", action="store_true")
    args = ap.parse_args()

    cam = OnvifPTZCamera(config.CAMERA_IP, config.CAMERA_ONVIF_PORT, config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    low, high = config.ALLOWED_PAN_RANGE
    report = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    detected_sign = cam.pan_command_sign
    p0 = None
    sign_measured = False
    try:
        p0 = settle(cam, 0.6)
        print("Start pose (pan, tilt, zoom):", p0)
        if p0 is None:
            print("FAIL: camera does not report its position; closed-loop control is impossible.")
            return 1
        if not low <= p0[0] <= high:
            print(f"FAIL: start pan {p0[0]:.4f} is outside ALLOWED_PAN_RANGE {config.ALLOWED_PAN_RANGE}; "
                  "move the camera into the wildlife arc first.")
            return 1

        # 1. zoom out (zoom does not depend on the pan sign).  Pan speed is scaled
        #    by zoom on this camera: at 14.8x a pan pulse barely moves the image.
        pose = p0
        t0 = time.time()
        while pose[2] > 0.03 and time.time() - t0 < 8.0:
            cam.velocity_move(0, 0, -0.5, pose=pose, timeout=0.6)
            pose = settle(cam, 0.3)
        cam.stop()
        pw = settle(cam, 1.0)
        report["continuous_zoom_works"] = abs(pw[2] - p0[2]) > 0.005 or p0[2] <= 0.03
        print(f"ContinuousMove zoom: {p0[2]:.4f} -> {pw[2]:.4f}  -> "
              f"{'OK' if report['continuous_zoom_works'] else 'NO ZOOM'}")

        # 2. pan direction.  Command "toward the middle" using the sign the camera's
        #    EFlip setting predicts, in short pulses, and stop the moment any motion
        #    is seen: if the prediction is wrong the camera moves OUTWARD, so it
        #    must only be allowed to creep a few thousandths.
        direction = -1 if pw[0] > (low + high) / 2 else 1
        cam.pan_command_sign = detected_sign
        p1, t0 = pw, time.time()
        while abs(p1[0] - pw[0]) < 0.004 and time.time() - t0 < 1.5:
            cam.velocity_move(0.30 * direction, 0, 0, pose=p1, edge_margin=config.PATROL_EDGE_MARGIN,
                              timeout=0.4)
            time.sleep(0.1)
            p1 = cam.get_pose() or p1
        cam.stop()
        p1 = settle(cam, 0.8)
        moved = p1[0] - pw[0]
        report["continuous_pan_works"] = abs(moved) > 0.003
        if report["continuous_pan_works"]:
            sign = detected_sign if moved * direction > 0 else -detected_sign
            report["pan_command_sign"] = sign
            sign_measured = True
            print(f"ContinuousMove pan : moved {moved:+.4f}  -> OK, pan commands are "
                  f"{'normal' if sign == 1 else 'MIRRORED (EFlip / upside-down mount)'}"
                  f" (camera EFlip setting predicted {detected_sign:+d})")
        else:
            sign = detected_sign
            print(f"ContinuousMove pan : moved {moved:+.4f}  -> NO MOVEMENT; AbsoluteMove NOT tested "
                  "(it is unsafe without a measured pan direction)")
        cam.pan_command_sign = sign

        # 3. absolute pan, only with a measured sign
        if sign_measured:
            target = max(low + 0.01, min(high - 0.01, p1[0] + 0.04 * direction))
            cam.goto_position(target, p1[1], zoom=None)
            p2 = settle(cam, 4.0)
            report["absolute_move_works"] = abs(target - p2[0]) < 0.006
            print(f"AbsoluteMove pan   : target {target:.4f} -> {p2[0]:.4f}  -> "
                  f"{'OK' if report['absolute_move_works'] else 'NOT PRECISE / IGNORED'}")

        report["focus_service"] = bool(cam.focus_available)
        print("Focus (ONVIF Imaging):", cam.get_focus_info() if cam.focus_available else "not available")
        return 0
    except Abort as exc:
        print("ABORTED:", exc)
        return 2
    finally:
        try:
            cam.stop()
            if p0 is not None and sign_measured and report.get("absolute_move_works"):
                cam.goto_position(p0[0], p0[1], zoom=p0[2])
                back = settle(cam, 6.0)
                report["absolute_zoom_works"] = abs(back[2] - p0[2]) < 0.006
                print(f"Returned to start  : {back}  (absolute zoom "
                      f"{'OK' if report['absolute_zoom_works'] else 'NOT PRECISE'})")
            elif p0 is not None:
                pose = cam.get_pose()
                t0 = time.time()
                while pose and abs(pose[2] - p0[2]) > 0.006 and time.time() - t0 < 10.0:
                    cam.velocity_move(0, 0, 0.3 if p0[2] > pose[2] else -0.3, pose=pose, timeout=0.5)
                    time.sleep(0.15)
                    pose = cam.get_pose()
                cam.stop()
                print("Zoom restored (pan could not be restored without a measured direction):", cam.get_pose())
        except Exception as exc:
            print("Could not return to the start pose:", exc)
        if report.get("absolute_move_works"):
            verdict = "AbsoluteMove verified -> PTZ_MOVE_MODE='auto' will use it"
        elif report.get("continuous_pan_works"):
            verdict = "continuous only -> main.py uses closed-loop ContinuousMove automatically"
        else:
            verdict = "PAN NOT VERIFIED - check ONVIF user PTZ rights / PTZ lock, then run again"
        print("\nVerdict:", verdict)
        old = {}
        try:
            with open(config.PTZ_CAPABILITIES_FILE, encoding="utf-8") as f:
                old = json.load(f)
        except (OSError, ValueError):
            pass
        old.update(report)
        if "absolute_move_works" in report:
            old["decided_at"] = report["checked_at"]
        os.makedirs(os.path.dirname(config.PTZ_CAPABILITIES_FILE) or ".", exist_ok=True)
        with open(config.PTZ_CAPABILITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(old, f, indent=2)
        print("Saved ->", config.PTZ_CAPABILITIES_FILE)
        cam.close()


if __name__ == "__main__":
    sys.exit(main())
