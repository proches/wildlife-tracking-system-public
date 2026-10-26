"""Re-centre the water-hole preset on the water, measured from the live image.

    .venv313\\Scripts\\python.exe tools\\center_water_preset.py --shift 160 -80

``--shift SX SY``: how many pixels (1920x1080 frame) the picture content must
move: +SX = water moves right, -SY = water moves up.  The camera is nudged with
short ContinuousMove pulses and the real shift is measured by image
correlation after every pulse, so the camera's mirrored pan / hidden tilt do
not matter.  Then:

* the water-hole preset (same token) is overwritten with the centred view;
* a "water_wide" preset (same pan/tilt, sweep zoom) is saved - the patrol
  glides there and zooms in slowly instead of jumping to the zoomed preset;
* ptz_presets.json, water_hole.json and water_roi.json are updated;
* water_center_before.jpg / water_center_after.jpg show the result.
Stop main.py first.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import config  # noqa: E402
from camera.onvif_camera import OnvifPTZCamera  # noqa: E402
from main import add_rtsp_credentials  # noqa: E402
from utils.imageio import write_jpeg  # noqa: E402

W, H = 1920, 1080


def grab(uri):
    cap = cv2.VideoCapture(uri)
    frame = None
    for _ in range(15):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


def gray(frame):
    g = cv2.cvtColor(cv2.resize(frame, (W // 2, H // 2)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g


def shift_of(ref, frame):
    """Content shift of ``frame`` relative to ``ref`` in full-resolution pixels."""
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref, gray(frame), win)
    return dx * 2.0, dy * 2.0, resp


def pulse(cam, pan, tilt, seconds):
    # the water-hole is near the arc end: tiny nudges may use a small edge margin
    sent = cam.velocity_move(pan, tilt, 0.0, pose=cam.get_pose(), edge_margin=0.004, timeout=1)
    time.sleep(seconds)
    cam.stop()
    time.sleep(1.5)
    return sent


def crosshair(frame, label):
    im = frame.copy()
    cv2.line(im, (W // 2 - 60, H // 2), (W // 2 + 60, H // 2), (0, 0, 255), 2)
    cv2.line(im, (W // 2, H // 2 - 60), (W // 2, H // 2 + 60), (0, 0, 255), 2)
    cv2.putText(im, label, (40, H - 40), 0, 1.2, (0, 0, 255), 3)
    return im


def shift_polygon(poly, sx, sy):
    return [[round(min(1.0, max(0.0, x + sx / W)), 4), round(min(1.0, max(0.0, y + sy / H)), 4)] for x, y in poly]


def scale_polygon(poly, r):
    return [[round(min(1.0, max(0.0, 0.5 + (x - 0.5) * r)), 4), round(min(1.0, max(0.0, 0.5 + (y - 0.5) * r)), 4)]
            for x, y in poly]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", nargs=2, type=float, required=True, metavar=("SX", "SY"))
    ap.add_argument("--tol", type=float, default=14.0)
    a = ap.parse_args()
    sx, sy = a.shift
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{config.RTSP_TRANSPORT}")
    with open(config.PTZ_PRESETS_FILE, encoding="utf-8") as fh:
        presets = json.load(fh)
    water = presets["water_hole"]
    cam = OnvifPTZCamera(config.CAMERA_IP, config.CAMERA_ONVIF_PORT, config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    uri = add_rtsp_credentials(cam.get_stream_uri(), config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    try:
        if not cam.goto_preset_safe(water["token"], water["pan"]):
            print("Camera is not near the water-hole (safety) - aim it there first")
            return 1
        time.sleep(7)
        f0 = grab(uri)
        ref = gray(f0)
        write_jpeg("water_center_before.jpg", crosshair(f0, "before"))
        gain = {"pan": None, "tilt": None}       # measured px per second of pulse (signed)
        speed = {"pan": 0.20, "tilt": 0.20}
        done = False
        for it in range(16):
            dx, dy, resp = shift_of(ref, grab(uri))
            ex, ey = sx - dx, sy - dy
            print(f"iter {it}: shift ({dx:+.0f},{dy:+.0f}) error ({ex:+.0f},{ey:+.0f}) resp {resp:.2f}", flush=True)
            if abs(ex) <= a.tol and abs(ey) <= a.tol:
                done = True
                break
            axis = "pan" if abs(ex) > a.tol and (abs(ex) >= abs(ey) or abs(ey) <= a.tol) else "tilt"
            err = ex if axis == "pan" else ey
            g = gain[axis]
            if g is None:
                secs, sign = 0.4, 1.0
            else:
                sign = 1.0 if err * g > 0 else -1.0
                secs = min(1.0, max(0.12, abs(err / g)))
            before = (dx, dy)
            v = sign * speed[axis]
            sent = pulse(cam, v if axis == "pan" else 0.0, v if axis == "tilt" else 0.0, secs)
            if axis == "pan" and sent[0] == 0.0:
                print("   pan pulse refused by the arc guard (edge) - trying the other direction")
                sign = -sign
                v = sign * speed[axis]
                sent = pulse(cam, v, 0.0, secs)
                if sent[0] == 0.0:
                    print("   pan refused both ways - abort")
                    break
            ndx, ndy, _ = shift_of(ref, grab(uri))
            moved = (ndx - before[0]) if axis == "pan" else (ndy - before[1])
            if abs(moved) < 3:
                speed[axis] = min(0.35, speed[axis] + 0.05)
                print(f"   {axis} pulse moved nothing; speed -> {speed[axis]:.2f}")
                continue
            new_gain = moved / (sign * secs)
            gain[axis] = new_gain if g is None else 0.5 * (g + new_gain)
            print(f"   {axis} {sign * speed[axis]:+.2f} x {secs:.2f}s moved {moved:+.0f}px (gain {gain[axis]:+.0f}px/s)")
        final = grab(uri)
        fdx, fdy, _ = shift_of(ref, final)
        write_jpeg("water_center_after.jpg", crosshair(final, f"after shift ({fdx:+.0f},{fdy:+.0f})"))
        if not done:
            print("Did not converge - presets NOT changed (see water_center_after.jpg)")
            return 2
        pose = cam.get_pose()
        cam.set_preset("MeliaWaterHole", token=water["token"])
        presets["water_hole"] = {"token": str(water["token"]), "pan": round(pose[0], 4), "zoom": round(pose[2], 4)}
        print("water_hole preset overwritten", presets["water_hole"])
        cam.zoom_only(config.PATROL_SWEEP_ZOOM)          # keeps pan and the hidden tilt
        time.sleep(6)
        wpose = cam.get_pose()
        old = presets.get("water_wide", {}).get("token")
        token = cam.set_preset("MeliaWaterWide", token=old)
        presets["water_wide"] = {"token": str(token), "pan": round(wpose[0], 4), "zoom": round(wpose[2], 4)}
        print("water_wide preset", presets["water_wide"])
        with open(config.PTZ_PRESETS_FILE, "w", encoding="utf-8") as fh:
            json.dump(presets, fh, indent=2)
        try:
            with open("water_hole.json", encoding="utf-8") as fh:
                wh = json.load(fh)
            wh["pan"] = round(pose[0], 4)
            with open("water_hole.json", "w", encoding="utf-8") as fh:
                json.dump(wh, fh, indent=2)
        except (OSError, ValueError):
            pass
        try:
            with open(config.WATER_ROI_FILE, encoding="utf-8") as fh:
                roi = json.load(fh)
            mag = lambda z: 1.0 + 95.0 * z   # measured on this camera: 0.03 = 3.85x, 0.2193 = 21.8x
            base = next(z for z in roi["zones"] if z["name"] == "water_preset")
            base["polygon"] = shift_polygon(base["polygon"], fdx, fdy)
            base["pose"]["pan"] = round(pose[0], 4)
            for z in roi["zones"]:
                if z is not base:
                    z["polygon"] = scale_polygon(base["polygon"], mag(z["pose"]["zoom"]) / mag(base["pose"]["zoom"]))
                    z["pose"]["pan"] = round(pose[0], 4)
            with open(config.WATER_ROI_FILE, "w", encoding="utf-8") as fh:
                json.dump(roi, fh, indent=2)
            print("water_roi.json updated")
        except (OSError, ValueError, StopIteration, KeyError) as exc:
            print("water_roi.json not updated:", exc)
        cam.goto_preset_safe(water["token"], pose[0])
        time.sleep(6)
        write_jpeg("water_center_check.jpg", crosshair(grab(uri), "preset recall check"))
        return 0
    finally:
        cam.close()


if __name__ == "__main__":
    sys.exit(main())
