"""Pictures of one camera view, for offline detector checks.  The tracker must be stopped.

    .venv313\\Scripts\\python.exe tools\\capture_view.py --pan 0.2818 --tilt -0.91 ^
        --zooms 0.0908 0.15 0.03 --out eval\\captures\\20261023_scan7

23/10 09:28: the operator saw far gazelles in a 9.6x view where MegaDetector and
the site model found nothing, and no picture of that view was kept (only frames
with a candidate are saved).  This moves the camera with goto_position (measured
pan sign, allowed arc, never into a privacy sector), waits until the pose stops
changing, and saves <out>/z<zoom>_<k>.jpg plus poses.json.  It refuses to run
while main.py runs: two programs must never steer one camera.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from camera.onvif_camera import OnvifPTZCamera  # noqa: E402
from main import add_rtsp_credentials  # noqa: E402
from tools.tracker_watchdog import TRACKER, python_processes, scripts  # noqa: E402
from utils.privacy import in_privacy_zone  # noqa: E402


def wait_still(cam, timeout=20.0):
    """Pose read every 0.7 s until two reads agree (the SUNBA ignores move speeds)."""
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        pose = cam.get_pose()
        if pose and last and all(abs(a - b) < 0.002 for a, b in zip(pose, last)):
            return pose
        last = pose
        time.sleep(0.7)
    return last


def grab(uri, n, gap=1.0, warm=30):
    cap = cv2.VideoCapture(uri)
    frames = []
    try:
        for _ in range(warm):                 # drop buffered frames from before the move
            cap.read()
        t = 0.0
        while len(frames) < n:
            ok, f = cap.read()
            if ok and time.time() - t >= gap:
                frames.append(f)
                t = time.time()
    finally:
        cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pan", type=float, required=True)
    ap.add_argument("--tilt", type=float, required=True)
    ap.add_argument("--zooms", type=float, nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--focus-seconds", type=float, default=3.0, help="autofocus time after each zoom")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if any(TRACKER in scripts(c) for _, c in python_processes()):
        print("main.py is running - stop the tracker first (two programs must not steer the camera)")
        return 1
    for z in a.zooms:
        if in_privacy_zone((a.pan, a.tilt, z)):
            print(f"pose ({a.pan}, {a.tilt}, {z}) is inside a privacy sector - refused")
            return 1
    os.makedirs(a.out, exist_ok=True)
    cam = OnvifPTZCamera(config.CAMERA_IP, config.CAMERA_ONVIF_PORT, config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    uri = add_rtsp_credentials(cam.get_stream_uri(), config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    log = []
    try:
        for z in a.zooms:
            cam.goto_position(a.pan, a.tilt, z)
            time.sleep(1.0)
            pose = wait_still(cam)
            time.sleep(a.focus_seconds)
            frames = grab(uri, a.frames)
            for k, f in enumerate(frames):
                name = f"z{z:.4f}_{k}.jpg"
                cv2.imwrite(os.path.join(a.out, name), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
                log.append({"file": name, "asked": [a.pan, a.tilt, z], "pose": pose,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"zoom {z}: pose {pose}, {len(frames)} frames")
    finally:
        with open(os.path.join(a.out, "poses.json"), "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=1)
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
