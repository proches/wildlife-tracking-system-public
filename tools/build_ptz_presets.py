"""Store every patrol station as a preset INSIDE the camera.

    .venv313\\Scripts\\python.exe tools\\build_ptz_presets.py            # scan stations
    .venv313\\Scripts\\python.exe tools\\build_ptz_presets.py --water    # also save the CURRENT view as the water-hole

Why: the plain and the water-hole lie above tilt -1.0, which this camera can
neither report nor be sent.  A preset keeps the real motor position.  For each
scan station: AbsoluteMove to (pan, -1.0, PATROL_SWEEP_ZOOM), tilt up for
STATION_TILT_UP_SECONDS[name] (0.4-s steps, speed 0.3 - as calibrated), SetPreset.
Writes ptz_presets.json and a contact sheet ptz_presets.jpg to check by eye.
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
from tracking.observation_patrol import ObservationPatrol  # noqa: E402
from utils.imageio import write_jpeg  # noqa: E402


def grab(uri):
    cap = cv2.VideoCapture(uri)
    frame = None
    for _ in range(12):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


def tilt_up(cam, seconds):
    remaining = seconds
    while remaining > 1e-6:
        step = min(0.4, remaining)
        cam.velocity_move(0, 0.3, 0, pose=cam.get_pose(), timeout=1)
        time.sleep(step)
        cam.stop()
        time.sleep(1.2)
        remaining -= step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--water", action="store_true", help="save the current view as the water-hole preset")
    a = ap.parse_args()
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{config.RTSP_TRANSPORT}")
    cam = OnvifPTZCamera(config.CAMERA_IP, config.CAMERA_ONVIF_PORT, config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    uri = add_rtsp_credentials(cam.get_stream_uri(), config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    try:
        with open(config.PTZ_PRESETS_FILE, encoding="utf-8") as fh:
            presets = json.load(fh)
    except (OSError, ValueError):
        presets = {}
    tiles = []
    try:
        if a.water:
            pose = cam.get_pose()
            token = cam.set_preset("MeliaWaterHole")
            presets["water_hole"] = {"token": str(token), "pan": round(pose[0], 4), "zoom": round(pose[2], 4)}
            print("water_hole preset", presets["water_hole"])
        patrol = ObservationPatrol(cam)
        for st in patrol.stations:
            name = st["name"]
            if name not in config.STATION_TILT_UP_SECONDS:
                continue
            cam.goto_position(st["pan"], -1.0, zoom=config.PATROL_SWEEP_ZOOM)
            time.sleep(6)
            tilt_up(cam, config.STATION_TILT_UP_SECONDS[name])
            pose = cam.get_pose()
            token = cam.set_preset(f"Melia_{name}")
            presets[name] = {"token": str(token), "pan": round(pose[0], 4), "zoom": round(pose[2], 4)}
            print(name, presets[name])
            f = grab(uri)
            if f is not None:
                im = cv2.resize(f, (480, 270))
                cv2.putText(im, f"{name} token {token}", (8, 255), 0, 0.7, (0, 0, 255), 2)
                tiles.append(im)
    finally:
        with open(config.PTZ_PRESETS_FILE, "w", encoding="utf-8") as fh:
            json.dump(presets, fh, indent=2)
        print("Saved", config.PTZ_PRESETS_FILE)
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            write_jpeg("ptz_presets.jpg", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
        w = presets.get("water_hole")
        if w:
            cam.goto_position(w["pan"], -1.0, zoom=config.PATROL_SWEEP_ZOOM)
            time.sleep(5)
            cam.goto_preset_safe(w["token"], w["pan"])
            time.sleep(5)
        cam.close()


if __name__ == "__main__":
    main()
