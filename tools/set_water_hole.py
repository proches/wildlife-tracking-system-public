"""Aim the camera at the water-hole interactively and save that exact pose.

    .venv313\\Scripts\\python.exe tools\\set_water_hole.py

A window shows the live picture with a centre cross.  Keys:
    a / d   pan left / right (small step)      A / D  bigger step
    w / s   zoom in / out                       f      autofocus once more
    ENTER   save -> water_hole.json (used by the patrol from the next start)
    q / ESC quit without saving
Every move stays inside ALLOWED_PAN_RANGE.  Stop the tracker (main.py) first.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import config  # noqa: E402
from camera.onvif_camera import OnvifPTZCamera  # noqa: E402
from main import FrameGrabber, add_rtsp_credentials  # noqa: E402
from utils.runtime_metrics import RuntimeMetrics  # noqa: E402


def main():
    import threading
    cam = OnvifPTZCamera(config.CAMERA_IP, config.CAMERA_ONVIF_PORT, config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{config.RTSP_TRANSPORT}")
    stop = threading.Event()
    grabber = FrameGrabber(add_rtsp_credentials(cam.get_stream_uri(), config.CAMERA_USERNAME, config.CAMERA_PASSWORD),
                           stop, RuntimeMetrics())
    grabber.start()
    low, high = config.ALLOWED_PAN_RANGE
    w = config.WATER_HOLE_POSITION
    pan, tilt, zoom = w["pan"], w["tilt"], w["zoom"]
    cam.goto_position(pan, tilt, zoom=zoom)
    win = "Water-hole: centre the water  (a/d pan, w/s zoom, ENTER save, q quit)"
    saved = False
    try:
        while True:
            frame = grabber.frame()
            if frame is not None:
                view = cv2.resize(frame, (1280, 720))
                cv2.line(view, (620, 360), (660, 360), (0, 0, 255), 2)
                cv2.line(view, (640, 340), (640, 380), (0, 0, 255), 2)
                cv2.rectangle(view, (480, 270), (800, 450), (0, 255, 255), 1)
                p = cam.get_pose() or (pan, tilt, zoom)
                cv2.putText(view, f"pan {p[0]:.4f}  zoom {p[2]:.3f}", (20, 700), 0, 0.8, (0, 255, 0), 2)
                cv2.imshow(win, view)
            k = cv2.waitKey(50) & 0xFF
            step = None
            if k in (ord("a"), ord("d"), ord("A"), ord("D")):
                size = 0.008 if k in (ord("A"), ord("D")) else 0.002
                step = size if k in (ord("a"), ord("A")) else -size   # if reversed, use the other key
                pan = max(low + 0.002, min(high - 0.002, pan + step))
                cam.goto_position(pan, tilt, zoom=zoom)
            elif k in (ord("w"), ord("s")):
                zoom = max(config.ZOOM_SEARCH_POSITION, min(config.ZOOM_POSITION_MAX - config.ZOOM_POSITION_MARGIN,
                                                             zoom + (0.01 if k == ord("w") else -0.01)))
                cam.set_autofocus(True)
                cam.goto_position(pan, tilt, zoom=zoom)
            elif k == ord("f"):
                cam.set_autofocus(True)
            elif k == 13:
                time.sleep(0.8)
                p = cam.get_pose()
                if p and low <= p[0] <= high:
                    data = {"pan": round(p[0], 4), "tilt": round(p[1], 3), "zoom": round(p[2], 4),
                            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    with open("water_hole.json", "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print("Saved water_hole.json:", data)
                    saved = True
                    break
            elif k in (ord("q"), 27):
                break
    finally:
        stop.set()
        cv2.destroyAllWindows()
        cam.close()
    if not saved:
        print("Nothing saved.")


if __name__ == "__main__":
    main()
