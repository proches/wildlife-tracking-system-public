"""Collect training frames from THIS camera while it patrols (no AI needed).

    python training/harvest_patrol.py --minutes 30 --out training_data/site/images

The camera runs the normal cinematic patrol (all stations + water-hole, wide
and zoomed).  In every observe phase it saves a few SHARP frames, spaced in
time.  Most of them contain only grass, branches, bushes, water and shadows:
after review they become the "hard negatives" that teach the detector that
vegetation is not an animal - and any real animal or bird gets labelled.

Stop main.py first: only one program may steer the camera.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from camera.focus import sharpness
from camera.onvif_camera import OnvifPTZCamera
from main import FrameGrabber, add_rtsp_credentials
from utils.imageio import write_jpeg
from tracking.observation_patrol import OBSERVE_PHASES, ObservationPatrol
from utils.runtime_metrics import RuntimeMetrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--out", default="training_data/site/images")
    ap.add_argument("--per-phase", type=int, default=2, help="frames saved per observe phase")
    ap.add_argument("--dwell", type=float, default=6.0, help="seconds per observe phase")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    config.SMART_PATROL_WIDE_SECONDS = a.dwell
    config.SMART_PATROL_ZOOM_SECONDS = a.dwell
    config.WATER_HOLE_DWELL_SECONDS = a.dwell * 2
    cam = OnvifPTZCamera(config.CAMERA_IP, config.CAMERA_ONVIF_PORT, config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    if config.RTSP_TRANSPORT:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{config.RTSP_TRANSPORT}")
    import threading
    stop = threading.Event()
    grabber = FrameGrabber(add_rtsp_credentials(cam.get_stream_uri(), config.CAMERA_USERNAME, config.CAMERA_PASSWORD),
                           stop, RuntimeMetrics())
    grabber.start()
    patrol = ObservationPatrol(cam)
    saved, epoch_saves, last_save, best = 0, {}, 0.0, {}
    t_end = time.time() + a.minutes * 60
    try:
        while time.time() < t_end:
            pose = cam.get_pose()
            patrol.tick(pose=pose)
            if patrol.phase in OBSERVE_PHASES:
                patrol.note_result(patrol.observe_epoch)
                frame = grabber.frame()
                ep = patrol.observe_epoch
                if (frame is not None and pose is not None and epoch_saves.get(ep, 0) < a.per_phase
                        and time.time() - last_save >= a.dwell / max(1, a.per_phase)):
                    key = (round(pose[0], 2), round(pose[2], 2))
                    s = sharpness(frame)
                    if s >= 0.5 * best.get(key, 0.0):
                        best[key] = max(best.get(key, 0.0), s)
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                        name = f"{stamp}_site_{patrol.station['name']}_{patrol.phase}.jpg"
                        write_jpeg(os.path.join(a.out, name), frame, 93)
                        epoch_saves[ep] = epoch_saves.get(ep, 0) + 1
                        saved += 1
                        last_save = time.time()
            time.sleep(0.125)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cam.stop()
        w = config.WATER_HOLE_POSITION
        cam.goto_position(w["pan"], w["tilt"], zoom=w["zoom"])
        time.sleep(3)
        cam.close()
        print(f"Saved {saved} frames to {a.out}; camera returned to the water-hole.")


if __name__ == "__main__":
    main()
