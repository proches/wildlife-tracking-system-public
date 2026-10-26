"""Collect training images from video files and/or the live camera.

    python training/harvest_frames.py --video logs/highlights/*.mp4 --out dataset_raw/images --every 2
    python training/harvest_frames.py --live --minutes 120 --out dataset_raw/images --every 20

* Skips near-duplicate frames (perceptual hash) so you do not label 500 copies
  of the same zebra.
* Filenames carry a timestamp, so build_dataset.py can split train/val by time
  (no leakage between neighbouring frames).
* --live reads the RTSP stream; stop the main tracker first or run it on a
  different machine - the camera allows few RTSP clients.
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.imageio import write_jpeg  # noqa: E402


def dhash(frame, size=16):
    g = cv2.cvtColor(cv2.resize(frame, (size + 1, size)), cv2.COLOR_BGR2GRAY)
    return (g[:, 1:] > g[:, :-1]).flatten()


def too_similar(h, hashes, max_bits=10):
    return any((h != o).sum() <= max_bits for o in hashes[-40:])


def save(frame, out, stamp, hashes, force=False):
    h = dhash(frame)
    if not force and too_similar(h, hashes):
        return False
    hashes.append(h)
    write_jpeg(os.path.join(out, f"{stamp}.jpg"), frame, 93)
    return True


def from_video(path, out, every, hashes):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps * every))
    base = datetime.fromtimestamp(os.path.getmtime(path))
    n = saved = 0
    name = Path(path).stem[:24]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            ts = base.timestamp() + n / fps
            stamp = datetime.fromtimestamp(ts).strftime("%Y%m%dT%H%M%S") + f"_{name}_{n}"
            saved += save(frame, out, stamp, hashes)
        n += 1
    cap.release()
    print(f"{path}: {saved} frames kept")


def from_live(out, minutes, every, hashes):
    import config
    from camera.onvif_camera import OnvifPTZCamera
    from main import add_rtsp_credentials
    cam = OnvifPTZCamera(config.CAMERA_IP, config.CAMERA_ONVIF_PORT, config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    uri = add_rtsp_credentials(cam.get_stream_uri(), config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    cap = cv2.VideoCapture(uri)
    end = time.time() + minutes * 60
    last = 0.0
    saved = 0
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue
        if time.time() - last >= every:
            pose = cam.get_pose()
            tag = "" if pose is None else f"_p{pose[0]:.3f}_z{pose[2]:.3f}"
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S") + tag
            saved += save(frame, out, stamp, hashes)
            last = time.time()
    cap.release()
    cam.close()
    print("live: kept", saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="*", default=[])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--every", type=float, default=5.0, help="seconds between kept frames")
    ap.add_argument("--out", default="dataset_raw/images")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hashes = []
    for pattern in a.video:
        for p in glob.glob(pattern):
            from_video(p, a.out, a.every, hashes)
    if a.live:
        from_live(a.out, a.minutes, a.every, hashes)


if __name__ == "__main__":
    main()
