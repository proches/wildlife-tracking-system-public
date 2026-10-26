"""Collect THIS camera's hard examples for the second training round.

    .venv313\\Scripts\\python.exe training\\harvest_round2.py --out training_data/site2/images

19/10/2026: the site model (round 1) scored NOTHING on baboons - near and clear,
resting in shade, walking - while MegaDetector found them (0.42-0.70).  The
frames the running system already saved are exactly the examples it lacks:

* logs/review/*_{candidate,unidentified,water}.jpg  - things the AI half-saw;
* logs/highlights/*.mp4                             - clips of followed animals
  (one frame per --every seconds, near-duplicates skipped);
* training_data/missed/*                            - frames saved with the M key;
* a sample of logs/review/*_negative_sample.jpg     - this camera's empty views
  (hard negatives: bushes, stumps, the lodge wall, water glare).

Files are named <YYYYmmddTHHMMSS>_<source>.jpg so assemble_training_set.py can
split train/val by time.  Labels are made afterwards by training/teacher_label.py.
"""

import argparse
import glob
import os
import random
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from utils.imageio import write_jpeg  # noqa: E402
from utils.privacy import in_privacy_zone  # noqa: E402

_TS = re.compile(r"(\d{8}T\d{6})")


def stamp_of(path):
    m = _TS.search(os.path.basename(path))
    return m.group(1) if m else datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%dT%H%M%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training_data/site2/images")
    ap.add_argument("--since", default="20261017", help="only files from this date (YYYYmmdd) on")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between frames taken from a clip")
    ap.add_argument("--negatives", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(7)
    n = {"review": 0, "clip": 0, "missed": 0, "negative": 0}

    def private(img_path):
        """Never train on (or keep) a view of the lodge pool / guests."""
        import json
        meta = os.path.splitext(img_path)[0] + ".json"
        try:
            pose = json.load(open(meta, encoding="utf-8")).get("pose")
        except (OSError, ValueError):
            return False
        return bool(pose) and (in_privacy_zone(pose) or (0.58 <= float(pose[0]) <= 0.75 and float(pose[1]) > -0.985))

    def keep(src_path, name):
        if private(src_path):
            n["private_skipped"] = n.get("private_skipped", 0) + 1
            return False
        dst = os.path.join(a.out, name)
        if not os.path.exists(dst):
            shutil.copy2(src_path, dst)
            return True
        return False

    for kind in ("candidate", "unidentified", "water", "animal", "motion"):
        for p in sorted(glob.glob(f"logs/review/*_{kind}.jpg")):
            if stamp_of(p)[:8] >= a.since:
                n["review"] += keep(p, f"{stamp_of(p)}_{Path(p).stem.split('_')[1]}_{kind}.jpg")

    negs = [p for p in glob.glob("logs/review/*_negative_sample.jpg") if stamp_of(p)[:8] >= a.since]
    for p in rng.sample(negs, min(a.negatives, len(negs))):
        n["negative"] += keep(p, f"{stamp_of(p)}_{Path(p).stem.split('_')[1]}_negative.jpg")

    for p in sorted(glob.glob("training_data/missed/*")):
        if p.lower().endswith((".jpg", ".png")):
            img = cv2.imread(p)
            if img is not None:
                dst = os.path.join(a.out, f"{stamp_of(p)}_missed.jpg")
                if not os.path.exists(dst):
                    write_jpeg(dst, img)
                    n["missed"] += 1

    for clip in sorted(glob.glob("logs/highlights/*.mp4")):
        st = stamp_of(clip)
        if st[:8] < a.since:
            continue
        t0 = datetime.strptime(st, "%Y%m%dT%H%M%S")
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        step = max(1, int(round(fps * a.every)))
        i, last = 0, None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
                if last is None or float(np.mean(np.abs(small - last))) > 2.0:      # skip near-duplicates
                    name = f"{(t0 + timedelta(seconds=i / fps)).strftime('%Y%m%dT%H%M%S')}_clip{i:04d}.jpg"
                    dst = os.path.join(a.out, name)
                    if not os.path.exists(dst):
                        write_jpeg(dst, fr)
                        n["clip"] += 1
                    last = small
            i += 1
        cap.release()
    print("harvested:", n, "->", a.out)


if __name__ == "__main__":
    main()
