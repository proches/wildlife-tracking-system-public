"""Hand-verified examples from THIS camera -> native-resolution 640-px crops for training.

    .venv313\\Scripts\\python.exe training\\build_site_verified.py

training/verified_21oct.json lists boxes the running system produced (review frames)
or the operator saved (M key), each checked by eye:
  "animal" / "bird" -> a labelled positive (the small far buffalo, waterbuck, primates
                       in shade and birds the site model missed or doubted);
  "no"              -> an EMPTY crop: the acacia, tree trunk or dark canopy the
                       detectors called an animal - the hard negatives of this view.
A crop is 640 x 640 around the box at native resolution (the size the site model
sees per tile), so a 20-px animal stays 20 px - the far animals the model must learn.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from training.labels_io import write_labels, xyxy_to_yolo  # noqa: E402
from utils.imageio import write_jpeg  # noqa: E402

CLASS = {"animal": 0, "bird": 1}


def item_frame_and_box(item):
    if "review" in item:
        if not os.path.exists(item["review"]):
            return None, None, os.path.basename(item["review"])[:23]     # logs/ not restored
        meta = json.load(open(item["review"], encoding="utf-8"))
        img = cv2.imread(os.path.splitext(item["review"])[0] + ".jpg")
        det = max(meta.get("detections") or [{}], key=lambda d: d.get("det_confidence", 0.0))
        return img, det.get("box"), os.path.basename(item["review"])[:23]
    img = cv2.imread(item["image"])
    if img is not None:
        img = cv2.resize(img, (1920, 1080))          # the boxes were measured on 1920 x 1080
    return img, item.get("box"), os.path.splitext(os.path.basename(item["image"]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="training/verified_21oct.json")
    ap.add_argument("--out", default="training_data/site3v/images")
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.list, encoding="utf-8"))["items"]
    done = {"animal": 0, "bird": 0, "no": 0, "skipped": 0, "kept": 0}
    for item in items:
        img, box, stem = item_frame_and_box(item)
        if img is None or not box:
            out = os.path.join(a.out, f"{stem}_verified.jpg")
            if os.path.exists(out) and os.path.exists(os.path.splitext(out)[0] + ".txt"):
                # the source frame is gone (a restored PC has no logs/review), the crop made
                # from it is in the repository: keep it
                done[item["verdict"]] += 1
                done["kept"] += 1
            else:
                done["skipped"] += 1
            continue
        h, w = img.shape[:2]
        s = min(a.size, w, h)
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        x1 = int(min(max(0.0, cx - s / 2.0), w - s))
        y1 = int(min(max(0.0, cy - s / 2.0), h - s))
        crop = img[y1:y1 + s, x1:x1 + s]
        out = os.path.join(a.out, f"{stem}_verified.jpg")
        write_jpeg(out, crop, 95)
        labels = []
        if item["verdict"] in CLASS:
            b = [max(0.0, box[0] - x1), max(0.0, box[1] - y1), min(s, box[2] - x1), min(s, box[3] - y1)]
            labels.append((CLASS[item["verdict"]],) + xyxy_to_yolo(b, s, s))
        write_labels(os.path.splitext(out)[0] + ".txt", labels)
        done[item["verdict"]] += 1
    print(f"verified crops in {a.out}: {done}")
    return 0 if done["animal"] + done["bird"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
