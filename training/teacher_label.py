"""Teacher labels for this camera's frames: MegaDetector boxes, SpeciesNet veto.

    python training/teacher_label.py training_data/site/images            # runs the AI (Python 3.13 venv)
    python training/teacher_label.py training_data/site/images --from-json training_data/site/prelabels.json

Rules (measured on this camera, 16/10/2026):
* MegaDetector alone calls bushes/branches "animal" up to 0.65 - but SpeciesNet
  labels those crops "blank" with 0.95-1.00.  So a box is kept only when
  SpeciesNet does NOT say blank/vehicle/human, and the detector score is
  >= --min-conf.  Everything vetoed simply gets no box: the frame (or tile)
  becomes a hard negative - exactly what teaches the student model that
  vegetation is not wildlife.
* Bird names -> class 1 (bird), everything else -> class 0 (animal).
* Camera OSD text / near-field / edge slivers are dropped (detection.validator.prefilter).
Writes <image>.txt (YOLO) and <image>.teacher.json; then CHECK the positives
(training/review_labels.py, or the contact sheet this tool writes).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from detection.constants import CLASS_ANIMAL, NON_ANIMAL_LABELS
from detection.validator import prefilter
from training.labels_io import label_path, list_images, write_labels, xyxy_to_yolo
from training.prelabel import BIRD_WORDS
from utils.imageio import write_jpeg


def run_ai(images, imgsz=0):
    from detection.pipeline import WildlifePipeline
    import time
    pipe = WildlifePipeline(water_zone=None)
    t0 = time.time()
    while pipe.detector is None:          # MegaDetector loads in the background: without
        if time.time() - t0 > 300:        # this wait the first frames got NO boxes at all
            raise SystemExit("MegaDetector did not load - teacher labels need it")
        time.sleep(1.0)
    if pipe.backend.name != "speciesnet":
        raise SystemExit("SpeciesNet is required for the veto - run with the Python 3.13 venv (.venv313).")
    out = {}
    for i, p in enumerate(images, 1):
        fr = cv2.imread(p)
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        # 0 = native resolution (up to 1920): small far animals need it (19/10:
        # a baboon in shade scored 0.47 at 1280 and 0.62 at 1920)
        size = imgsz or min(1920, max(1280, fr.shape[1]))
        dets = []
        for d in pipe._detect(rgb, 0.10, imgsz=size):
            name, conf = pipe._classify(pipe._context_crop(rgb, d["box"], 0.15))
            dets.append(dict(box=d["box"], cls=d["class_id"], conf=d["det_confidence"], species=name, sconf=conf))
        out[os.path.basename(p)] = dets
        print(f"{i}/{len(images)} {os.path.basename(p)}: {len(dets)} raw", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--from-json", default=None)
    ap.add_argument("--min-conf", type=float, default=0.15)
    ap.add_argument("--sheet", default=None, help="contact sheet of kept boxes (jpg)")
    ap.add_argument("--imgsz", type=int, default=0, help="MegaDetector input size (0 = native, max 1920)")
    ap.add_argument("--unsure-conf", type=float, default=0.40,
                    help="a box this strong that SpeciesNet calls blank makes the frame unusable (left out)")
    a = ap.parse_args()
    images = list_images(a.folder)
    raw = json.load(open(a.from_json)) if a.from_json else run_ai(images, a.imgsz)
    crops, n_pos, n_neg, n_veto, n_skip = [], 0, 0, 0, 0
    for p in images:
        dets = raw.get(os.path.basename(p))
        if dets is None:
            continue
        fr = cv2.imread(p)
        h, w = fr.shape[:2]
        boxes, kept = [], []
        unsure = False
        for d in dets:
            if d["cls"] != CLASS_ANIMAL or d["conf"] < a.min_conf:
                continue
            if not prefilter({"box": d["box"]}, w, h)[0]:
                continue
            name = str(d["species"]).lower()
            if name in NON_ANIMAL_LABELS or name.endswith("blank"):
                n_veto += 1
                # 19/10: two baboons at 3.85x were "blank" to SpeciesNet (too small)
                # but 0.69/0.42 to MegaDetector.  Labelling that frame EMPTY would
                # teach the model that baboons are background: leave it out.
                if d["conf"] >= a.unsure_conf and name.endswith("blank"):
                    unsure = True
                continue
            cls = 1 if any(word in name for word in BIRD_WORDS) else 0
            boxes.append((cls,) + xyxy_to_yolo(d["box"], w, h))
            kept.append(dict(d, label_cls=cls))
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            pad = max(20, (x2 - x1))
            crop = fr[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()
            if crop.size:
                crop = cv2.resize(crop, (160, 160))
                cv2.putText(crop, f"{name[:12]} {d['conf']:.2f}", (3, 14), 0, 0.4, (0, 255, 255), 1)
                crops.append(crop)
        if not boxes and "_missed" in os.path.basename(p):
            # an M-key frame: the operator SAW an animal the AI missed - labelling it empty
            # would teach the model that animal is background.  Leave it out.
            unsure = True
        if unsure:
            n_skip += 1
            if os.path.exists(label_path(p)):
                os.remove(label_path(p))      # no label file = not used for training
            continue
        write_labels(label_path(p), boxes)
        with open(os.path.splitext(p)[0] + ".teacher.json", "w") as f:
            json.dump(kept, f)
        n_pos += bool(boxes)
        n_neg += not boxes
    print(f"frames with animals: {n_pos}  pure negatives: {n_neg}  boxes vetoed as blank/non-wildlife: {n_veto}"
          f"  frames left out (strong box called blank): {n_skip}")
    if a.sheet and crops:
        rows = [np.hstack(crops[i:i + 8] + [np.zeros_like(crops[0])] * (8 - len(crops[i:i + 8])))
                for i in range(0, len(crops), 8)]
        write_jpeg(a.sheet, np.vstack(rows))
        print("contact sheet:", a.sheet)


if __name__ == "__main__":
    main()
