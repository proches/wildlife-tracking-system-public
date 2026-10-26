"""Auto-draw first-draft boxes (MegaDetector + tiles) so labelling is 10x faster.

    python training/prelabel.py dataset_raw/images            # writes <image>.txt next to each image
    python training/prelabel.py logs/review --classify        # also guess bird vs animal with SpeciesNet/AI4G

The drafts are deliberately generous (low threshold, tiles for small birds);
you then CONFIRM / FIX them with  training/review_labels.py.  Never train on
unreviewed drafts: the model would just learn MegaDetector's mistakes.
Existing .txt files are kept unless --overwrite.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from detection.constants import CLASS_ANIMAL
from detection.validator import prefilter
from training.labels_io import list_images, label_path, write_labels, xyxy_to_yolo

BIRD_WORDS = ("bird", "guineafowl", "ostrich", "bustard", "stork", "heron", "egret", "ibis", "eagle", "vulture",
              "hawk", "kite", "goose", "duck", "plover", "lapwing", "francolin", "hornbill", "weaver", "starling",
              "dove", "pigeon", "crane", "secretary", "owl", "flamingo", "pelican", "kingfisher", "roller", "aves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-tiles", action="store_true")
    a = ap.parse_args()

    from detection.pipeline import WildlifePipeline
    pipe = WildlifePipeline(water_zone=None)
    done = 0
    for img_path in list_images(a.folder):
        lp = label_path(img_path)
        if os.path.exists(lp) and not a.overwrite:
            continue
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = pipe._detect(rgb, a.conf)
        if not a.no_tiles:
            dets = pipe._fuse(dets, pipe._enhanced_detect(rgb))
        boxes = []
        for d in dets:
            if d["class_id"] != CLASS_ANIMAL:
                continue
            ok, _ = prefilter(d, w, h)
            if not ok:
                continue
            cls = 0
            if a.classify:
                crop = pipe._context_crop(rgb, d["box"], 0.15)
                name, conf = pipe._classify(crop)
                if any(word in str(name).lower() for word in BIRD_WORDS):
                    cls = 1
            boxes.append((cls,) + xyxy_to_yolo(d["box"], w, h))
        write_labels(lp, boxes)
        done += 1
        print(f"{os.path.basename(img_path)}: {len(boxes)} draft boxes")
    print(f"Drafted {done} images. Now review them: python training/review_labels.py {a.folder}")


if __name__ == "__main__":
    main()
