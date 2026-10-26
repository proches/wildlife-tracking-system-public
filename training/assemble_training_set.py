"""Merge Serengeti camera-trap labels + this camera's reviewed frames into one YOLO set.

    python training/assemble_training_set.py --lila training_data/lila_yolo \
        --site training_data/site/images --out training_data/dataset

* LILA Snapshot Serengeti (training/build_lila_serengeti.py): whole images;
  validation = whole camera-trap LOCATIONS held out (no near-duplicate leakage).
* Site frames (training/harvest_patrol.py + review): cut into native-resolution
  tiles; empty tiles - grass, branches, bushes, water glare and shadows from
  this exact view - are the hard negatives that stop the detector calling
  vegetation an animal.  --empty-keep keeps a fraction of them for TRAINING;
  validation keeps all of them (finalize_model.py counts false boxes on them).
  Validation = held-out time blocks of each folder.
  --site-repeat duplicates the site training tiles (the site is the domain that
  matters most but has far fewer images).
* --verified: hand-checked crops (training/build_site_verified.py).  They all go
  to training, and a teacher-labelled frame taken in the same second is left
  out: on 23/10, 14 of the 67 checked pictures had a teacher label saying the
  opposite (primates labelled "nothing", an acacia labelled "animal").
"""

import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.build_dataset import emit_tiles, image_time, split_by_time
from training.labels_io import CLASSES, label_path, list_images, read_labels


_STAMP = re.compile(r"(\d{8}T\d{6})")


def _stamp(path):
    m = _STAMP.search(os.path.basename(path))
    return m.group(1) if m else None


def lila_location(stem):
    parts = stem.split("_")             # S1_D05_R5_PICT1288
    return parts[1] if len(parts) > 2 else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lila", nargs="+", default=["training_data/lila_yolo"],
                    help="one or more LILA folders (images/ + labels/); --lila-max applies to each")
    ap.add_argument("--site", nargs="+", default=["training_data/site/images"],
                    help="one or more folders of labelled frames from this camera")
    ap.add_argument("--out", default="training_data/dataset")
    ap.add_argument("--val", type=float, default=0.12)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--site-repeat", type=int, default=2)
    ap.add_argument("--empty-keep", type=float, default=1.0,
                    help="fraction of EMPTY site TRAIN tiles kept (23/10: all of them made a 26 000-tile set)")
    ap.add_argument("--verified", nargs="*", default=[],
                    help="folders of hand-checked crops: all to training, every empty tile kept")
    ap.add_argument("--verified-repeat", type=int, default=3,
                    help="the hand-checked crops are few (67) and are the cases the model gets wrong")
    ap.add_argument("--lila-max", type=int, default=0,
                    help="at most this many LILA TRAIN images (0 = all); validation keeps every held-out location")
    a = ap.parse_args()
    rng = random.Random(7)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    dirs = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(a.out, kind, split)
            os.makedirs(d, exist_ok=True)
            dirs[(kind, split)] = d
    counts = {"lila_train": 0, "lila_val": 0, "site_train_tiles": 0, "site_val_tiles": 0}

    # --- LILA, split by location
    per_folder, seen = [], set()
    for folder in a.lila:
        imgs = []
        for p in (list_images(os.path.join(folder, "images")) if os.path.isdir(folder) else []):
            if Path(p).stem not in seen:              # the same LILA picture in two folders: once
                seen.add(Path(p).stem)
                imgs.append((p, folder))
        per_folder.append(imgs)
    locs = sorted({lila_location(Path(p).stem) for imgs in per_folder for p, _ in imgs})
    rng.shuffle(locs)
    val_locs = set(locs[:max(1, int(len(locs) * a.val))]) if locs else set()
    lila_imgs = []
    for imgs in per_folder:
        if a.lila_max:
            tr = [x for x in imgs if lila_location(Path(x[0]).stem) not in val_locs]
            va = [x for x in imgs if lila_location(Path(x[0]).stem) in val_locs]
            rng.shuffle(tr)
            imgs = tr[:a.lila_max] + va
        lila_imgs += imgs
    for p, folder in lila_imgs:
        stem = Path(p).stem
        lbl = os.path.join(folder, "labels", stem + ".txt")
        if not os.path.exists(lbl):
            continue
        split = "val" if lila_location(stem) in val_locs else "train"
        shutil.copy2(p, os.path.join(dirs[("images", split)], "lila_" + stem + ".jpg"))
        shutil.copy2(lbl, os.path.join(dirs[("labels", split)], "lila_" + stem + ".txt"))
        counts[f"lila_{split}"] += 1

    # --- site frames, tiled, split by time (per folder: on 23/10 one joint split put 59 %
    # of the older frames in validation)
    verified = [p for folder in a.verified if os.path.isdir(folder)
                for p in list_images(folder) if os.path.exists(label_path(p))]
    hand_checked = {_stamp(p) for p in verified} - {None}
    jobs, names = [], set()
    for folder in a.site:
        site = []
        for p in (list_images(folder) if os.path.isdir(folder) else []):
            name = os.path.basename(p)
            if not os.path.exists(label_path(p)) or name in names:
                continue                                  # unlabelled, or the same frame twice
            if _stamp(p) in hand_checked:
                counts["site_frames_replaced_by_verified"] = counts.get("site_frames_replaced_by_verified", 0) + 1
                continue
            names.add(name)
            site.append((p, image_time(p) or 0.0))
        train, val = split_by_time(site, a.val) if site else ([], [])
        jobs += [("train", p, a.site_repeat, a.empty_keep) for p in train]
        jobs += [("val", p, 1, 1.0) for p in val]
    jobs += [("train", p, a.verified_repeat, 1.0) for p in verified]
    for split, p, reps, empty_keep in jobs:
        labels = read_labels(label_path(p))
        for r in range(max(1, reps)):
            stem = f"site_{Path(p).stem}_r{r}"
            counts[f"site_{split}_tiles"] += emit_tiles(
                p, labels, dirs[("images", split)], dirs[("labels", split)], stem,
                a.tile, 0.2, empty_keep, rng)

    yaml = os.path.join(a.out, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(CLASSES):
            f.write(f"  {i}: {n}\n")
    print(counts)
    print("Wrote", yaml)


if __name__ == "__main__":
    main()
