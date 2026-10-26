"""Turn reviewed images into an Ultralytics dataset (train/val split by TIME, optional tiling).

    python training/build_dataset.py dataset_raw/images --out dataset --tile
    python training/build_dataset.py logs/review dataset_raw/images --out dataset --tile --val 0.15

Only images with a '.reviewed' marker are used (see review_labels.py); pass
--include-unreviewed to override (not recommended).

Why split by time: frames a few seconds apart are almost identical.  A random
split puts near-copies in both train and val, so the model looks great on val
and fails in the field.  Here whole 10-minute blocks go to one side.

Why --tile: a bird that is 14 px wide in a 1920 px frame is *erased* when the
frame is shrunk to 640/960 for training.  Tiles keep native pixels, so small
birds stay visible and the detector actually learns them.  Run the trained model
on the full frame at imgsz 1920 (CUSTOM_YOLO_IMGSZ) - it is fully convolutional.
"""

import argparse
import os
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.labels_io import CLASSES, label_path, list_images, read_labels, write_labels, xyxy_to_yolo, yolo_to_xyxy
from utils.imageio import write_jpeg

_TS = re.compile(r"(\d{8}T\d{6})")


def image_time(path):
    m = _TS.search(os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").timestamp()
        except ValueError:
            pass
    return os.path.getmtime(path)


def split_by_time(items, val_fraction, gap_seconds=600, seed=7):
    """items: list of (path, time).  Returns (train, val) lists of paths, split by time blocks."""
    items = sorted(items, key=lambda x: x[1])
    blocks, cur, last = [], [], None
    for p, t in items:
        if last is not None and t - last > gap_seconds and cur:
            blocks.append(cur)
            cur = []
        cur.append(p)
        last = t
    if cur:
        blocks.append(cur)
    if len(blocks) < 2:
        # everything is one block: split the sequence itself, newest part = val
        cut = max(1, int(len(items) * (1 - val_fraction)))
        paths = [p for p, _ in items]
        return paths[:cut], paths[cut:]
    rng = random.Random(seed)
    order = list(range(len(blocks)))
    rng.shuffle(order)
    total = sum(len(b) for b in blocks)
    val, train, val_n = [], [], 0
    for i in order:
        if val_n < total * val_fraction:
            val += blocks[i]
            val_n += len(blocks[i])
        else:
            train += blocks[i]
    return train, val


def tile_positions(size, tile, overlap):
    if size <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    pos = list(range(0, size - tile + 1, step))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def tile_labels(boxes_xyxy, x0, y0, tile_w, tile_h, min_visible=0.5):
    """Clip boxes (cls, x1,y1,x2,y2 in px) to a tile; keep those >= min_visible visible.
    Returns list of (cls, cx, cy, w, h) normalised to the tile."""
    out = []
    for cls, x1, y1, x2, y2 in boxes_xyxy:
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            continue
        cx1, cy1 = max(x1, x0), max(y1, y0)
        cx2, cy2 = min(x2, x0 + tile_w), min(y2, y0 + tile_h)
        vis = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
        if vis / area < min_visible or cx2 - cx1 < 4 or cy2 - cy1 < 4:
            continue
        out.append((cls,) + xyxy_to_yolo((cx1 - x0, cy1 - y0, cx2 - x0, cy2 - y0), tile_w, tile_h))
    return out


def emit_plain(img_path, labels, out_img_dir, out_lbl_dir, stem):
    shutil.copy2(img_path, os.path.join(out_img_dir, stem + ".jpg"))
    write_labels(os.path.join(out_lbl_dir, stem + ".txt"), labels)


def emit_tiles(img_path, labels, out_img_dir, out_lbl_dir, stem, tile, overlap, empty_keep, rng):
    frame = cv2.imread(img_path)
    if frame is None:
        return 0
    h, w = frame.shape[:2]
    boxes = [(c,) + yolo_to_xyxy((cx, cy, bw, bh), w, h) for c, cx, cy, bw, bh in labels]
    n = 0
    for y0 in tile_positions(h, tile, overlap):
        for x0 in tile_positions(w, tile, overlap):
            tw, th = min(tile, w), min(tile, h)
            tl = tile_labels(boxes, x0, y0, tw, th)
            if not tl and rng.random() > empty_keep:
                continue
            name = f"{stem}_x{x0}_y{y0}"
            write_jpeg(os.path.join(out_img_dir, name + ".jpg"), frame[y0:y0 + th, x0:x0 + tw], 94)
            write_labels(os.path.join(out_lbl_dir, name + ".txt"), tl)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--tile", action="store_true")
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--empty-tile-keep", type=float, default=0.25, help="fraction of empty tiles kept as negatives")
    ap.add_argument("--include-unreviewed", action="store_true")
    a = ap.parse_args()

    items = []
    for folder in a.folders:
        for p in list_images(folder):
            reviewed = os.path.exists(os.path.splitext(p)[0] + ".reviewed")
            if reviewed or a.include_unreviewed:
                items.append((p, image_time(p)))
    if not items:
        sys.exit("No reviewed images. Run training/review_labels.py first (or pass --include-unreviewed).")
    train, val = split_by_time(items, a.val)
    rng = random.Random(3)
    stats = {}
    for split, paths in (("train", train), ("val", val)):
        di, dl = (os.path.join(a.out, "images", split), os.path.join(a.out, "labels", split))
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        n_img = n_box = n_neg = 0
        for k, p in enumerate(paths):
            labels = read_labels(label_path(p))
            stem = f"{k:05d}_{Path(p).stem[:40]}"
            if a.tile and split == "train":
                n_img += emit_tiles(p, labels, di, dl, stem, a.tile_size, a.overlap, a.empty_tile_keep, rng)
            else:
                emit_plain(p, labels, di, dl, stem)   # validate on FULL frames: that is what runs in the field
                n_img += 1
            n_box += len(labels)
            n_neg += not labels
        stats[split] = (len(paths), n_img, n_box, n_neg)
    yaml = os.path.join(a.out, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
    for split, (src, out, boxes, neg) in stats.items():
        print(f"{split}: {src} source images -> {out} training images, {boxes} boxes, {neg} hard negatives")
    print("Dataset ready:", yaml)
    if stats["val"][0] == 0 or stats["val"][2] == 0:
        print("WARNING: validation set has no objects - label more images before trusting any metric.")


if __name__ == "__main__":
    main()
