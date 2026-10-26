"""Tiny YOLO-label helpers shared by the training scripts (no heavy imports)."""

import os

CLASSES = ["animal", "bird"]          # extend later, e.g. split 'animal' into species
IMG_EXTS = (".jpg", ".jpeg", ".png")


def label_path(img_path):
    return os.path.splitext(img_path)[0] + ".txt"


def read_labels(path):
    """-> list of (cls, cx, cy, w, h) normalised floats."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                try:
                    out.append((int(float(p[0])),) + tuple(float(v) for v in p[1:5]))
                except ValueError:
                    pass
    return out


def write_labels(path, boxes):
    with open(path, "w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in boxes:
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def xyxy_to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h, (x2 - x1) / w, (y2 - y1) / h)


def yolo_to_xyxy(b, w, h):
    cx, cy, bw, bh = b
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def list_images(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)
