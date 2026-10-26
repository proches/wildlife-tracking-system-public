"""Recall / precision by object SIZE - shows exactly where small birds are lost.

    # predictions from your trained model (save txt + conf):
    yolo predict model=models/melia_wildlife.pt source=dataset/images/val imgsz=1920 conf=0.15 save_txt=True save_conf=True
    python training/eval_by_size.py --gt dataset/labels/val --pred runs/detect/predict/labels --width 1920 --height 1080

GT and prediction folders hold YOLO .txt files with the same stems; predictions
may carry a 6th column (confidence).  Matching: IoU >= 0.5 (class ignored unless --classes).
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BUCKETS = [("tiny  (<16px)", 0, 16), ("small (16-32px)", 16, 32), ("medium(32-96px)", 32, 96), ("large (>=96px)", 96, 1e9)]


def read(path, w, h, with_conf=False):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines:
        p = line.split()
        if len(p) < 5:
            continue
        cls, cx, cy, bw, bh = int(float(p[0])), *[float(v) for v in p[1:5]]
        conf = float(p[5]) if len(p) > 5 else 1.0
        out.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, conf))
    return out


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def evaluate(gt_dir, pred_dir, w, h, conf=0.15, iou_thr=0.5, use_classes=False):
    stats = {b[0]: [0, 0] for b in BUCKETS}          # [gt, found]
    fp = tp = 0
    for name in sorted(os.listdir(gt_dir)):
        if not name.endswith(".txt"):
            continue
        gts = read(os.path.join(gt_dir, name), w, h)
        preds = [p for p in read(os.path.join(pred_dir, name), w, h) if p[5] >= conf]
        used = set()
        for g in gts:
            size = max(g[3] - g[1], g[4] - g[2])
            label = next(b[0] for b in BUCKETS if b[1] <= size < b[2])
            stats[label][0] += 1
            best, bi = 0.0, None
            for i, p in enumerate(preds):
                if i in used or (use_classes and p[0] != g[0]):
                    continue
                v = iou(g[1:5], p[1:5])
                if v > best:
                    best, bi = v, i
            if best >= iou_thr:
                used.add(bi)
                stats[label][1] += 1
        tp += len(used)
        fp += len(preds) - len(used)
    return stats, tp, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--classes", action="store_true")
    a = ap.parse_args()
    stats, tp, fp = evaluate(a.gt, a.pred, a.width, a.height, a.conf, use_classes=a.classes)
    print(f"{'size bucket':<18}{'objects':>8}{'found':>8}{'recall':>9}")
    for label, (n, f) in stats.items():
        print(f"{label:<18}{n:>8}{f:>8}{(100 * f / n if n else 0):>8.0f}%")
    print(f"\nprecision at conf>={a.conf}: {100 * tp / max(1, tp + fp):.0f}%  ({fp} false positives)")


if __name__ == "__main__":
    main()
