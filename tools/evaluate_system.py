"""Offline benchmark for detection logs against a simple frame-level GT file.

Ground truth JSON format:
[
  {"frame_id": 1, "boxes": [[x1,y1,x2,y2], ...]},
  ...
]

Prediction JSONL format is intentionally simple and can be exported from a
recorded run. Each line: {"frame_id": 1, "box": [...], "track_id": 3}

This tool reports precision/recall/F1 at configurable IoU. It does not claim
that log-level counts are accuracy; you need reviewed ground truth to measure
that honestly.
"""

import argparse
import csv
import json
from collections import defaultdict


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def load_gt(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(row["frame_id"]): list(row.get("boxes", [])) for row in data}


def load_predictions(path):
    result = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            item = json.loads(row)
            result[int(item["frame_id"])].append(item["box"])
    return result


def score(gt, pred, threshold):
    tp = fp = fn = 0
    for frame_id in sorted(set(gt) | set(pred)):
        g = list(gt.get(frame_id, []))
        p = list(pred.get(frame_id, []))
        matched = set()
        for box in p:
            best = max(((iou(box, x), i) for i, x in enumerate(g) if i not in matched), default=(0.0, -1))
            if best[0] >= threshold:
                tp += 1
                matched.add(best[1])
            else:
                fp += 1
        fn += len(g) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def export_csv_jsonl(csv_path, out_path):
    # detections.csv lacks frame IDs because it is intended for operations;
    # this helper therefore creates sequential frame IDs for every unique row.
    # For rigorous benchmarking, generate explicit frame IDs from a recorded run.
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            try:
                box = json.loads(row["box"].replace("'", '"'))
            except Exception:
                continue
            rows.append({"frame_id": i, "box": box, "track_id": row.get("track_id")})
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    result = score(load_gt(args.ground_truth), load_predictions(args.predictions), args.iou)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
