"""Explain what is in logs/detections.csv - and what is probably a false positive.

    python tools/log_report.py [logs/detections.csv]

Prints: detections per species, how many are low-confidence, and 'sticky'
locations (same box seen again and again from the same pose = clutter, not an
animal).  Works on both the old (6/9-column) and the v3 format.
"""

import csv
import sys
from collections import Counter, defaultdict


def main(path="logs/detections.csv"):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            rows.append(r)
    rows = rows[1:]
    species, low, boxes = Counter(), 0, defaultdict(int)
    for r in rows:
        try:
            if len(r) >= 14:
                conf, sp, box = float(r[3]), r[4], r[8]
            elif len(r) >= 9:
                conf, sp, box = float(r[3]), r[4], r[8]
            else:
                conf, sp, box = float(r[2]), r[3], r[5]
        except (ValueError, IndexError):
            continue
        species[sp] += 1
        low += conf < 0.30
        try:
            nums = [float(v) for v in box.strip("[]\"").split(",")]
            key = tuple(round(v / 40) for v in nums)
            boxes[key] += 1
        except ValueError:
            pass
    total = sum(species.values())
    print(f"{total} detections; {low} ({100 * low / max(1, total):.0f}%) have detector confidence < 0.30")
    for sp, n in species.most_common(15):
        print(f"  {sp or '(blank)':<18}{n}")
    sticky = [(k, n) for k, n in boxes.items() if n >= 5]
    sticky.sort(key=lambda kv: -kv[1])
    print(f"\n{len(sticky)} locations were 'detected' 5+ times at the same spot (likely clutter):")
    for k, n in sticky[:10]:
        print(f"  box~{[v * 40 for v in k]}  x{n}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
