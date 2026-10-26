"""Measure the whole detection pipeline on a fixed set of real frames.

The set (tools/eval_set_20oct.json, 65 frames from this camera):
  missed   - 13 frames where the operator SAW animals the AI missed
             (vervet monkeys / a baboon in shade at 16-41x)
  animal   - 32 frames with confirmed animals
  negative - 20 empty scenes

Every frame goes through detection.pipeline.process_frame (detectors, SpeciesNet,
blank veto, prefilter), MegaDetector on every frame, and the table shows how many
frames per kind end with at least one kept animal.  20/10/2026 baseline:

  kind      MD_ONLY 0.35 (config)   MD_ONLY 0.20
  missed     0 / 13                  0 / 13   (weak boxes vetoed as "blank")
  animal    29 / 32                 30 / 32
  negative   0 / 20                  0 / 20

Use it to compare a retrained site model with the old one.  Never run it while the
tracker runs (both need the CPU); it writes nothing under logs/.

    .venv313\\Scripts\\python.exe tools\\eval_detection.py
    .venv313\\Scripts\\python.exe tools\\eval_detection.py --md-only 0.20 --out eval.json
"""

import argparse
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402

import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default=os.path.join("tools", "eval_set_20oct.json"))
    ap.add_argument("--md-only", type=float, default=None, help="what-if value for MD_ONLY_MIN_CONFIDENCE")
    ap.add_argument("--out", default=None, help="write the per-frame results (JSON) here")
    args = ap.parse_args()

    frames = json.load(open(args.set, encoding="utf-8"))["frames"]
    config.MEGADETECTOR_EVERY_N_WITH_CUSTOM = config.MEGADETECTOR_EVERY_N_ZOOMED = 1
    if args.md_only is not None:
        config.MD_ONLY_MIN_CONFIDENCE = args.md_only
    from detection.pipeline import WildlifePipeline
    from detection.validator import StaticFalsePositiveMemory

    tmp = tempfile.mkdtemp()            # the clutter memory must not learn from this run
    p = WildlifePipeline(static_memory=StaticFalsePositiveMemory(os.path.join(tmp, "static.json")))
    t_wait = time.time()
    while p.detector is None and not getattr(p, "_md_failed", False) and time.time() - t_wait < 300:
        time.sleep(1)
    if p.detector is None:
        print("MegaDetector did not load - the numbers would not be comparable", file=sys.stderr)
        return 2

    out, summary = [], {}
    for item in frames:
        path, kind = item["path"], item["kind"]
        img = cv2.imread(path)
        if img is None:
            print(f"missing: {path}", file=sys.stderr)
            continue
        img = cv2.resize(img, (1920, 1080))
        try:
            pose = tuple(json.load(open(os.path.splitext(path)[0] + ".json", encoding="utf-8")).get("pose")
                         or (0.5, -1.0, 0.03))
        except (OSError, ValueError):
            pose = (0.5, -1.0, 0.03)
        p.invalidate_classification_cache()
        t = time.time()
        res = p.process_frame(img, scan_mode=True, pose=pose, deep=True, settled=False)
        m = p.last_metrics
        rec = {"path": path, "kind": kind, "seconds": round(time.time() - t, 1),
               "boxes_after_md_gate": m["base_detections"], "kept": len(res),
               "vetoed_blank": m["dropped_non_wildlife"], "rejected": m["rejected"],
               "animals": [(round(d["det_confidence"], 2), d["species"], round(float(d["species_confidence"] or 0), 2),
                            d.get("source")) for d in res]}
        out.append(rec)
        s = summary.setdefault(kind, [0, 0, 0.0])
        s[0] += 1
        s[1] += 1 if res else 0
        s[2] += rec["seconds"]
        print(f"{kind:8s} kept={len(res)} {os.path.basename(path)[:40]} {rec['animals']}", flush=True)

    print(f"\nMD_ONLY_MIN_CONFIDENCE = {config.MD_ONLY_MIN_CONFIDENCE}")
    print("kind      frames with an animal   mean s/frame")
    for kind in ("missed", "animal", "negative"):
        if kind in summary:
            n, hit, secs = summary[kind]
            print(f"{kind:8s}  {hit:3d} / {n:<3d}              {secs / n:5.1f}")
    if args.out:
        json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
