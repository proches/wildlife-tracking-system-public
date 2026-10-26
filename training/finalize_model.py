"""Validate a freshly trained detector and switch it on only if it is good enough.

    python training/finalize_model.py --weights runs/wildlife/melia_v1/weights/best.pt

Gate (on training_data/dataset val = held-out Serengeti locations + held-out
time blocks of this camera): mAP50 >= --min-map50 and bird recall >= --min-bird-recall.
Pass -> copied to models/melia_wildlife.pt and enabled in .env
(WILDLIFE_CUSTOM_YOLO_ENABLED=true).  Fail -> nothing changes; the report says why.
Also reports false positives on this camera's empty tiles (vegetation).
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)


def set_env(key, value, path=".env"):
    lines = open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []
    out, done = [], False
    for line in lines:
        if line.split("=", 1)[0].strip() == key:
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/runs/wildlife/melia_v1/weights/best.pt")
    ap.add_argument("--data", default="training_data/dataset/data.yaml")
    ap.add_argument("--min-map50", type=float, default=0.50)
    ap.add_argument("--min-bird-recall", type=float, default=0.40)
    ap.add_argument("--baseline", default=None,
                    help="current model: the new one must be at least as good on the SAME validation set")
    ap.add_argument("--no-install", action="store_true",
                    help="evaluate and print only: no report file, no model copy, no .env change")
    a = ap.parse_args()
    data_dir = os.path.dirname(os.path.abspath(a.data))
    empty = [p for p in glob.glob(os.path.join(data_dir, "images", "val", "site_*.jpg"))
             if os.path.getsize(p.replace("images", "labels").rsplit(".", 1)[0] + ".txt") == 0]
    site_pos = [p for p in glob.glob(os.path.join(data_dir, "images", "val", "site_*.jpg"))
                if os.path.getsize(p.replace("images", "labels").rsplit(".", 1)[0] + ".txt") > 0]

    def subset_data(prefix):
        """data.yaml whose validation is only the site_* or only the lila_* images.
        25/10: round 3 lost on birds, and one mixed mAP could not say whether on this
        camera or on the Serengeti camera traps."""
        imgs = sorted(glob.glob(os.path.join(data_dir, "images", "val", prefix + "_*.jpg")))
        if not imgs:
            return None
        with open(os.path.join(data_dir, f"val_{prefix}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(imgs) + "\n")
        lines = [f"val: val_{prefix}.txt" if ln.startswith("val:") else ln
                 for ln in open(a.data, encoding="utf-8").read().splitlines()]
        path = os.path.join(data_dir, f"data_val_{prefix}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    sources = {s: subset_data(s) for s in ("site", "lila")}

    def per_class(m):
        # class_result(k) counts the classes PRESENT in the set (ap_class_index), not class
        # ids: a validation part without birds made class_result(1) raise IndexError (25/10)
        out = {}
        for k, c in enumerate(m.box.ap_class_index):
            p, r, ap50, _ = m.box.class_result(k)
            out[m.names[int(c)]] = dict(precision=round(float(p), 3), recall=round(float(r), 3),
                                        map50=round(float(ap50), 3))
        return out

    def evaluate(weights):
        from ultralytics import YOLO
        model = YOLO(weights)
        m = model.val(data=a.data, imgsz=640, batch=8, device="cpu", conf=0.15, plots=False, verbose=False)
        per = per_class(m)
        by_source = {}
        for src, data in sources.items():
            if data:
                ms = model.val(data=data, imgsz=640, batch=8, device="cpu", conf=0.15, plots=False, verbose=False)
                by_source[src] = dict(map50=round(float(ms.box.map50), 3), per_class=per_class(ms))
        # false positives on this camera's EMPTY validation tiles (vegetation, water, shadows)
        fp = sum(int(len(model.predict(p, imgsz=640, conf=0.25, device="cpu", verbose=False)[0].boxes) > 0)
                 for p in empty)
        # this camera's labelled animals (baboons, buffalo ...): tiles where it finds something
        hit = sum(int(len(model.predict(p, imgsz=640, conf=0.25, device="cpu", verbose=False)[0].boxes) > 0)
                  for p in site_pos)
        return dict(weights=weights, map50=round(float(m.box.map50), 3), per_class=per, by_source=by_source,
                    site_empty_tiles=len(empty), site_empty_tiles_with_false_box=fp,
                    site_animal_tiles=len(site_pos), site_animal_tiles_found=hit)

    report = evaluate(a.weights)
    report["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    map50 = report["map50"]
    bird_recall = report["per_class"].get("bird", {}).get("recall", 0.0)
    ok = map50 >= a.min_map50 and bird_recall >= a.min_bird_recall
    if a.baseline and os.path.exists(a.baseline):
        base = evaluate(a.baseline)
        report["baseline"] = base
        better_site = report["site_animal_tiles_found"] >= base["site_animal_tiles_found"]
        no_worse = map50 >= base["map50"] - 0.01
        clean = (report["site_empty_tiles_with_false_box"]
                 <= base["site_empty_tiles_with_false_box"] + max(1, len(empty) // 50))
        # this camera is what the model is for: its own mAP must not drop either
        new_site = report["by_source"].get("site", {}).get("map50")
        old_site = base["by_source"].get("site", {}).get("map50")
        site_ok = new_site is None or old_site is None or new_site >= old_site - 0.01
        report["vs_baseline"] = dict(site_recall_not_worse=better_site, map50_not_worse=no_worse,
                                     vegetation_not_worse=clean, site_map50_not_worse=site_ok)
        ok = ok and better_site and no_worse and clean and site_ok
    report["accepted"] = ok
    print(json.dumps(report, indent=2))
    if a.no_install:
        print("--no-install: nothing written")
        return 0 if ok else 1
    os.makedirs("models", exist_ok=True)
    json.dump(report, open("models/melia_wildlife_report.json", "w"), indent=2)
    if ok:
        if os.path.exists("models/melia_wildlife.pt"):
            backup = time.strftime("models/melia_wildlife_before_%Y%m%d_%H%M.pt")
            shutil.copy2("models/melia_wildlife.pt", backup)
            print("previous model kept as", backup)
        # copy then rename: a job stopped mid-copy never leaves a half-written model behind
        shutil.copy2(a.weights, "models/melia_wildlife.pt.tmp")
        os.replace("models/melia_wildlife.pt.tmp", "models/melia_wildlife.pt")
        set_env("WILDLIFE_CUSTOM_YOLO_ENABLED", "true")
        set_env("WILDLIFE_CUSTOM_YOLO_MODEL_PATH", "models/melia_wildlife.pt")
        print("ACCEPTED -> models/melia_wildlife.pt, enabled in .env (restart the tracker)")
    else:
        # name the checks that really failed (21/10: the old text "map50 0.64 < 0.5 or bird
        # recall 0.66 < 0.4" was printed although the model failed the BASELINE comparison)
        why = []
        if map50 < a.min_map50:
            why.append(f"map50 {map50:.3f} < minimum {a.min_map50}")
        if bird_recall < a.min_bird_recall:
            why.append(f"bird recall {bird_recall:.3f} < minimum {a.min_bird_recall}")
        vs, base = report.get("vs_baseline", {}), report.get("baseline", {})
        if vs and not vs.get("map50_not_worse", True):
            why.append(f"map50 {map50:.3f} is worse than the current model's {base.get('map50', 0.0):.3f}")
        if vs and not vs.get("site_recall_not_worse", True):
            why.append(f"finds {report['site_animal_tiles_found']} of {report['site_animal_tiles']} site animal "
                       f"tiles, the current model {base.get('site_animal_tiles_found')}")
        if vs and not vs.get("site_map50_not_worse", True):
            why.append(f"mAP50 on this camera {report['by_source']['site']['map50']:.3f} is worse than the "
                       f"current model's {base['by_source']['site']['map50']:.3f}")
        if vs and not vs.get("vegetation_not_worse", True):
            why.append(f"false boxes on {report['site_empty_tiles_with_false_box']} empty site tiles, "
                       f"the current model {base.get('site_empty_tiles_with_false_box')}")
        print(f"NOT accepted ({'; '.join(why) or 'see the report above'}); the current model stays.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
