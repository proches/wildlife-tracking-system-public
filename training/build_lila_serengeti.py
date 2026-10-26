"""Build a labelled Serengeti detection set from LILA "Snapshot Serengeti".

    python training/build_lila_serengeti.py --meta training_data/lila --out training_data/lila_yolo \
        --birds 1200 --animals 2400

Snapshot Serengeti = camera traps inside Serengeti National Park, with ~146 000
human-drawn boxes (SnapshotSerengetiBboxes_20190903.json) and species labels
(SnapshotSerengeti_S1-11_v2_1.json).  Both files: https://lila.science (CDLA-permissive).

* keeps images whose sequence has exactly ONE species (no ambiguous labels) and
  no "group" boxes;
* class 0 = animal, 1 = bird (guineafowl, ostrich, secretary bird, kori bustard,
  other birds ...) - the classes of training/labels_io.py;
* balances species (cap per species), prefers daylight, keeps some night/IR;
* downloads in parallel and resizes to --max-side px (labels are normalised).
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.imageio import write_jpeg  # noqa: E402

BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/"
BIRD_WORDS = ("bird", "guineafowl", "ostrich", "bustard", "secretary", "stork", "heron", "eagle", "vulture")
SKIP = {"empty", "blank", "human", "vehicle", "fire", "", "none"}


def load(meta_dir):
    bbox_file = glob.glob(os.path.join(meta_dir, "SnapshotSerengetiBboxes*.json"))[0]
    species_file = glob.glob(os.path.join(meta_dir, "SnapshotSerengeti_S1-11*.json"))[0]
    bb = json.load(open(bbox_file, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in bb["categories"]}
    images = {im["id"]: im for im in bb["images"]}
    boxes = defaultdict(list)
    bad = set()
    for a in bb["annotations"]:
        name = cats.get(a["category_id"])
        if name == "animal":
            boxes[a["image_id"]].append(a["bbox"])
        else:
            bad.add(a["image_id"])            # person / vehicle / herd "group" box
    del bb
    # The species file is 5.5 GB of pretty-printed, not-quite-valid JSON (bare
    # NaN values).  Scan it line by line: categories at the top, then one
    # annotation object per block with "category_id" / "subject_id" (= the
    # bbox file's seq_id) / "image_id".
    import re
    field = re.compile(r'^\s*"(id|name|category_id|subject_id|image_id)":\s*"?([^",]*)"?,?\s*$')
    wanted_seq = {im.get("seq_id") for im in images.values()}
    scats, seq_species = {}, defaultdict(set)
    section, cur = None, {}
    with open(species_file, "r", encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            if section != "annotations":
                if '"categories": [' in line:
                    section = "categories"
                elif '"annotations": [' in line:
                    section = "annotations"
                elif '"images": [' in line:
                    section = "images"
                elif section == "categories":
                    m = field.match(line)
                    if m and m.group(1) == "id" and m.group(2).isdigit():
                        cur = {"id": int(m.group(2))}
                    elif m and m.group(1) == "name" and "id" in cur:
                        scats[cur["id"]] = m.group(2).lower()
                continue
            m = field.match(line)
            if m:
                cur[m.group(1)] = m.group(2)
            elif line.strip().startswith("}"):
                seq = cur.get("subject_id")
                if seq in wanted_seq and "category_id" in cur:
                    seq_species[seq].add(scats.get(int(cur["category_id"]), ""))
                cur = {}
            if n % 20_000_000 == 0:
                print(f"  species file lines read: {n:,}", flush=True)
    print("species categories:", len(scats), "sequences matched:", len(seq_species), flush=True)
    return images, boxes, bad, seq_species


def is_bird(name):
    return any(w in name for w in BIRD_WORDS)


def daylight(im):
    try:
        hour = int(str(im.get("datetime", ""))[11:13])
    except ValueError:
        return True
    return 7 <= hour <= 18


def fetch(item, out_img, out_lbl, max_side, deadline=None):
    image_id, file_name, cls, bxs, w, h = item
    stem = image_id.replace("/", "_")
    dst = os.path.join(out_img, stem + ".jpg")
    if os.path.exists(dst):
        return True
    if deadline is not None and time.time() > deadline:
        return False                      # out of time: the night's training needs the CPU
    try:
        with urllib.request.urlopen(BASE_URL + file_name, timeout=30) as r:
            data = r.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        ih, iw = img.shape[:2]
        scale = min(1.0, max_side / max(ih, iw))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        W, H = (w or iw), (h or ih)
        lines = []
        for x, y, bw, bh in bxs:
            cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
            nw, nh = bw / W, bh / H
            if nw > 0.002 and nh > 0.002:
                lines.append(f"{cls} {min(1,max(0,cx)):.6f} {min(1,max(0,cy)):.6f} {min(1,nw):.6f} {min(1,nh):.6f}")
        if not lines:
            return False
        write_jpeg(dst, img, 90)
        with open(os.path.join(out_lbl, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="training_data/lila")
    ap.add_argument("--out", default="training_data/lila_yolo")
    ap.add_argument("--birds", type=int, default=1200)
    ap.add_argument("--animals", type=int, default=2400)
    ap.add_argument("--per-species", type=int, default=140)
    ap.add_argument("--night-fraction", type=float, default=0.15)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--species", default="",
                    help="comma-separated Snapshot Serengeti species to keep (empty = all), e.g. "
                         "baboon,monkeyvervet,warthog,guineafowl,otherbird")
    ap.add_argument("--small-share", type=float, default=0.0,
                    help="share of each species' daylight picks taken from images whose animals are SMALL "
                         "(every box < --small-area of the picture): far animals, like this camera's")
    ap.add_argument("--small-area", type=float, default=0.01)
    ap.add_argument("--small-birds-only", action="store_true",
                    help="prefer small animals only for bird species.  25/10: a far-herd picture had 2 of "
                         "~9 gazelles boxed - unboxed animals would be taught as background")
    ap.add_argument("--skip-in", nargs="*", default=[],
                    help="LILA folders already downloaded: their images are not picked again")
    ap.add_argument("--dry-run", action="store_true", help="report what would be downloaded, download nothing")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop downloading after this long (0 = no limit); 25/10: the lodge link kept "
                         "dropping, and 3 700 x 30-60 s timeouts would have eaten the training night")
    a = ap.parse_args()
    rng = random.Random(7)
    images, boxes, bad, seq_species = load(a.meta)
    have = {Path(p).stem for d in a.skip_in for p in glob.glob(os.path.join(d, "images", "*.jpg"))}

    def is_small(im):
        area = float(im.get("width") or 2048) * float(im.get("height") or 1536)
        return all(bw * bh / area < a.small_area for _, _, bw, bh in boxes[im["id"]])

    by_species = defaultdict(list)
    for image_id, bxs in boxes.items():
        im = images.get(image_id)
        if im is None or image_id in bad or image_id.replace("/", "_") in have:
            continue
        sps = {s for s in seq_species.get(im.get("seq_id"), set()) if s not in SKIP}
        if len(sps) != 1:
            continue
        by_species[sps.pop()].append(im)
    wanted = {s.strip().lower() for s in a.species.split(",") if s.strip()}
    if wanted:
        missing = sorted(wanted - set(by_species))
        if missing:
            print("species not in the dataset (check the spelling):", missing)
        by_species = {s: v for s, v in by_species.items() if s in wanted}
    birds = {s: v for s, v in by_species.items() if is_bird(s)}
    animals = {s: v for s, v in by_species.items() if not is_bird(s)}
    print("bird species:", {s: len(v) for s, v in birds.items()})
    print("animal species:", {s: len(v) for s, v in animals.items()})
    if a.small_share > 0:
        print("images with only small animals (< %.1f%% of the picture):" % (100 * a.small_area),
              {s: sum(is_small(i) for i in v) for s, v in by_species.items()})

    def pick(groups, total, cap):
        chosen = []
        per = max(cap, total // max(1, len(groups)))
        for s, ims in groups.items():
            day = [i for i in ims if daylight(i)]
            night = [i for i in ims if not daylight(i)]
            rng.shuffle(day)
            rng.shuffle(night)
            n_night = int(per * a.night_fraction)
            if a.small_share > 0 and (is_bird(s) or not a.small_birds_only):
                small = [i for i in day if is_small(i)]
                large = [i for i in day if not is_small(i)]
                k = int((per - n_night) * a.small_share)
                day = small[:k] + large + small[k:]      # the small ones first, up to their share
            chosen += day[:per - n_night] + night[:n_night]
        rng.shuffle(chosen)
        return chosen[:total]

    sel = [(im, 1) for im in pick(birds, a.birds, a.birds)] + [(im, 0) for im in pick(animals, a.animals, a.per_species)]
    if a.dry_run:
        print(f"dry run: would download {len(sel)} images ({sum(c for _, c in sel)} birds), "
              f"{sum(is_small(im) for im, _ in sel)} of them with only small animals; "
              f"{len(have)} already downloaded were skipped")
        return
    out_img, out_lbl = os.path.join(a.out, "images"), os.path.join(a.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    items = [(im["id"], im["file_name"], cls, boxes[im["id"]], im.get("width"), im.get("height")) for im, cls in sel]
    print(f"downloading {len(items)} images ({sum(1 for i in items if i[2] == 1)} birds)")
    ok = 0
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes > 0 else None
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(lambda it: fetch(it, out_img, out_lbl, a.max_side, deadline), items), 1):
            ok += bool(res)
            if i % 200 == 0:
                print(f"  {i}/{len(items)} ok={ok}", flush=True)
    late = deadline is not None and time.time() > deadline
    print(f"done: {ok} images in {out_img}" + (f" (stopped at the {a.max_minutes:.0f}-min limit)" if late else ""))


if __name__ == "__main__":
    main()
