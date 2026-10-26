"""Third training round, unattended (overnight 22-23/10/2026).

    .venv313\\Scripts\\python.exe training\\round3.py            # everything
    .venv313\\Scripts\\python.exe training\\round3.py --reuse    # from the dataset on (23/10 night)
    .venv313\\Scripts\\python.exe training\\round3.py --check    # can the scheduled run start?

23/10: the first run (22/10 night) never finished.  The harvest took every clip
frame (4048 frames, 10 h of teacher labels) and the set grew to 26 000 tiles, about
3.1 h per epoch.  Its console was then closed at about 07:46 (exit 0xC000013A), so the
tracker restart at the end never ran.  The re-run reuses that work.  It keeps
every 6th clip frame and 30 % of the empty training tiles, and lets the
hand-checked crops replace the teacher labels of the same pictures.  It trains
only until --deadline minus 1 h, and runs without a window (tools/run_hidden.vbs).

Round 2 was rejected (mAP50 0.640 < 0.663, 3x more false boxes on vegetation) and
could not learn the small primates: their frames had no labels (the teacher could
not see them either).  Round 3 adds what round 2 lacked:
* hand-checked examples from THIS camera (training/verified_21oct.json ->
  training_data/site3v): 25 animals + 7 birds the system missed or doubted (far
  buffalo, waterbuck, primates in shade, birds) and 35 acacias / trunks / dark
  canopy it called animals - native-resolution 640-px crops;
* LILA Snapshot Serengeti images of the species this camera sees and the models
  miss (baboon, vervet, warthog, waterbuck ... and birds), downloaded at night
  (the stream needs the bandwidth by day), 960 px so the animals are smaller;
* the frames saved since 21/10 (training/harvest_round2.py, teacher-labelled).
Same gate as round 2: the new model replaces the current one only if it is not
worse on the same validation set (finalize_model.py).  Log: logs/round3.log.
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = str(ROOT / ".venv313" / "Scripts" / "python.exe")
LOG = ROOT / "logs" / "round3.log"

LILA_SPECIES = ("baboon,monkeyvervet,warthog,waterbuck,buffalo,bushbuck,impala,dikdik,reedbuck,"
                "mongoose,hyenaspotted,jackal,guineafowl,otherbird,koribustard,secretarybird,vulture,ostrich")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, step):
    log(f"--- {step}: {' '.join(args)}")
    with open(LOG, "a", encoding="utf-8") as f:
        rc = subprocess.call(args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    log(f"--- {step}: exit {rc}")
    return rc


def subsample_site3h(src="training_data/site3h/images", dst="training_data/site3s/images", clip_every=6):
    """Every labelled frame except clips + every Nth clip frame (clip frames are 1 s apart and
    nearly identical: 3420 of them made the 23/10 set 26 000 tiles - 3.1 h per epoch)."""
    import shutil
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    frames = sorted(glob.glob(os.path.join(src, "*.jpg")))
    kept = clips = 0
    for p in frames:
        lbl = os.path.splitext(p)[0] + ".txt"
        if not os.path.exists(lbl):
            continue                       # left out by the teacher (unsure): never train on it
        if "clip" in os.path.basename(p):
            clips += 1
            if clips % clip_every:
                continue
        shutil.copy2(p, dst)
        shutil.copy2(lbl, dst)
        kept += 1
    log(f"site3s: {kept} labelled frames (every {clip_every}th of {clips} clip frames)")
    return kept


def training_hours(deadline, cap, now=None, margin=1.0):
    """Hours the training may run so that everything ends by `deadline` (HH:MM, the next one
    to come).  Round 2 ran 20 min past its --time (last validation) and finalize took
    18 min: hence the 1-h margin."""
    now = time.time() if now is None else now
    hh, mm = (int(v) for v in deadline.split(":"))
    t = time.localtime(now)
    end = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, hh, mm, 0, 0, 0, -1))
    if end <= now:
        end += 24 * 3600
    return min(cap, (end - now) / 3600.0 - margin)


def main():
    from training.round2 import start_tracker, stop_tracker
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0, help="training wall-clock cap")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--deadline", default="05:30",
                    help="HH:MM: training ends by then (finalize + tracker restart follow) - the "
                         "system must run in the morning (23/10: the job would have run until noon)")
    ap.add_argument("--reuse", action="store_true",
                    help="LILA, harvest and teacher labels are done: start at the dataset")
    ap.add_argument("--lila-max", type=int, default=1000, help="LILA training images per LILA folder")
    ap.add_argument("--no-stop", action="store_true", help="do not stop/restart the tracker")
    ap.add_argument("--dataset-only", action="store_true",
                    help="build the dataset and report its size and epoch time, nothing else "
                         "(the tracker keeps running)")
    ap.add_argument("--check", action="store_true",
                    help="only verify that the scheduled run can start, then exit")
    a = ap.parse_args()
    a.no_stop = a.no_stop or a.dataset_only
    if a.check:
        import ultralytics  # noqa: F401
        need = ["models/melia_wildlife.pt", "training/verified_21oct.json",
                "training_data/lila/SnapshotSerengetiBboxes_20190903.json"]
        if a.reuse:                 # what the first run made and the re-run starts from
            need += ["training_data/lila3_yolo/images", "training_data/site3h/images"]
        missing = [p for p in need if not os.path.exists(p)]
        ok = not missing
        log(f"check {'ok' if ok else 'FAILED'}: python {sys.version.split()[0]}, cwd {os.getcwd()}, "
            f"{len(need)} inputs present" + (f", missing {missing}" if missing else ""))
        return 0 if ok else 1
    log("=== round 3 start (dataset only) ===" if a.dataset_only else "=== round 3 start ===")
    if not a.no_stop:
        stop_tracker(log)
    try:
        if not a.reuse:
            # LILA first: if the download fails (no internet), train without it
            if run([PY, "training/build_lila_serengeti.py", "--meta", "training_data/lila",
                    "--out", "training_data/lila3_yolo", "--species", LILA_SPECIES, "--birds", "600",
                    "--animals", "1000", "--per-species", "150", "--max-side", "960", "--workers", "8"],
                   "LILA targeted") != 0:
                log("LILA download failed - training without the new LILA images")
            run([PY, "training/harvest_round2.py", "--out", "training_data/site3h/images", "--since", "20261021",
                 "--every", "6"], "harvest since 21/10")
            if run([PY, "training/teacher_label.py", "training_data/site3h/images", "--imgsz", "1280"],
                   "teacher labels") != 0:
                log("teacher labelling failed - the new frames are left out")
        subsample_site3h()
        if run([PY, "training/build_site_verified.py"], "hand-verified crops") != 0:
            log("verified crops missing - keeping the current model")
            return 1
        if run([PY, "training/assemble_training_set.py", "--out", "training_data/dataset3",
                "--lila", "training_data/lila_yolo", "training_data/lila3_yolo",
                "--site", "training_data/site/images", "training_data/site2/images",
                "training_data/site3s/images", "--verified", "training_data/site3v/images",
                "--site-repeat", "1", "--lila-max", str(a.lila_max), "--empty-keep", "0.3"],
               "assemble") != 0:
            return 1
        n_train = len(os.listdir("training_data/dataset3/images/train"))
        n_val = len(os.listdir("training_data/dataset3/images/val"))
        # measured on this PC: 3.4-3.5 s per batch of 8 (23/10), 53.5 min per epoch for
        # 6506 train + 2019 val images (round 2)
        epoch_min = (0.43 * n_train + 0.2 * n_val) / 60.0
        log(f"dataset3: {n_train} train / {n_val} val images, about {epoch_min:.0f} min per epoch")
        if a.dataset_only:
            return 0
        hours = training_hours(a.deadline, a.hours)
        if hours < 1.0:
            log(f"only {hours:.1f} h left before {a.deadline} - no training tonight, the current model stays")
            return 1
        log(f"training for at most {hours:.1f} h (deadline {a.deadline}): about "
            f"{hours * 60 / epoch_min:.0f} epochs")
        name = time.strftime("melia_v3_%Y%m%d")
        run([PY, "training/train_custom_yolo.py", "--data", "training_data/dataset3/data.yaml",
             "--preset", "cpu", "--device", "cpu", "--model", "models/melia_wildlife.pt",
             "--epochs", str(a.epochs), "--time", f"{hours:.2f}", "--name", name], "train")
        best = sorted(glob.glob(f"runs/**/{name}*/weights/best.pt", recursive=True), key=os.path.getmtime)
        if not best:
            log("no trained weights found - keeping the current model")
            return 1
        rc = run([PY, "training/finalize_model.py", "--weights", best[-1],
                  "--data", "training_data/dataset3/data.yaml", "--baseline", "models/melia_wildlife.pt"],
                 "finalize")
        log("NEW MODEL ACCEPTED" if rc == 0 else "new model NOT better - the current model stays")
        return rc
    finally:
        if not a.no_stop:
            start_tracker(log)
        log("=== round 3 end ===")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
