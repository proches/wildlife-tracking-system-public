"""Fourth training round, unattended (night 25-26/10/2026).

    .venv313\\Scripts\\python.exe training\\round4.py                # everything (scheduled task)
    .venv313\\Scripts\\python.exe training\\round4.py --check        # can the scheduled run start?
    .venv313\\Scripts\\python.exe training\\round4.py --dataset-only # build the set, no training

Round 3 finished (23-24/10) but was not accepted: mAP50 0.690 vs 0.718.  It was better
on vegetation (107 vs 121 false boxes on 1354 empty tiles), equal on animals, worse
on birds (0.679 vs 0.738).  Two measured causes and what this round changes:
* the learning rate.  Ultralytics "auto" chose AdamW lr 0.001667 with 3 warm-up epochs
  of 7 - made for training from COCO weights.  mAP50 was best after epoch 1 (0.759)
  and then fell.  Now train_custom_yolo preset "cpu_finetune": AdamW lr 0.0003, a
  half-epoch warm-up, weights saved every epoch, and the epoch with the best mAP50
  (the acceptance measure) goes to the check, not the last one.
* the Snapshot Serengeti share (LILA, human-drawn boxes - better labels than our
  MegaDetector teacher, which contradicted 14 of 67 hand-checked site pictures).
  Round 3 had 2 000 LILA pictures and no gazelles at all; the operator saw far
  gazelles the system missed (23/10 09:28).  This round downloads gazelles, topi,
  hartebeest, eland, wildebeest, zebra ... and more birds, half of each species from
  pictures whose animals are SMALL (far, like this camera's) - for BIRDS only: a far-herd
  picture had 2 of ~9 gazelles boxed, and unboxed animals would be taught as background -
  skipping the ones we have, and stopping after 45 min (the link kept dropping 25/10).
The acceptance check now also reports mAP50 separately for this camera and for the
Serengeti camera traps, and the camera's own mAP50 must not drop.  Log: logs/round4.log.
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".venv313" / "Scripts" / "python.exe")
LOG = ROOT / "logs" / "round4.log"

LILA4_SPECIES = ("gazellethomsons,gazellegrants,impala,topi,hartebeest,eland,wildebeest,zebra,warthog,"
                 "buffalo,giraffe,elephant,waterbuck,reedbuck,dikdik,baboon,monkeyvervet,hyenaspotted,"
                 "jackal,lionfemale,cheetah,guineafowl,otherbird,koribustard,secretarybird,ostrich")
# measured on this PC: finalize took 21 min for 2 457 validation pictures (round 3); the
# per-source passes add about one more pass per model; the last validation adds ~20 min
MARGIN_HOURS = 1.4


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


def best_epoch_weights(run_dir):
    """-> (weights path, epoch, mAP50) of the epoch with the best validation mAP50.
    results.csv counts epochs from 1, the trainer saves weights/epoch{epoch - 1}.pt
    (ultralytics/engine/trainer.py save_metrics / save_model)."""
    best = None
    path = os.path.join(run_dir, "results.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}
                try:
                    epoch, m = int(float(row["epoch"])), float(row["metrics/mAP50(B)"])
                except (KeyError, ValueError):
                    continue
                w = os.path.join(run_dir, "weights", f"epoch{epoch - 1}.pt")
                if os.path.exists(w) and (best is None or m > best[2]):
                    best = (w, epoch, m)
    if best is None and os.path.exists(os.path.join(run_dir, "weights", "best.pt")):
        best = (os.path.join(run_dir, "weights", "best.pt"), None, None)
    return best


def main():
    from training.round2 import start_tracker, stop_tracker
    from training import round3
    round3.LOG = LOG                       # its helpers log here
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=9.0, help="training wall-clock cap")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--deadline", default="05:30")
    ap.add_argument("--lila-max", type=int, default=1500, help="LILA training pictures per LILA folder")
    ap.add_argument("--no-download", action="store_true", help="use only the LILA pictures already here")
    ap.add_argument("--no-stop", action="store_true", help="do not stop/restart the tracker")
    ap.add_argument("--dataset-only", action="store_true")
    ap.add_argument("--check", action="store_true", help="only verify that the scheduled run can start")
    a = ap.parse_args()
    a.no_stop = a.no_stop or a.dataset_only
    if a.check:
        import ultralytics  # noqa: F401
        need = ["models/melia_wildlife.pt", "training/verified_21oct.json", "training_data/site3h/images",
                "training_data/lila_yolo/images", "training_data/lila3_yolo/images",
                "training_data/lila/SnapshotSerengetiBboxes_20190903.json",
                "training_data/lila/SnapshotSerengeti_S1-11_v2.1.json"]
        missing = [p for p in need if not os.path.exists(p)]
        log(f"check {'ok' if not missing else 'FAILED'}: python {sys.version.split()[0]}, "
            f"{len(need)} inputs" + (f", missing {missing}" if missing else " present"))
        return 0 if not missing else 1
    log("=== round 4 start (dataset only) ===" if a.dataset_only else "=== round 4 start ===")
    if not a.no_stop:
        stop_tracker(log)
    try:
        if not a.no_download:
            if run([PY, "training/build_lila_serengeti.py", "--meta", "training_data/lila",
                    "--out", "training_data/lila4_yolo", "--species", LILA4_SPECIES,
                    "--birds", "1500", "--animals", "2200", "--per-species", "110", "--max-side", "960",
                    "--small-share", "0.5", "--small-birds-only",
                    "--skip-in", "training_data/lila_yolo", "training_data/lila3_yolo",
                    "--workers", "8", "--max-minutes", "45"],
                   "LILA Snapshot Serengeti (gazelles, plains species, birds)") != 0:
                log("LILA download failed - training with the LILA pictures already here")
        round3.subsample_site3h()
        if run([PY, "training/build_site_verified.py"], "hand-verified crops") != 0:
            log("verified crops missing - keeping the current model")
            return 1
        if run([PY, "training/assemble_training_set.py", "--out", "training_data/dataset4",
                "--lila", "training_data/lila_yolo", "training_data/lila3_yolo", "training_data/lila4_yolo",
                "--site", "training_data/site/images", "training_data/site2/images",
                "training_data/site3s/images", "--verified", "training_data/site3v/images",
                "--site-repeat", "1", "--lila-max", str(a.lila_max), "--empty-keep", "0.2"],
               "assemble") != 0:
            return 1
        n_train = len(os.listdir("training_data/dataset4/images/train"))
        n_val = len(os.listdir("training_data/dataset4/images/val"))
        epoch_min = (0.43 * n_train + 0.2 * n_val) / 60.0       # measured rates, see round3.py
        log(f"dataset4: {n_train} train / {n_val} val images, about {epoch_min:.0f} min per epoch")
        if a.dataset_only:
            return 0
        hours = round3.training_hours(a.deadline, a.hours, margin=MARGIN_HOURS)
        if hours < 1.0:
            log(f"only {hours:.1f} h left before {a.deadline} - no training tonight, the current model stays")
            return 1
        log(f"training for at most {hours:.1f} h (deadline {a.deadline}): about {hours * 60 / epoch_min:.0f} epochs")
        name = time.strftime("melia_v4_%Y%m%d")
        run([PY, "training/train_custom_yolo.py", "--data", "training_data/dataset4/data.yaml",
             "--preset", "cpu_finetune", "--device", "cpu", "--model", "models/melia_wildlife.pt",
             "--epochs", str(a.epochs), "--time", f"{hours:.2f}", "--name", name], "train")
        runs = sorted(glob.glob(f"runs/**/{name}*/results.csv", recursive=True), key=os.path.getmtime)
        pick = best_epoch_weights(os.path.dirname(runs[-1])) if runs else None
        if not pick:
            log("no trained weights found - keeping the current model")
            return 1
        log(f"weights for the check: {pick[0]} (epoch {pick[1]}, validation mAP50 {pick[2]})")
        rc = run([PY, "training/finalize_model.py", "--weights", pick[0],
                  "--data", "training_data/dataset4/data.yaml", "--baseline", "models/melia_wildlife.pt"],
                 "finalize")
        log("NEW MODEL ACCEPTED" if rc == 0 else "new model NOT better - the current model stays")
        return rc
    finally:
        if not a.no_stop:
            start_tracker(log)
        log("=== round 4 end ===")


if __name__ == "__main__":
    sys.exit(main())
