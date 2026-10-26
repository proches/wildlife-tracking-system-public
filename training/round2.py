"""Second training round, unattended (overnight): the site model learns this camera's
baboons, buffalo, warthogs ... from the frames the running system saved.

    .venv313\\Scripts\\python.exe training\\round2.py            # everything
    .venv313\\Scripts\\python.exe training\\round2.py --hours 6  # training time cap

Steps (each logged to logs/round2.log):
1. stop the tracker (the i5 cannot train and track at the same time);
2. harvest new frames (training/harvest_round2.py);
3. teacher labels at native resolution (MegaDetector + SpeciesNet "blank" veto);
4. assemble LILA Serengeti + round-1 site frames + new frames -> training_data/dataset2;
5. fine-tune the CURRENT site model (models/melia_wildlife.pt), CPU preset, time-capped;
6. finalize: the new model replaces the current one ONLY if, on the same validation
   set, it finds at least as many of this camera's animals, keeps mAP50 and adds no
   false boxes on vegetation (the old model is backed up);
7. restart the tracker.
Any failure -> the current model stays and the tracker is restarted.
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
LOG = ROOT / "logs" / "round2.log"


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


def stop_tracker(log_fn=None):
    ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          "Where-Object { $_.CommandLine -like '*main.py*' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }; "
          "Get-CimInstance Win32_Process -Filter \"name='cmd.exe'\" | "
          "Where-Object { $_.CommandLine -like '*run_tracker.bat*' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }")
    subprocess.call(["powershell", "-NoProfile", "-Command", ps])
    (log_fn or log)("tracker stopped")


def start_tracker(log_fn=None):
    # its own console window, shown even when the caller runs hidden (tools/run_hidden.vbs:
    # on 23/10 the visible training console was closed and this restart never ran)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 1                                    # SW_SHOWNORMAL
    subprocess.Popen(["cmd", "/c", str(ROOT / "run_tracker.bat")], cwd=ROOT,
                     creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=si)
    (log_fn or log)("tracker started")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=7.0, help="training wall-clock cap")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--no-stop", action="store_true", help="do not stop/restart the tracker")
    ap.add_argument("--check", action="store_true",
                    help="only verify that the scheduled run can start (python, venv, folder, ultralytics) and exit")
    a = ap.parse_args()
    if a.check:
        import ultralytics  # noqa: F401  (the training dependency must import)
        log(f"check ok: python {sys.version.split()[0]} at {sys.executable}, cwd {os.getcwd()}, "
            f"models/melia_wildlife.pt present: {os.path.exists('models/melia_wildlife.pt')}")
        return 0
    log("=== round 2 start ===")
    if not a.no_stop:
        stop_tracker()
    try:
        run([PY, "training/harvest_round2.py"], "harvest")
        # 1280 px: ~7 s per frame (1920 px took ~20 s - ~4 h for the whole set)
        if run([PY, "training/teacher_label.py", "training_data/site2/images", "--imgsz", "1280",
                "--sheet", "training_data/site2/teacher_sheet.jpg"], "teacher labels") != 0:
            log("teacher labelling failed - keeping the current model")
            return 1
        if run([PY, "training/assemble_training_set.py", "--out", "training_data/dataset2",
                "--site", "training_data/site/images", "training_data/site2/images",
                "--site-repeat", "1", "--lila-max", "1500"], "assemble") != 0:
            return 1
        name = time.strftime("melia_v2_%Y%m%d")
        run([PY, "training/train_custom_yolo.py", "--data", "training_data/dataset2/data.yaml",
             "--preset", "cpu", "--device", "cpu", "--model", "models/melia_wildlife.pt",
             "--epochs", str(a.epochs), "--time", str(a.hours), "--name", name], "train")
        best = sorted(glob.glob(f"runs/**/{name}*/weights/best.pt", recursive=True), key=os.path.getmtime)
        if not best:
            log("no trained weights found - keeping the current model")
            return 1
        rc = run([PY, "training/finalize_model.py", "--weights", best[-1],
                  "--data", "training_data/dataset2/data.yaml", "--baseline", "models/melia_wildlife.pt"],
                 "finalize")
        log("NEW MODEL ACCEPTED" if rc == 0 else "new model NOT better - the current model stays")
        return rc
    finally:
        if not a.no_stop:
            start_tracker()
        log("=== round 2 end ===")


if __name__ == "__main__":
    sys.exit(main())
