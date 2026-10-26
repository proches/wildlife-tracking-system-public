"""Watchdog: the tracker must run, whatever happened.

Scheduled every 15 minutes (task MeliaWildlifeTrackerWatchdog, through
tools/run_hidden.vbs + tools/run_watchdog.bat).  On 23/10/2026:
- the night training was killed at about 07:46 before its own last step (restart
  the tracker) could run, and the system stayed off until 08:32;
- 13:45-14:43 the running tracker froze without an error (most likely a click in
  its console window with QuickEdit on) until someone restarted it by hand.

* tracker running and its control loop writes logs/ptz_trace.csv -> nothing to do;
* tracker running but no ptz_trace.csv line for FROZEN_MINUTES -> frozen: restarted;
* model check (finalize) running -> wait for it, at most --wait-minutes: it may be
  installing the new model, and it restarts the tracker itself;
* any other night job running    -> stopped, but only between 05:30 (the training
  deadline) and 12:00; at other hours a running night job is left alone;
* then the tracker is started in its own window and checked 20 s later.

    .venv313\\Scripts\\python.exe tools\\tracker_watchdog.py           # what the task runs
    .venv313\\Scripts\\python.exe tools\\tracker_watchdog.py --check   # report only
Log: logs/watchdog.log.  Switch it off: schtasks /Change /TN MeliaWildlifeTrackerWatchdog /DISABLE
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "watchdog.log"
TRACKER = "main.py"
FINALIZE = "finalize_model.py"
NIGHT_JOBS = {"round2.py", "round3.py", "round4.py", "train_custom_yolo.py", "teacher_label.py",
              "assemble_training_set.py", "build_lila_serengeti.py", "harvest_round2.py"}
HEARTBEAT = ROOT / "logs" / "ptz_trace.csv"     # one line per second from the control loop
FROZEN_MINUTES = 5.0


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def python_processes():
    """-> [(pid, command line, seconds since it started)] of every python.exe."""
    ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          "Select-Object ProcessId, CommandLine, "
          "@{n='Age';e={((Get-Date) - $_.CreationDate).TotalSeconds}} | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                         text=True, timeout=120, check=True).stdout.strip()
    rows = json.loads(out) if out else []
    if isinstance(rows, dict):
        rows = [rows]
    return [(int(r["ProcessId"]), r.get("CommandLine") or "", float(r.get("Age") or 0.0)) for r in rows]


def scripts(cmd):
    """File names on a command line: '... training\\round3.py --reuse' -> {..., 'round3.py', ...}."""
    return {tok.strip('"').replace("\\", "/").rsplit("/", 1)[-1] for tok in cmd.split()}


def plan(procs):
    """procs: [(pid, command line)] -> ("ok", []) the tracker runs;
    ("wait", pids) a model check is running; ("start", pids) stop these night jobs, start the tracker."""
    if any(TRACKER in scripts(c) for _, c, *_ in procs):
        return "ok", []
    finalize = [p for p, c, *_ in procs if FINALIZE in scripts(c)]
    if finalize:
        return "wait", finalize
    return "start", [p for p, c, *_ in procs if scripts(c) & NIGHT_JOBS]


def frozen(procs, heartbeat_mtime, now, minutes=FROZEN_MINUTES):
    """A tracker that has run longer than `minutes` but whose control loop wrote no
    heartbeat line for `minutes` is frozen (a just-started one is still loading models)."""
    ages = [rest[0] for _, c, *rest in procs if TRACKER in scripts(c) and rest]
    if not ages or min(ages) < minutes * 60:
        return False
    return heartbeat_mtime is None or now - heartbeat_mtime > minutes * 60


def morning(t=None):
    t = time.localtime() if t is None else t
    return (5, 30) <= (t.tm_hour, t.tm_min) and t.tm_hour < 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--wait-minutes", type=float, default=45.0,
                    help="how long a running model check may take before it is stopped")
    a = ap.parse_args()
    give_up = time.time() + a.wait_minutes * 60
    waited = grace_done = False
    sys.path.insert(0, str(ROOT))
    from training.round2 import start_tracker, stop_tracker
    procs = python_processes()
    beat = HEARTBEAT.stat().st_mtime if HEARTBEAT.exists() else None
    if frozen(procs, beat, time.time()):
        quiet = (time.time() - beat) / 60 if beat else float("inf")
        if a.check:
            log(f"check: tracker FROZEN - no control-loop line in ptz_trace.csv for {quiet:.0f} min")
            return 0
        log(f"tracker frozen: no control-loop line in ptz_trace.csv for {quiet:.0f} min - restarting it")
        stop_tracker(log)
        time.sleep(3)
    while True:
        procs = python_processes()
        what, pids = plan(procs)
        if what == "ok":
            log("tracker running - nothing to do" + (" (it was started after the model check)" if waited else ""))
            return 0
        if a.check:
            log(f"check: tracker NOT running; next step would be '{what}' (pids {pids})")
            return 0
        if what == "start" and waited and not grace_done:
            grace_done = True          # the model check just ended: its job restarts the tracker
            time.sleep(60)             # itself - never start a second one next to it
            continue
        if what != "wait" or time.time() > give_up:
            break
        if not waited:
            log(f"tracker not running; the model check is (pid {pids}) - waiting for it")
            waited = True
        time.sleep(30)
    if pids and not morning():
        # every 15 min, also during the night training: before its 05:30 deadline a
        # running night job is left alone
        log(f"tracker not running, a night job is (pids {pids}) and it is before 05:30 - left alone")
        return 0
    if what == "wait":
        log(f"the model check is still running after {a.wait_minutes:.0f} min - stopping it")
        pids = [p for p, c in procs if scripts(c) & (NIGHT_JOBS | {FINALIZE})]
    for pid in pids:
        rc = subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log(f"night job pid {pid} stopped (taskkill exit {rc}; 128 = already gone with its parent)")
    if pids:
        best = sorted(glob.glob(str(ROOT / "runs" / "**" / "melia_v*" / "weights" / "best.pt"),
                                recursive=True), key=os.path.getmtime)
        if best:
            log("to judge the weights it saved: .venv313\\Scripts\\python.exe training\\finalize_model.py "
                f"--weights {best[-1]} --data training_data/dataset3/data.yaml "
                "--baseline models/melia_wildlife.pt   (stop the tracker first)")
    stop_tracker(log)              # a crashed tracker leaves its window waiting for a key
    start_tracker(log)
    time.sleep(20)
    if plan(python_processes())[0] == "ok":
        log("tracker is running")
        return 0
    log("tracker did NOT start - read the error in its window (run_tracker.bat pauses on exit)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
