"""Structured wildlife logs plus a bounded human-review / training-data queue.

v3 changes
----------
* Only *confirmed* animals are written to detections.csv (unless
  LOG_UNCONFIRMED_DETECTIONS) so the log reflects what the system really believes.
* Rows carry the camera pose, the water-zone flag and a ``confirmed`` column.
  A CSV written by an older version (different columns) is rotated aside instead
  of being appended to - mixed columns made the old file unreadable.
* The review queue stores the FULL FRAME plus a YOLO pre-label ``.txt`` and a
  ``.json`` with context, not a tiny crop.  Full frames are what a detector can
  actually be trained on (see training/).  Saves are de-duplicated per track.
"""

import csv
import json
import logging
import os
import random
import time
from datetime import datetime, timezone

import cv2

import config
from detection.validator import species_label
from utils.imageio import write_jpeg

os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(config.REVIEW_DIR, exist_ok=True)

_CSV_PATH = os.path.join(config.LOG_DIR, "detections.csv")
_CSV_HEADERS = [
    "timestamp", "track_id", "class_id", "det_confidence", "species",
    "species_confidence", "track_hits", "source", "box",
    "pan", "tilt", "zoom", "in_water_zone", "confirmed",
]
_last_review_save = 0.0
_review_by_key = {}
_last_random_save = 0.0


def disable_console_quick_edit():
    """23/10 13:45-14:43: the tracker stood still for an hour, no error, both the control loop
    and the log stopped in the same second.  With QuickEdit on (the Windows default), a click
    in the console window starts a text selection and every write to the console blocks
    until a key is pressed - the log goes to the console first, so every thread stops.
    Returns True when QuickEdit was switched off for this window."""
    if os.name != "nt":
        return False
    import ctypes
    k32 = ctypes.windll.kernel32
    handle = k32.GetStdHandle(-10)                       # STD_INPUT_HANDLE
    mode = ctypes.c_uint32()
    if not k32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False                                     # no console (hidden job, redirected)
    new = (mode.value & ~0x0040) | 0x0080                # QUICK_EDIT off, EXTENDED_FLAGS on
    return new == mode.value or bool(k32.SetConsoleMode(handle, new))


def setup_logging():
    fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    # 19/10: decisions (pauses, zooms, locks) were only visible in the console window
    try:
        from logging.handlers import RotatingFileHandler
        os.makedirs(config.LOG_DIR, exist_ok=True)
        fh = RotatingFileHandler(os.path.join(config.LOG_DIR, "tracker.log"), maxBytes=5_000_000,
                                 backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        fh.setLevel(logging.INFO)
        logging.getLogger().addHandler(fh)
    except OSError:
        pass
    try:
        if disable_console_quick_edit():
            logging.getLogger("wildlife_tracker.main").info(
                "Console QuickEdit off: a click in this window no longer freezes the tracker")
    except OSError as exc:
        logging.getLogger("wildlife_tracker.main").warning("Could not switch console QuickEdit off: %s", exc)
    if os.path.exists(_CSV_PATH):
        try:
            with open(_CSV_PATH, "r", encoding="utf-8") as f:
                first = f.readline().strip().split(",")
        except OSError:
            first = []
        if first != _CSV_HEADERS:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            legacy = os.path.join(config.LOG_DIR, f"detections_legacy_{stamp}.csv")
            try:
                os.replace(_CSV_PATH, legacy)
                logging.getLogger("wildlife_tracker").info("Old-format detections.csv moved to %s", legacy)
            except OSError:
                pass
    if not os.path.exists(_CSV_PATH):
        with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_CSV_HEADERS)


def log_detection(entry, frame=None, pose=None):
    """Append one animal row (confirmed only, unless configured otherwise)."""
    confirmed = bool(entry.get("track_confirmed", True))
    if not confirmed and not config.LOG_UNCONFIRMED_DETECTIONS:
        return False
    timestamp = datetime.now(timezone.utc).isoformat()
    pan = tilt = zoom = ""
    if pose is not None:
        pan, tilt, zoom = [round(float(v), 4) for v in pose]
    box = entry.get("box", "")
    if box != "":
        box = [round(float(v), 1) for v in box]
    with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            timestamp,
            entry.get("track_id", ""),
            entry.get("class_id", ""),
            round(float(entry.get("det_confidence", 0.0)), 3),
            species_label(entry),
            round(float(entry.get("species_confidence") or 0.0), 3) if entry.get("species_confidence") is not None else "",
            entry.get("track_hits", ""),
            entry.get("source", ""),
            box,
            pan, tilt, zoom,
            int(bool(entry.get("in_water_zone"))),
            int(confirmed),
        ])
    return True


def _yolo_line(box, w, h, cls=0):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _prune_review(max_files=3000):
    try:
        jpgs = sorted(f for f in os.listdir(config.REVIEW_DIR) if f.endswith(".jpg"))
    except OSError:
        return
    for name in jpgs[:max(0, len(jpgs) - max_files)]:
        base = os.path.join(config.REVIEW_DIR, name[:-4])
        for ext in (".jpg", ".txt", ".json"):
            try:
                os.remove(base + ext)
            except OSError:
                pass


def save_review(frame, detections, reason, pose=None, key=None):
    """Save a full frame + YOLO pre-labels for later human review / training.

    ``reason``: 'unidentified' | 'candidate' | 'negative_sample' | 'water'.
    ``key``: dedupe key (usually the track id); one save per key per
    REVIEW_MIN_SECONDS_PER_TRACK.
    """
    global _last_review_save
    if frame is None or not config.REVIEW_SAVE_FULL_FRAMES:
        return None
    now = time.monotonic()
    if now - _last_review_save < 2.0:
        return None
    if key is not None:
        last = _review_by_key.get(key, -1e9)
        if now - last < config.REVIEW_MIN_SECONDS_PER_TRACK:
            return None
        _review_by_key[key] = now
        if len(_review_by_key) > 2000:
            _review_by_key.clear()
    h, w = frame.shape[:2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    base = os.path.join(config.REVIEW_DIR, f"{stamp}_{reason}")
    if not write_jpeg(base + ".jpg", frame, 92):
        return None
    with open(base + ".txt", "w", encoding="utf-8") as f:
        for d in detections:
            f.write(_yolo_line(d["box"], w, h) + "\n")
    meta = {
        "reason": reason,
        "pose": None if pose is None else [round(float(v), 4) for v in pose],
        "detections": [
            {"box": [round(float(v), 1) for v in d["box"]],
             "det_confidence": round(float(d.get("det_confidence", 0.0)), 3),
             "species": d.get("species"),
             "species_confidence": d.get("species_confidence"),
             "track_id": d.get("track_id"),
             "confirmed": d.get("track_confirmed"),
             "in_water_zone": d.get("in_water_zone")}
            for d in detections
        ],
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    _last_review_save = now
    if random.random() < 0.02:
        _prune_review()
    return base + ".jpg"


def maybe_save_negative_sample(frame, pose=None, every_seconds=600.0):
    """Occasionally keep an ordinary frame (hard-negative / background examples)."""
    global _last_random_save
    now = time.monotonic()
    if now - _last_random_save < every_seconds:
        return None
    _last_random_save = now
    return save_review(frame, [], "negative_sample", pose=pose)
