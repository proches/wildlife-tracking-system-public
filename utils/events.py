"""Wildlife event detection and optional highlight recording."""

import csv
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone

import cv2

import config

logger = logging.getLogger("wildlife_tracker.events")


class EventClipRecorder:
    def __init__(self):
        self.buffer = deque(maxlen=max(1, int(config.EVENT_PREBUFFER_SECONDS * config.EVENT_CLIP_FPS)))
        self.writer = None
        self.active = False
        self.started = 0.0
        self.last_activity = 0.0
        self.event_name = ""
        self.path = None

    def push(self, frame):
        if not config.EVENT_CLIPS_ENABLED:
            return
        self.buffer.append(frame.copy())
        if self.writer is not None:
            self.writer.write(self._resize(frame))

    def start(self, event_name, frame):
        if not config.EVENT_CLIPS_ENABLED or self.active:
            return
        os.makedirs(config.EVENT_CLIP_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = os.path.join(config.EVENT_CLIP_DIR, f"{stamp}_{event_name}.mp4")
        first = self._resize(frame)
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, config.EVENT_CLIP_FPS, (w, h))
        if not self.writer.isOpened():
            logger.warning("Could not open event video writer; disabling this clip")
            self.writer = None
            return
        for item in self.buffer:
            self.writer.write(self._resize(item))
        self.active = True
        self.started = time.monotonic()
        self.last_activity = self.started

    def update_activity(self, active):
        if active:
            self.last_activity = time.monotonic()

    def maybe_finish(self):
        if self.writer is None:
            return
        if self.active and time.monotonic() - self.last_activity >= config.EVENT_POSTBUFFER_SECONDS:
            self.writer.release()
            logger.info("Saved wildlife highlight: %s", self.path)
            self.writer = None
            self.active = False
            self.path = None

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.active = False

    @staticmethod
    def _resize(frame):
        max_width = config.EVENT_MAX_CLIP_WIDTH
        if frame.shape[1] <= max_width:
            return frame
        scale = max_width / frame.shape[1]
        return cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


class WildlifeEventEngine:
    def __init__(self):
        self.active_tracks = {}
        self.last_event_time = {}
        self.recorder = EventClipRecorder()
        self._ensure_csv()

    def update(self, tracks, tracker_locked_id, frame):
        now = time.monotonic()
        current = {t["track_id"]: t for t in tracks if t.get("track_confirmed")}
        events = []
        for tid, track in current.items():
            species = track.get("species") or "unidentified"
            confidence = float(track.get("det_confidence", 0.0))
            if tid not in self.active_tracks and confidence >= config.EVENT_MIN_CONFIDENCE:
                if self._cooldown_ok(species, now):
                    events.append(self._emit("SIGHTING_STARTED", track, tid))
                    self.recorder.start(self._safe_name(species), frame)
                    self.last_event_time[species] = now
            previous = self.active_tracks.get(tid)
            if previous and previous.get("species") != species and species != "unidentified":
                events.append(self._emit("SPECIES_STABLE", track, tid))
            self.active_tracks[tid] = dict(track)
        ended = [tid for tid in self.active_tracks if tid not in current]
        for tid in ended:
            events.append({"timestamp": self._timestamp(), "type": "TRACK_ENDED", "track_id": tid,
                           "species": self.active_tracks[tid].get("species", "unidentified")})
            del self.active_tracks[tid]

        locked = tracker_locked_id in current if tracker_locked_id is not None else False
        self.recorder.update_activity(bool(current))
        if current and not locked:
            # Keep the recording useful even if the PTZ has not committed to a target yet.
            pass
        self.recorder.maybe_finish()
        self._write_events(events)
        return events

    def push_frame(self, frame):
        self.recorder.push(frame)

    def close(self):
        self.recorder.close()

    def _cooldown_ok(self, species, now):
        return now - self.last_event_time.get(species, -1e9) >= config.EVENT_COOLDOWN_SECONDS

    @staticmethod
    def _timestamp():
        return datetime.now(timezone.utc).isoformat()

    def _emit(self, event_type, track, tid):
        return {
            "timestamp": self._timestamp(),
            "type": event_type,
            "track_id": tid,
            "species": track.get("species") or "unidentified",
            "confidence": round(float(track.get("det_confidence", 0.0)), 3),
        }

    @staticmethod
    def _safe_name(value):
        return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)[:40] or "wildlife"

    @staticmethod
    def _ensure_csv():
        os.makedirs(os.path.dirname(config.EVENTS_CSV) or ".", exist_ok=True)
        if os.path.exists(config.EVENTS_CSV):
            return
        with open(config.EVENTS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["timestamp", "type", "track_id", "species", "confidence"]).writeheader()

    @staticmethod
    def _write_events(events):
        if not events:
            return
        with open(config.EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "type", "track_id", "species", "confidence"])
            writer.writerows(events)
