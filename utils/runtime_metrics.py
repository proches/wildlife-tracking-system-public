"""Runtime metrics and field-performance telemetry."""

import json
import os
import time
from collections import Counter, deque
from datetime import datetime, timezone

import config


class RuntimeMetrics:
    def __init__(self):
        self.frame_times = deque(maxlen=120)
        self.inference_times = deque(maxlen=120)
        self.detection_counts = deque(maxlen=120)
        self.species = Counter()
        self.target_lock_count = 0
        self.target_lost_count = 0
        self.reacquired_count = 0
        self.ptz_boundary_events = 0
        self.stream_reconnects = 0
        self._last_lock = False
        self._last_searching = False
        self.started = time.monotonic()
        self.frames = 0

    def record_frame(self, frame_seconds, inference_seconds, detections):
        self.frame_times.append(max(1e-6, frame_seconds))
        self.inference_times.append(max(0.0, inference_seconds))
        self.detection_counts.append(len(detections))
        self.frames += 1
        for d in detections:
            label = d.get("species") or "unidentified"
            if label != "unidentified":
                self.species[label] += 1

    def note_tracker(self, locked, searching, boundary_event=False):
        if locked and not self._last_lock:
            self.target_lock_count += 1
            if self.frames > 0:
                self.reacquired_count += 1 if self.target_lost_count else 0
        if self._last_lock and not locked:
            self.target_lost_count += 1
        if boundary_event:
            self.ptz_boundary_events += 1
        self._last_lock = locked
        self._last_searching = searching

    def note_reconnect(self):
        self.stream_reconnects += 1

    def snapshot(self):
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0.0
        inference_ms = (sum(self.inference_times) / len(self.inference_times) * 1000.0) if self.inference_times else 0.0
        det_avg = (sum(self.detection_counts) / len(self.detection_counts)) if self.detection_counts else 0.0
        uptime = max(0.0, time.monotonic() - self.started)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fps": round(fps, 2),
            "inference_ms": round(inference_ms, 2),
            "detections_per_frame": round(det_avg, 2),
            "frames": self.frames,
            "uptime_seconds": round(uptime, 1),
            "target_locks": self.target_lock_count,
            "target_losses": self.target_lost_count,
            "reacquisitions": self.reacquired_count,
            "ptz_boundary_events": self.ptz_boundary_events,
            "stream_reconnects": self.stream_reconnects,
            "top_species": self.species.most_common(10),
        }

    def write(self):
        path = config.METRICS_JSONL
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self.snapshot()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return data
