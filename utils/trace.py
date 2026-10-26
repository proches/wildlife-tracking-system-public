"""Structured, rate-limited pipeline trace (logs/pipeline_trace.jsonl).

One JSON object per line.  Kinds:
* ``ai``    - one AI result: frame size, capture age, inference + end-to-end
              latency, raw / kept / rejected detections, the boxes, state and
              the selected target with the reason;
* ``ptz``   - one tracking PTZ command: who issued it (ai / flow / feedforward /
              coast), target centre, normalised errors, speeds, pulse, ok/error;
* ``state`` - every tracking-state transition with its reason (never limited).

Rate limits keep the file small: per kind a minimum interval, except records
that carry detections / transitions.
"""

import json
import os
import threading
import time

import config


class PipelineTrace:
    def __init__(self, path=None):
        self.path = path or config.PIPELINE_TRACE_FILE
        self._lock = threading.Lock()
        self._last = {}
        self._f = None

    def _open(self):
        if self._f is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._f = open(self.path, "a", encoding="utf-8")
        return self._f

    def log(self, kind, record, min_interval=0.0, force=False):
        now = time.time()
        with self._lock:
            if not force and now - self._last.get(kind, 0.0) < min_interval:
                return False
            self._last[kind] = now
            rec = {"t": round(now, 3), "kind": kind}
            rec.update(record)
            try:
                f = self._open()
                f.write(json.dumps(rec, default=_plain) + "\n")
                f.flush()
            except OSError:
                return False
            return True

    def close(self):
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None


def _plain(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


class NullTrace:
    """Drop-in when tracing is off (tests)."""

    def log(self, *args, **kwargs):
        return False

    def close(self):
        pass
