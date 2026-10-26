"""Main runtime for the Melia Serengeti autonomous wildlife camera (v3, threaded).

Threads
-------
* FrameGrabber    - reads RTSP continuously, always keeps only the NEWEST frame.
* DetectionWorker - runs the (slow, CPU) AI on the newest *settled* frame.
* Control loop    - 8 Hz: reads the real PTZ pose, runs patrol / tracking, feeds
                    the AI worker, logs.  Never waits for the AI.
* Main thread     - overlay, YouTube publishing, local preview.

Because the camera is commanded 8 times a second regardless of how long the AI
takes, patrol and tracking no longer freeze while MegaDetector is busy.
"""

import csv
import json
import logging
import os
import platform
import queue
import threading
import time
from collections import deque
from types import SimpleNamespace
from urllib.parse import quote, urlsplit, urlunsplit

import cv2

import config
from camera.focus import FocusController
from camera.onvif_camera import OnvifPTZCamera
from detection.constants import CLASS_NAMES
from detection.validator import species_label
from stream.rtmp_publisher import RTMPPublisher
from tracking.multi_object_tracker import MultiObjectTracker, camera_shift, shift_box
from tracking.manual_look import ManualLook, OperatorHold
from utils.motion_cue import MotionCue
from utils.trace import NullTrace, PipelineTrace
from tracking.box_flow import BoxFlowTracker
from tracking.ptz_tracker import PTZTracker
from utils.events import WildlifeEventEngine
from tracking.flight import flight_score, is_bird, to_small
from utils.health import write_health
from utils.imageio import write_jpeg
from utils.privacy import in_privacy_zone, privacy_filter
from utils.logger import (log_detection, maybe_save_negative_sample, save_review,
                          setup_logging)
from utils.runtime_metrics import RuntimeMetrics

logger = logging.getLogger("wildlife_tracker.main")

# classifier labels that name nothing in particular (the site model's bare "bird" included)
GENERIC_LABELS = {"", "unidentified", "blank", "other", "animal", "bird", "no cv result"}


def worth_a_closer_look(d, night):
    """Is this unconfirmed box worth stopping the sweep for and zooming in on?

    Hand-checked 21/10 on the 64 candidates saved since 19/10 (logs/review):
    * a box SpeciesNet NAMES (african buffalo, cervidae family, ...) was a real animal
      13 of 15 times in daylight (the other two: acacias called "bovidae family" and
      "helmeted guineafowl");
    * during the sweep over the plain a GENERIC box ("unidentified", bare "bird") was an
      animal one could follow 1 of 16 times (12 acacias / bushes, 3 birds flying past);
    * at night (IR) 0 of 11 sweep candidates: tree trunks at the frame edge, dark canopy,
      insects and birds flying past the lens.
    """
    if night or d.get("track_confirmed") or float(d.get("det_confidence", 0.0)) < config.PATROL_PAUSE_MIN_SCORE:
        return False
    return (d.get("species") or "").strip().lower().rstrip("?").strip() not in GENERIC_LABELS


def add_rtsp_credentials(uri, username, password):
    parts = urlsplit(uri)
    if "@" in parts.netloc or not password:
        return uri
    netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{parts.netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


# --------------------------------------------------------------------------- threads
class FrameGrabber(threading.Thread):
    """Continuously drains the RTSP stream so ``latest()`` is never stale."""

    def __init__(self, uri, stop, metrics):
        super().__init__(name="grabber", daemon=True)
        self.uri, self.stop_evt, self.metrics = uri, stop, metrics
        self._lock = threading.Lock()
        self._frame, self._t, self._seq = None, 0.0, 0
        self.ok = False
        # small grey copies of the last seconds: is a bird box a bird in flight? (26/10)
        self._ring = deque(maxlen=max(1, int(config.FLIGHT_RING_SECONDS * config.FLIGHT_RING_FPS)))
        self._ring_t = 0.0

    def _open(self):
        cap = cv2.VideoCapture(self.uri)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def run(self):
        cap = self._open()
        drops = 0
        while not self.stop_evt.is_set():
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if not ok:
                drops += 1
                self.ok = False
                if drops >= config.MAX_CONSECUTIVE_DROPPED_FRAMES or not cap.isOpened():
                    logger.warning("Stream looks disconnected - reconnecting")
                    cap.release()
                    self.metrics.note_reconnect()
                    self.stop_evt.wait(config.STREAM_RECONNECT_DELAY_SECONDS)
                    cap = self._open()
                    drops = 0
                else:
                    time.sleep(0.03)
                continue
            drops = 0
            self.ok = True
            now = time.monotonic()
            with self._lock:
                self._frame, self._t, self._seq = frame, now, self._seq + 1
            if config.FLIGHT_BIRDS_ENABLED and now - self._ring_t >= 1.0 / config.FLIGHT_RING_FPS:
                small = to_small(frame)
                with self._lock:
                    self._ring.append((now, small))
                self._ring_t = now
        cap.release()

    def latest(self):
        with self._lock:
            return self._frame, self._t, self._seq

    def around(self, t, lo, hi):
        """[(dt, small grey frame)] grabbed between lo and hi seconds before or after t."""
        with self._lock:
            return [(ft - t, img) for ft, img in self._ring if lo <= abs(ft - t) <= hi]

    def frame(self):
        return self.latest()[0]


class DetectionWorker(threading.Thread):
    """Runs the AI on one job at a time; a newer job replaces an unstarted older one."""

    def __init__(self, pipeline, stop):
        super().__init__(name="detector", daemon=True)
        self.pipeline, self.stop_evt = pipeline, stop
        self.tracks = MultiObjectTracker()
        self._jobs = queue.Queue(maxsize=1)
        self._results = queue.Queue(maxsize=2)
        self._reset = threading.Event()
        self.busy = False
        self.last_seconds = 0.0
        self.last_stages = None

    def submit(self, job):
        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            return False

    def request_reset(self):
        self._reset.set()

    def take_result(self):
        res = None
        while True:           # keep only the newest result
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                return res

    def run(self):
        while not self.stop_evt.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            self.busy = True
            try:
                if self._reset.is_set():
                    self._reset.clear()
                    self.tracks.reset()
                    # new view: cached species labels belong to the old pixels
                    self.pipeline.invalidate_classification_cache()
                frame = job["frame"]
                h, w = frame.shape[:2]
                dets = self.pipeline.process_frame(
                    frame, scan_mode=job["scan_mode"], pose=job["pose"], roi=job["roi"],
                    deep=job["deep"], settled=True)
                metrics = dict(self.pipeline.last_metrics)
                if config.NIGHT_DROP_BIRDS and (metrics.get("frame_quality") or {}).get("low_light"):
                    # 24-26/10: every "bird" in a night frame was a stump or a bush (2 of 2 checked,
                    # one of them "confirmed"), like the recurring 22/10 night false positive
                    kept = [d for d in dets if not is_bird(d.get("species"))]
                    if len(kept) != len(dets):
                        metrics["night_birds_dropped"] = len(dets) - len(kept)
                    dets = kept
                tracks = self.tracks.update(dets, w, h, pose=job["pose"])
                self.last_seconds = metrics.get("inference_seconds", 0.0)
                self.last_stages = metrics.get("stages")
                res = dict(job, tracks=tracks, metrics=metrics, done_at=time.monotonic())
                res.pop("frame_ref", None)
                try:
                    self._results.put_nowait(res)
                except queue.Full:
                    try:
                        self._results.get_nowait()
                    except queue.Empty:
                        pass
                    self._results.put_nowait(res)
            except Exception:
                logger.exception("Detection job failed")
            finally:
                self.busy = False


# ------------------------------------------------------------------------ overlay
def draw_overlay(frame, tracks, tracker, status_text, metrics_snapshot=None, health=None,
                 show_candidates=False):
    for d in tracks:
        confirmed = d.get("track_confirmed", True)
        if not confirmed and not show_candidates:
            continue
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        label = species_label(d) if d.get("species") else CLASS_NAMES.get(d.get("class_id"), "?")
        is_target = tracker.locked_track_id is not None and d.get("track_id") == tracker.locked_track_id
        if not confirmed:
            color = (190, 190, 190)
        elif is_target:
            color = (0, 0, 255)
        elif label != "unidentified":
            color = (0, 200, 0)
        else:
            color = (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_target else 2)
        tid = f" T#{d['track_id']}" if d.get("track_id") is not None else ""
        water = " [water]" if d.get("in_water_zone") else ""
        text = f"{'TRACKING: ' if is_target else ''}{label}{tid}{water} {d.get('det_confidence', 0):.2f}"
        if not confirmed:
            text = f"checking... {label} {d.get('det_confidence', 0):.2f}"
        cv2.putText(frame, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    h, w = frame.shape[:2]
    n = sum(1 for d in tracks if d.get("track_confirmed", True))
    count = f"  |  {n} ANIMAL{'S' if n != 1 else ''} IN VIEW" if n else ""
    cv2.rectangle(frame, (0, 0), (w, 42), (18, 26, 22), -1)
    cv2.putText(frame, f"MELIA SERENGETI WILDLIFE LIVE  |  {status_text}{count}", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (238, 245, 240), 2)
    if metrics_snapshot:
        line = (f"AI {metrics_snapshot.get('inference_ms', 0) / 1000:.1f}s/frame | "
                f"Tracks {metrics_snapshot.get('target_locks', 0)}")
        cv2.putText(frame, line, (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 230, 224), 1)
    if health:
        cam = "CAM OK" if health.get("camera_status_ok") else "CAM CHECK"
        stream = "STREAM OK" if health.get("stream_ok") else "STREAM CHECK"
        cv2.putText(frame, f"{cam} | {stream}", (w - 210, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 230, 224), 1)
    return frame


_last_missed = [0.0]


def save_missed_example(frame, pose):
    """M key: keep a frame with an animal the AI missed (second training round)."""
    if time.monotonic() - _last_missed[0] < 2.0:      # a held key repeats - one frame is enough
        return
    _last_missed[0] = time.monotonic()
    folder = os.path.join("training_data", "missed")
    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    write_jpeg(os.path.join(folder, f"{stamp}_missed.jpg"), frame)
    with open(os.path.join(folder, f"{stamp}_missed.json"), "w", encoding="utf-8") as f:
        json.dump({"pose": list(pose) if pose else None, "saved": stamp}, f)
    logger.info("Saved missed-animal example %s (training_data/missed)", stamp)


# ---------------------------------------------------------------------- runtime
class Runtime:
    def __init__(self):
        self.stop = threading.Event()
        self.metrics = RuntimeMetrics()
        self.lock = threading.Lock()
        self.annotation = dict(tracks=[], pose=None, status="STARTING", health=None, stamp=0.0)
        self.events_lock = threading.Lock()
        self.last_pose = None
        self.click = None       # (x, y) from the preview window (click-to-zoom)
        self.goto_water = False # W key in the preview: patrol straight to the water-hole
        self.motion = MotionCue()
        self.flow = BoxFlowTracker()     # follows the locked box between AI results
        self.trace = NullTrace()         # PipelineTrace in production (main())
        self.state = "PATROL"            # authoritative tracking state (see _update_state)
        self.manual = None      # active ManualLook

    # -- control thread -------------------------------------------------
    def control_loop(self, camera, grabber, worker, tracker, focus, events):
        """8 Hz PTZ control.  Errors in one tick are logged and survived."""
        period = 1.0 / max(1.0, config.CONTROL_LOOP_HZ)
        patrol = tracker.observation_patrol
        S = SimpleNamespace(
            last_seq=-1, last_motion=time.monotonic(), last_epoch=-1, tracks=[], result_pose=None,
            result_t=0.0, full_counter=0, last_metrics=0.0, last_health=0.0, last_trace=0.0,
            prev_done=time.monotonic(), health=write_health(camera, True), allowed=False, epoch=-1,
            soft_since=0.0,
        )
        trace_path = config.PTZ_TRACE_CSV
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        new_trace = not os.path.exists(trace_path)
        trace_f = open(trace_path, "a", newline="", encoding="utf-8")
        trace = csv.writer(trace_f)
        if new_trace:
            trace.writerow(["time", "mode", "phase", "pan", "tilt", "zoom", "command",
                            "detect_allowed", "epoch", "locked", "ai_busy", "tracks", "absolute_ok"])
        errors = 0
        try:
            pose0 = camera.get_pose()
            if (config.OPERATOR_DETECT and patrol is not None and pose0 is not None
                    and not patrol.near_station(pose0)):
                # started while aimed somewhere by hand (e.g. at resting animals): watch it first
                # (a restart mid-travel also looks like this: keep it short unless
                # the camera is then moved by hand, which extends it to the full hold)
                zoomed_view = float(pose0[2]) > config.ZOOMED_VIEW_MIN_ZOOM
                self.manual = OperatorHold(pose0, resume_seconds=(config.OPERATOR_RESUME_SECONDS if zoomed_view
                                                                  else config.OPERATOR_STARTUP_HOLD_SECONDS),
                                           reason="started away from a patrol station")
            t_last = None
            while not self.stop.is_set():
                t0 = time.monotonic()
                if (t_last is not None and t0 - t_last > config.SLOW_TICK_SECONDS
                        and getattr(S, "last_tick_seconds", 0.0) < config.SLOW_TICK_SECONDS
                        and t0 - getattr(S, "gap_logged", -1e9) > 30.0):
                    # the tick itself was quick: this thread was not scheduled (CPU / GIL)
                    logger.warning("Control loop: %.1f s between ticks although the last tick took %.2f s",
                                   t0 - t_last, getattr(S, "last_tick_seconds", 0.0))
                    S.gap_logged = t0
                t_last = t0
                try:
                    self._tick(camera, grabber, worker, tracker, patrol, focus, events, S, trace, trace_f, t0)
                    errors = 0
                except Exception:
                    errors += 1
                    logger.exception("Control tick failed (%d in a row)", errors)
                    try:
                        camera.stop()
                    except Exception:
                        pass
                    if errors >= 25:
                        logger.error("Too many consecutive control errors; stopping")
                        self.stop.set()
                spare = period - (time.monotonic() - t0)
                if spare > 0:
                    self.stop.wait(spare)
        finally:
            trace_f.close()

    def _tick(self, camera, grabber, worker, tracker, patrol, focus, events, S, trace, trace_f, t0):
        S.prof = [("start", time.perf_counter())]
        pose = camera.get_pose()
        self._mark(S, "camera pose")
        self.last_pose = pose or self.last_pose
        frame, frame_t, seq = grabber.latest()
        fh, fw = (frame.shape[0], frame.shape[1]) if frame is not None else (1080, 1920)
        tracker.pose_hint = pose

        # ---- operator: W key = water-hole now ----------------------------
        if self.goto_water:
            self.goto_water = False
            if patrol is not None and patrol.goto_water_now():
                tracker.release_for_operator()
                self.manual = None
                logger.info("Operator: W key - going to the water-hole now")

        # ---- operator click-to-zoom -------------------------------------
        click, self.click = self.click, None
        last = getattr(S, "last_click", None)
        if (click is not None and last is not None and t0 - last[0] < config.CLICK_REPEAT_SECONDS
                and abs(click[0] - last[1][0]) < 120 and abs(click[1] - last[1][1]) < 120):
            click = None      # 20/10: ~20 clicks in a minute restarted the centring before it could finish
        if click is not None and frame is not None and pose is not None:
            S.last_click = (t0, click)
            tracker.release_for_operator()
            # the operator SEES an animal there: watch it (and follow whatever the AI finds
            # near the centre) for OPERATOR_RESUME_SECONDS, even if the AI cannot confirm it
            self.manual = ManualLook(camera, click, frame, pose, focus=focus,
                                     hold_seconds=config.OPERATOR_RESUME_SECONDS)
            self.manual.operator = True
        self._watch_operator(camera, tracker, patrol, pose, t0, S)
        manual = self.manual
        tracker.manual_active = manual is not None
        if manual is not None:
            if manual.tick(frame, frame_t, pose):
                S.last_motion = t0
            if manual.done:
                self.manual = manual = None
                tracker.manual_active = False
                if patrol is not None:
                    patrol.reset()
        self._mark(S, "operator / look")

        self._watch_motion(tracker, patrol, frame, frame_t, pose, t0, S)
        self._mark(S, "motion cue")

        # ---- feed the AI only with sharp, settled frames ----------------
        locked = tracker.locked
        if manual is not None and not locked:
            S.allowed, gate_time, S.epoch = manual.detection_allowed, manual.gate_time, -2
        elif not locked and tracker.holding(t0):
            S.allowed, gate_time, S.epoch = True, S.last_motion + config.SETTLE_AFTER_PAN_SECONDS * 0.6, -1
        elif locked:
            if tracker.view_changed:
                S.last_motion = t0
            gate_time = S.last_motion + config.SETTLE_AFTER_PAN_SECONDS * 0.6
            S.allowed, S.epoch = True, -1
        elif patrol is not None:
            S.allowed, gate_time, S.epoch = patrol.detection_allowed, patrol.since, patrol.observe_epoch
        else:
            S.allowed, gate_time, S.epoch = True, 0.0, -1
        if not config.DETECT_ONLY_WHEN_STILL:
            S.allowed = True
        if S.epoch != S.last_epoch:
            if S.epoch != -1:          # patrol arrived at a new view: old pixel boxes are meaningless
                worker.request_reset()
                S.tracks = []
            # (patrol -> lock keeps the tracks: the confirmed animal we just locked onto must survive)
            S.last_epoch = S.epoch
        if in_privacy_zone(pose):
            S.allowed = False         # never analyse (or save review frames of) guests
            # ...and never STAY there (hand control, restart, anything): raise the camera
            if t0 - getattr(S, "privacy_cmd", -1e9) >= 0.5:
                if getattr(S, "privacy_cmd", -1e9) < t0 - 5.0:
                    logger.warning("Camera looks at the lodge pool area (pose %s) - raising it (privacy)",
                                   tuple(round(float(v), 3) for v in pose))
                camera.velocity_move(0.0, config.PATROL_TILT_FAST_SPEED, 0.0, timeout=1)
                S.privacy_cmd = t0
        ready = (S.allowed and frame is not None and seq != S.last_seq and pose is not None
                 and frame_t >= gate_time + 0.5 and not worker.busy)
        if ready and focus is not None and focus.enabled and not focus.frame_is_sharp(frame, pose):
            # Soft frame (zoom/focus still settling, lens hunting): do not let the
            # AI guess at blur - but never block it for more than a while.
            S.soft_since = S.soft_since or t0
            if t0 - S.soft_since < config.SOFT_FRAME_MAX_BLOCK_SECONDS:
                ready = False
                S.last_seq = seq
        if ready:
            S.soft_since = 0.0
            roi = None
            if locked and tracker.locked_box is not None:
                S.full_counter += 1
                # The locked box is in the pixels of the LAST result: once the camera
                # has turned since then, a region around it looks at the wrong place
                # (19/10: a buffalo was lost 16 s after the lock) - look at everything.
                stale = (S.result_pose is None
                         or abs(pose[0] - S.result_pose[0]) > config.TRACK_ROI_MAX_PAN_SHIFT
                         or abs(pose[2] - S.result_pose[2]) > config.TRACK_ROI_MAX_ZOOM_SHIFT)
                if not stale and S.full_counter % max(1, config.TRACK_FULLFRAME_EVERY_N) != 0:
                    roi = self._roi_for(tracker.locked_box, frame.shape)
            elif manual is None or isinstance(manual, OperatorHold):
                # Something moved on the STILL view: look at that spot at full resolution.
                # 21/10, the 7 hand-checked small primates the AI missed: MegaDetector on the
                # whole frame scored none of them >= 0.35, on a 640-px crop around them 3.
                cue = getattr(S, "still_cue", None)
                same_view = (cue is not None and cue[2] is not None and pose is not None
                             and abs(pose[0] - cue[2][0]) < 0.003 and abs(pose[1] - cue[2][1]) < 0.01
                             and abs(pose[2] - cue[2][2]) < 0.005)
                if (same_view and t0 - cue[0] <= config.MOTION_ROI_MAX_AGE
                        and t0 - getattr(S, "motion_roi_sent", -1e9) >= config.MOTION_ROI_MIN_GAP):
                    roi = self._motion_roi(cue[1], frame.shape)
                    S.motion_roi_sent = t0
                    logger.info("Movement at (%.0f, %.0f): the next AI look is that spot at full resolution",
                                cue[1][0], cue[1][1])
            job = dict(frame=frame, pose=pose, epoch=S.epoch, scan_mode=not locked, roi=roi,
                       deep=not locked, seq=seq, captured=frame_t)
            if worker.submit(job):
                S.last_seq = seq
        self._mark(S, "AI gate / submit")

        # ---- collect a new AI result, if any -------------------------------
        res = worker.take_result()
        fresh = res is not None
        if fresh:
            S.tracks = res["tracks"]
            m = self.manual
            if isinstance(m, ManualLook) and m.phase == "hold" and (m.approach or m.auto or m.operator):
                # we zoomed in on an animal that was ALREADY confirmed: whatever is near
                # the centre now is that animal - follow it, no second confirmation needed
                fh_, fw_ = res["frame"].shape[:2]
                need = (config.OPERATOR_RELOCK_MIN_SCORE if m.operator else
                        config.APPROACH_RELOCK_MIN_SCORE if m.approach else config.INVESTIGATE_RELOCK_MIN_SCORE)
                near = [d for d in S.tracks if d.get("det_confidence", 0.0) >= need
                        and abs((d["box"][0] + d["box"][2]) / 2 - fw_ / 2) <= config.APPROACH_RELOCK_RADIUS * fw_
                        and abs((d["box"][1] + d["box"][3]) / 2 - fh_ / 2) <= config.APPROACH_RELOCK_RADIUS * fh_]
                if near:
                    best = max(near, key=lambda d: d.get("det_confidence", 0.0))
                    best["track_confirmed"] = True
                    best["operator_target"] = True     # lock on it even below the normal score
                    logger.info("Approach: the animal is in the centre (%.2f) - following it", best["det_confidence"])
            S.result_pose, S.result_t = res["pose"], res["done_at"]
            confirmed = [d for d in S.tracks if d.get("track_confirmed")]
            analysed = res["frame"]
            if patrol is not None:
                patrol.note_result(res["epoch"])
                patrol.candidates_pending = sum(
                    1 for d in S.tracks if not d.get("track_confirmed") and d.get("det_confidence", 0) >= 0.20)
                patrol.candidate_best = max(
                    [d.get("det_confidence", 0.0) for d in S.tracks if not d.get("track_confirmed")], default=0.0)
                night = bool((res["metrics"].get("frame_quality") or {}).get("low_light"))
                patrol.travel_candidate_best = max(
                    [d.get("det_confidence", 0.0) for d in S.tracks if worth_a_closer_look(d, night)], default=0.0)
            announced = S.__dict__.setdefault("announced", set())
            for d in confirmed:
                log_detection(d, pose=res["pose"])
                if d.get("track_id") not in announced:
                    announced.add(d.get("track_id"))
                    logger.info("ANIMAL CONFIRMED: %s (species %.2f, box %.2f) track %s at pan %.4f zoom %.3f",
                                species_label(d), float(d.get("species_confidence") or 0.0),
                                float(d.get("det_confidence") or 0.0), d.get("track_id"),
                                res["pose"][0], res["pose"][2])
                if d.get("species") in ("unidentified", "other") or not d.get("species_trusted", True):
                    save_review(analysed, [d], "unidentified", res["pose"], key=("u", d.get("track_id")))
                elif d.get("in_water_zone"):
                    save_review(analysed, [d], "water", res["pose"], key=("w", d.get("track_id")))
                else:      # every confirmed animal is a training example for the next round
                    save_review(analysed, [d], "animal", res["pose"], key=("a", d.get("track_id")))
            self._birds_in_flight(res, grabber, S, events)
            weak = [d for d in S.tracks if not d.get("track_confirmed") and d.get("det_confidence", 0) >= 0.25]
            if weak:
                save_review(analysed, weak, "candidate", res["pose"], key=("c", tuple(sorted(d["track_id"] for d in weak))))
            if not S.tracks:
                maybe_save_negative_sample(analysed, res["pose"])
            if confirmed and patrol is not None and tracker.searching and pose is not None:
                strongest = max(confirmed, key=lambda d: d.get("det_confidence", 0.0))
                patrol.record_sighting(pose[0], strongest.get("det_confidence", 0.0))
            self.metrics.record_frame(max(1e-3, res["done_at"] - S.prev_done),
                                      res["metrics"].get("inference_seconds", 0.0), confirmed)
            S.prev_done = res["done_at"]
            if events:   # also with an empty list: that is what ends tracks and closes clips
                with self.events_lock:
                    events.update(confirmed, tracker.locked_track_id, analysed)
        if events:
            with self.events_lock:
                events.recorder.maybe_finish()
        self._mark(S, "AI result / events")

        # ---- steer the camera -------------------------------------------------
        if fresh:
            self._maybe_investigate(camera, tracker, patrol, focus, frame, pose, t0, S, res)
        tracker.trace = self.trace
        # camera ownership as of NOW, before the tracker acts: a recovery hold that has
        # just expired hands the camera to the patrol in this tick (not one tick late)
        self._update_state(tracker, patrol, t0, S)
        flow_alive = (config.FLOW_ENABLED and tracker.locked and self.flow.ok
                      and t0 - getattr(S, "flow_ai_seen", -1e9) < config.FLOW_KEEPALIVE_SECONDS)
        tracker.update(S.tracks if fresh else [], fw, fh, fresh=fresh, now=t0,
                       conclusive=(bool(res["metrics"].get("conclusive", True)) and not flow_alive) if fresh else True,
                       capture_pose=res["pose"] if fresh else None,
                       capture_time=res.get("captured") if fresh else None)
        if (fresh and tracker.locked and self.manual is None and config.APPROACH_SMALL_TARGETS
                and pose is not None and pose[2] < config.AUTO_INVESTIGATE_MAX_ZOOM and frame is not None
                and not getattr(tracker, "_boundary_block_count", 0)):
            # (not at the end of the arc: there the camera cannot centre it and widens the view)
            # 19/10 17:01: a buffalo 15x25 px far away was confirmed but the tracker's
            # small zoom pulses would take minutes - go close in one smooth move
            x1, y1, x2, y2 = tracker.locked_box
            size = max(x2 - x1, y2 - y1)
            done = S.__dict__.setdefault("approached", set())
            if (size < config.APPROACH_MAX_PX and tracker.locked_track_id not in done
                    and pose[2] < config.APPROACH_ONLY_BELOW_ZOOM):
                done.add(tracker.locked_track_id)
                factor = max(1.5, min(config.APPROACH_MAX_FACTOR, config.APPROACH_TARGET_PX / max(1.0, size)))
                point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                g = tracker.group_box if tracker.group_count >= 2 else None
                if g is not None:
                    # a group (user 21/10: "all of them in the centre, not one"): centre the
                    # whole group and zoom only as far as all of them still fit
                    gw, gh = max(1.0, g[2] - g[0]), max(1.0, g[3] - g[1])
                    factor = max(1.0, min(factor, config.TRACK_GROUP_FILL_TARGET * min(fw / gw, fh / gh)))
                    point = ((g[0] + g[2]) / 2.0, (g[1] + g[3]) / 2.0)
                logger.info("Approaching %s (%.0f px): centre and zoom x%.1f",
                            f"a group of {tracker.group_count}" if g is not None else "a small animal", size, factor)
                tracker.release_for_operator()
                # the locked box belongs to the ANALYSED frame (like _maybe_investigate).  26/10
                # 13:09: a baboon confirmed from a frame at pan 0.329 while the patrol travelled
                # on; the approach took the LIVE picture (pan 0.345) as the reference, turned
                # 0.03 past the baboon and zoomed it out of the 11.6x view
                ref = res["frame"] if res.get("frame") is not None else frame
                if patrol is not None and patrol.moving:
                    camera.stop()            # the patrol was still travelling: stop it here
                self.manual = ManualLook(camera, point, ref, pose, focus=focus,
                                         hold_seconds=config.AUTO_INVESTIGATE_HOLD_SECONDS, zoom_factor=factor)
                self.manual.approach = True
                tracker.manual_active = True
        if isinstance(self.manual, ManualLook) and tracker.locked:
            logger.info("Operator look: the AI confirmed the animal - tracker follows it")
            if self.manual.auto and getattr(S, "investigated", None):
                # a real animal there: the 10-min "not the same spot again" rule is for trees
                # that fooled the AI - this spot may be zoomed in on again when it reappears
                S.investigated = S.investigated[:-1]
            self.manual = None
            tracker.manual_active = False
        self._mark(S, "tracker / patrol commands")
        if config.FLOW_ENABLED:
            self._visual_follow(tracker, frame, frame_t, seq, res if fresh else None, fw, fh, t0, S)
        self._mark(S, "visual follow")
        self._update_state(tracker, patrol, t0, S)
        if fresh:
            self._trace_result(res, tracker, S)
        # The patrol's phase says "moving" only while it really drives the camera.  During a
        # recovery hold / a look it is parked in "bootstrap" and must not hold back the AI:
        # 20/10 ptz_trace.csv - ai_busy=0 for the whole 60-s hold after every lost target
        # (4 of 4 holds), so an animal that stepped out from behind a bush was never seen.
        if (not locked and patrol is not None and patrol.moving and not patrol.suspended_by) or tracker.view_changed:
            S.last_motion = t0
        self.metrics.note_tracker(tracker.locked, tracker.searching, tracker.last_boundary_event)

        # ---- annotation for the overlay thread --------------------------------
        status = ("TRACKING" if tracker.locked else self.manual.status if self.manual is not None else
                  (f"PATROL {patrol.status_text()}" if patrol is not None else "SCANNING"))
        # Boxes are in the pixels of the analysed frame; the camera may have turned
        # since (sweep, tracking).  Move them with the camera instead of hiding them.
        shown = []
        if (S.result_pose is not None and pose is not None
                and t0 - S.result_t <= config.DETECTION_MAX_AGE_SECONDS
                and abs(pose[1] - S.result_pose[1]) <= 0.03
                and abs(pose[0] - S.result_pose[0]) <= config.OVERLAY_MAX_PAN_SHIFT):
            shift = camera_shift(S.result_pose, pose, fw)
            if shift is not None:
                dx, k, dy = shift
                shown = [dict(d, box=shift_box(d["box"], dx, k, fw, fh, dy)) for d in S.tracks]
        with self.lock:
            self.annotation = dict(tracks=shown, pose=pose,
                                   status=status, health=S.health, stamp=t0)

        now = time.monotonic()
        abs_ok = getattr(patrol.motion, "absolute_ok", None) if patrol else None
        if now - S.last_health >= config.CAMERA_STATUS_INTERVAL_SECONDS:
            snap = self.metrics.snapshot()
            snap["ai_seconds_per_frame"] = round(worker.last_seconds, 2)
            snap["ai_stages"] = getattr(worker, "last_stages", None)
            snap["animals_in_view"] = sum(1 for d in S.tracks if d.get("track_confirmed"))
            snap["patrol_phase"] = patrol.status_text() if patrol else ""
            snap["ptz_move_mode"] = "absolute" if abs_ok else ("continuous" if abs_ok is False else "probing")
            snap["focus"] = focus.last_result if focus else ""
            S.health = write_health(camera, grabber.ok, snap)
            S.last_health = now
        if now - S.last_metrics >= 5.0:
            self.metrics.write()
            S.last_metrics = now
        if now - S.last_trace >= config.PTZ_TRACE_EVERY_SECONDS:
            trace.writerow([
                f"{time.time():.2f}", "track" if tracker.locked else "patrol",
                patrol.phase if patrol else "",
                *(["%.4f" % v for v in pose] if pose else ["", "", ""]),
                patrol.motion.last_command if patrol else "",
                int(bool(S.allowed)), S.epoch, int(tracker.locked), int(worker.busy),
                len(S.tracks), "" if abs_ok is None else int(abs_ok),
            ])
            trace_f.flush()
            S.last_trace = now
        self._mark(S, "overlay / health / logs")
        self._report_slow_tick(S, t0)

    @staticmethod
    def _mark(S, name):
        S.__dict__.setdefault("prof", [("start", time.perf_counter())]).append((name, time.perf_counter()))

    @staticmethod
    def _report_slow_tick(S, t0):
        """21/10: ticks of 2-8 s (gaps in ptz_trace.csv) made our own moves look like a hand on
        the camera.  Name the part of the tick that took the time (at most every 30 s)."""
        prof = S.prof
        total = prof[-1][1] - prof[0][1]
        S.last_tick_seconds = total
        if total < config.SLOW_TICK_SECONDS or t0 - getattr(S, "slow_tick_logged", -1e9) < 30.0:
            return
        parts = [f"{name} {b - a:.1f}s" for (_, a), (name, b) in zip(prof, prof[1:]) if b - a >= 0.2]
        logger.warning("Slow control tick: %.1f s (%s)", total, ", ".join(parts) or "spread over all parts")
        S.slow_tick_logged = t0

    # ------------------------------------------------------------ visual follow
    def _visual_follow(self, tracker, frame, frame_t, seq, res, fw, fh, t0, S):
        """Follow the locked animal on every new frame between AI results (tracking/box_flow.py)."""
        hist = S.__dict__.setdefault("flow_hist", [])
        if not tracker.locked or frame is None:
            if self.flow.box is not None:
                self.flow.reset()
                hist.clear()
            tracker.visual_follow = False
            return
        sighting = res is not None and tracker.target_in_result
        within = t0 - getattr(S, "flow_ai_seen", -1e9) < config.FLOW_KEEPALIVE_SECONDS
        if res is not None and tracker.locked_box is not None and (sighting or (within and not self.flow.ok)):
            # Re-anchor on a fresh AI result that CONTAINS the target (the AI corrects the box).
            # The flow knows where the box was at the result's capture time - correct the drift
            # instead of re-propagating.  A flow that lost its points (animal behind a bush) is
            # restarted at the predicted position, but that is NOT a sighting: 21/10 06:59-07:41
            # re-anchoring on every result as if seen kept a vanished buffalo locked for 42 min.
            ai = tracker.locked_box
            hw, hh = (ai[2] - ai[0]) / 2, (ai[3] - ai[1]) / 2
            t_cap = res.get("captured")
            near = min(hist, key=lambda h: abs(h[0] - t_cap)) if (hist and t_cap is not None) else None
            agree = False
            if sighting and self.flow.ok and near is not None and abs(near[0] - t_cap) <= 0.5:
                hb = near[1]
                d = (((ai[0] + ai[2]) - (hb[0] + hb[2])) ** 2 + ((ai[1] + ai[3]) - (hb[1] + hb[3])) ** 2) ** 0.5 / 2
                agree = d <= max(2 * hw, 2 * hh)
            if agree:
                # the AI confirms what the flow followed: keep the (more recent) flow box,
                # re-seed its points and take the AI's size
                cur = self.flow.box
                cx, cy = (cur[0] + cur[2]) / 2, (cur[1] + cur[3]) / 2
            else:
                # flow lost or wrong: the AI box, carried into the current picture by the CAMERA's
                # own motion since that picture was taken (plus a walking speed only once it is
                # trusted).  22/10 17:50: a world prediction with a speed made up from lock hops
                # put the box on a bush - the camera then followed a branch in the wind.
                cx, cy = (ai[0] + ai[2]) / 2, (ai[1] + ai[3]) / 2
                box_pose = tracker.locked_pose          # where the camera was when that box was seen
                sh = camera_shift(box_pose, tracker.pose_hint, fw) if box_pose is not None \
                    and tracker.pose_hint is not None else None
                if sh is not None:
                    dx, k, dy = sh
                    sb = shift_box(ai, dx, k, fw, fh, dy)
                    cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
                    hw, hh = (sb[2] - sb[0]) / 2, (sb[3] - sb[1]) / 2
                px = tracker.predicted_center_x(fw, t0)
                if px is not None and tracker._trusted_world_vel() != 0.0:
                    cx = px
            self.flow.anchor(frame, [cx - hw, cy - hh, cx + hw, cy + hh])
            S.flow_seq = seq
            if sighting:
                S.flow_ai_seen = t0
            hist[:] = [(frame_t, list(self.flow.box))] if self.flow.box else []
            tracker.visual_follow = self.flow.ok
            return
        since_ai = t0 - getattr(S, "flow_ai_seen", -1e9)
        if since_ai >= config.FLOW_KEEPALIVE_SECONDS:
            # the AI has not seen the animal for a while: what the flow follows now is
            # probably background - stop steering and let the loss rules decide
            if self.flow.box is not None:
                logger.info("Visual tracker: no AI sighting for %.0f s - it stops steering the camera", since_ai)
                self.flow.reset()
                hist.clear()
            tracker.visual_follow = False
            return
        if seq == getattr(S, "flow_seq", None) or not self.flow.ok:
            tracker.visual_follow = self.flow.ok
            return
        S.flow_seq = seq
        box = self.flow.update(frame)
        if box is None:
            logger.info("Visual tracker lost the target (%s) - waiting for the AI", self.flow.fail_reason)
            tracker.visual_follow = False
            return
        hist.append((frame_t, list(box)))
        del hist[:-60]
        tracker.visual_follow = True
        tracker.follow_box(box, fw, fh, t0, keepalive=True)

    # ------------------------------------------------------------ state machine
    def _update_state(self, tracker, patrol, t0, S):
        """One authoritative tracking state, derived from who owns the camera.

        PATROL -> CANDIDATE -> ACQUIRING -> TRACKING -> TARGET_LOST -> RECOVERY -> PATROL
        (+ OPERATOR while a person steers the camera).  Every transition is logged
        with its reason; the patrol is suspended in every state it does not own.
        """
        if tracker.locked:
            state = "TRACKING" if tracker.missed_frames == 0 else "TARGET_LOST"
        elif isinstance(self.manual, ManualLook):
            state = "ACQUIRING"
        elif isinstance(self.manual, OperatorHold):
            state = "OPERATOR"
        elif tracker.holding(t0):
            state = "RECOVERY"
        elif patrol is not None and (patrol.phase == "verify" or patrol.candidates_pending):
            state = "CANDIDATE"
        else:
            state = "PATROL"
        if patrol is not None:
            patrol.suspended_by = None if state in ("PATROL", "CANDIDATE") else state
        if state != self.state:
            reason = self._transition_reason(self.state, state, tracker, patrol)
            logger.info("STATE %s -> %s: %s", self.state, state, reason)
            self.trace.log("state", {"from": self.state, "to": state, "reason": reason,
                                     "selected": list(tracker.last_selection)}, force=True)
            self.state = state

    def _transition_reason(self, old, new, tracker, patrol):
        tid, why = tracker.last_selection
        if new == "TRACKING":
            return (f"AI result contains the target again (track {tid})" if old == "TARGET_LOST"
                    else f"target locked: track {tid} - {why}")
        if new == "TARGET_LOST":
            return f"{tracker.missed_frames} AI result(s) without the target; visual tracker {'on' if tracker.visual_follow else 'off'}"
        if new == "RECOVERY":
            return f"target lost - holding the view for {config.LOST_HOLD_SECONDS:.0f} s before patrol"
        if new == "ACQUIRING":
            m = self.manual
            kind = "approach" if m.approach else "auto-investigate" if m.auto else "operator click" if m.operator else "look"
            return f"{kind}: centring and zooming on the candidate"
        if new == "OPERATOR":
            return f"{getattr(self.manual, 'reason', 'camera moved by hand')} - watching that view"
        if new == "CANDIDATE":
            return f"possible animal (best score {getattr(patrol, 'candidate_best', 0.0):.2f}) - patrol checks it"
        if new == "PATROL":
            return {"RECOVERY": "recovery hold expired without the target",
                    "CANDIDATE": "candidate rejected / faded",
                    "ACQUIRING": "look finished without a confirmed animal",
                    "OPERATOR": "no hand movement - patrol resumes"}.get(old, "patrol")
        return ""

    def _trace_result(self, res, tracker, S):
        m = res.get("metrics", {})
        tracks = res.get("tracks") or []
        dets = [{"id": d.get("track_id"), "species": d.get("species"), "conf": round(float(d.get("det_confidence", 0.0)), 3),
                 "box": [round(float(v)) for v in d["box"]], "confirmed": bool(d.get("track_confirmed")),
                 "hits": d.get("track_hits")} for d in tracks]
        fr = res.get("frame")
        rec = {"frame": None if fr is None else [int(fr.shape[1]), int(fr.shape[0])],
               "pose": None if res.get("pose") is None else [round(float(v), 4) for v in res["pose"]],
               "infer_s": round(float(m.get("inference_seconds", 0.0)), 2),
               "latency_s": round(float(res.get("done_at", 0.0) - res.get("captured", res.get("done_at", 0.0))), 2),
               "raw": m.get("base_detections"), "kept": m.get("animal_detections"),
               "rejected": m.get("rejected"), "dropped_non_wildlife": m.get("dropped_non_wildlife"),
               "stages": m.get("stages"), "roi": bool(res.get("roi")), "dets": dets, "state": self.state,
               "selected": list(tracker.last_selection) if tracker.locked else None}
        # every result with something in it; empty ones at most every 10 s
        self.trace.log("ai", rec, min_interval=10.0, force=bool(dets) or bool(m.get("base_detections")))

    def _watch_motion(self, tracker, patrol, frame, frame_t, pose, t0, S):
        """Detector-independent clue: something moving by itself (utils/motion_cue.py)."""
        if not config.MOTION_CUE_ENABLED or patrol is None or pose is None or frame is None:
            return
        phase = patrol.phase
        sweeping = phase == "travel"
        still = phase in ("observe_wide", "verify") or isinstance(self.manual, OperatorHold)
        sweeping = sweeping and config.MOTION_CUE_WHILE_SWEEPING     # 21/10: 0 of 169 sweep stops were animals
        usable = (not tracker.locked and not isinstance(self.manual, ManualLook)
                  and (sweeping or still) and pose[2] <= config.MOTION_CUE_MAX_ZOOM
                  and not in_privacy_zone(pose))
        key = (phase, sweeping)
        if not usable or getattr(S, "motion_key", None) != key:
            self.motion.reset()             # a new view / zoom / tilt: compare from scratch
            S.motion_key = key if usable else None
            if not usable:
                return
        cue = self.motion.update(frame, frame_t, camera_moving=sweeping)
        if cue is None:
            return
        # Learned "motion clutter": a spot that keeps moving visit after visit (a tree
        # crown against the sky, 19/10 16:51) is muted; a real animal moves on.
        mag = 1.0 + config.PTZ_ZOOM_MAG_PER_UNIT * max(0.0, float(pose[2]))
        spot_pan = pose[0] - (cue[0] - frame.shape[1] / 2.0) / (config.PAN_PIXELS_PER_UNIT_AT_1X * mag
                                                               * frame.shape[1] / 1920.0)
        spot = (round(spot_pan, 2), round(float(pose[1]), 2), int(cue[1] // (frame.shape[0] / 8.0)))
        mutes = S.__dict__.setdefault("motion_mute", {})
        m = mutes.setdefault(spot, {"last": -1e9, "episodes": 0, "muted_until": -1e9})
        if t0 < m["muted_until"]:
            return
        if t0 - m["last"] > config.MOTION_MUTE_EPISODE_GAP:
            m["episodes"] += 1
        m["last"] = t0
        if m["episodes"] >= config.MOTION_MUTE_EPISODES:
            m["muted_until"] = t0 + config.MOTION_MUTE_SECONDS
            logger.info("Motion at pan %.2f keeps coming back (wind in a tree?) - ignored for %.0f min",
                        spot[0], config.MOTION_MUTE_SECONDS / 60.0)
            return
        patrol.motion_cue = (t0, cue)
        if still:
            S.still_cue = (t0, cue, pose)      # the next AI look goes to that spot (motion ROI)
        if t0 - getattr(S, "motion_logged", -1e9) > 10.0:
            logger.info("Motion cue at (%.0f, %.0f) %s - %s", cue[0], cue[1], phase,
                        "pausing the sweep" if sweeping else "watching it")
            S.motion_logged = t0
        if still:
            events = [e for e in getattr(S, "still_motion_events", []) if t0 - e < 4.0] + [t0]
            S.still_motion_events = events
            if len(events) >= config.MOTION_CUE_INVESTIGATE_EVENTS:     # it keeps moving: worth a look
                S.still_motion = (t0, cue)

    def _birds_in_flight(self, res, grabber, S, events):
        """A bird box seen once is a bird FLYING past when the picture 0.2-1 s before or after
        shows it moved away (tracking/flight.py).

        24-26/10: 4 of 8 bird detections checked by eye were real birds, all flying past and
        each seen by one AI result - confirmation needs two, so all four were dropped.  Such a
        bird is now logged ("BIRD IN FLIGHT"), saved (logs/review/*_bird_flight.jpg), written to
        detections.csv and events.csv - not followed: it is gone long before the AI result
        (5-13 s after the frame) and no PTZ catches a flying bird."""
        if not config.FLIGHT_BIRDS_ENABLED or (res["metrics"].get("frame_quality") or {}).get("low_light"):
            return
        around = getattr(grabber, "around", None)
        cands = [d for d in S.tracks if not d.get("track_confirmed") and is_bird(d.get("species"))
                 and float(d.get("det_confidence") or 0.0) >= config.FLIGHT_MIN_SCORE]
        if not cands or around is None or res.get("captured") is None:
            return
        neighbours = around(res["captured"], config.FLIGHT_NEIGHBOUR_MIN_S, config.FLIGHT_NEIGHBOUR_MAX_S)
        if not neighbours:
            return
        seen = S.__dict__.setdefault("flight_seen", [])
        small = to_small(res["frame"])
        for d in cands:
            if d.get("track_id") in seen:
                continue
            score, dt = flight_score(d["box"], small, neighbours)
            self.trace.log("flight", {"track": d.get("track_id"), "species": d.get("species"),
                                      "conf": round(float(d.get("det_confidence") or 0.0), 3),
                                      "box": [round(float(v)) for v in d["box"]], "score": round(score, 2),
                                      "dt": None if dt is None else round(dt, 2), "pairs": len(neighbours)},
                           force=True)
            if score < config.FLIGHT_MIN_RATIO:
                continue
            seen.append(d.get("track_id"))
            del seen[:-200]
            bird = dict(d, track_confirmed=True, flight=True)
            log_detection(bird, pose=res["pose"])
            save_review(res["frame"], [bird], "bird_flight", res["pose"], key=("f", d.get("track_id")))
            if events:
                with self.events_lock:
                    events.record_flight(bird)
            logger.info("BIRD IN FLIGHT: %s (box %.2f, %.0f px) at pan %.4f - moved away within %.1f s "
                        "(score %.1f); logged and saved, not followed", species_label(d),
                        float(d.get("det_confidence") or 0.0), d["box"][2] - d["box"][0], res["pose"][0],
                        abs(dt), score)

    def _maybe_investigate(self, camera, tracker, patrol, focus, frame, pose, t0, S, res=None):
        """Zoom in by itself on something the AI keeps half-seeing.

        19/10: a baboon resting in deep shade scored 0.46 ("unidentified") and
        was never confirmed at 3.85x - 45x50 px, dark on dark.  At 2.5x more
        zoom it is big enough for the detectors and the classifier.  Normally only
        when the camera is still (patrol pause/observation or an operator view),
        never twice for the same spot within AUTO_INVESTIGATE_REPEAT_SECONDS.

        21/10: a box the classifier NAMES (worth_a_closer_look) is zoomed in on at
        once, also from the moving sweep: 15-22 px buffalo named "african buffalo"
        were passed because the short sweep pause never saw them again at 3.85x.
        """
        if (not config.AUTO_INVESTIGATE or tracker.locked or frame is None or pose is None
                or isinstance(self.manual, ManualLook) or pose[2] >= config.AUTO_INVESTIGATE_MAX_ZOOM):
            return
        still = isinstance(self.manual, OperatorHold) or (patrol is not None and not patrol.moving)
        night = bool(((res or {}).get("metrics") or {}).get("frame_quality", {}).get("low_light"))
        named = [d for d in S.tracks if worth_a_closer_look(d, night)] if res is not None else []
        if (not still and not named) or t0 - getattr(S, "last_investigate", -1e9) < config.AUTO_INVESTIGATE_COOLDOWN:
            return
        cands = [d for d in S.tracks if not d.get("track_confirmed")
                 and d.get("det_confidence", 0.0) >= config.AUTO_INVESTIGATE_MIN_SCORE
                 and (d.get("track_hits", 1) >= 2 or d.get("det_confidence", 0.0) >= config.AUTO_INVESTIGATE_SINGLE_SCORE)]
        if not still:
            cands = named                 # moving: only what the classifier named
        else:
            cands += [d for d in named if d not in cands]
        motion = getattr(S, "still_motion", None)
        if (not cands and config.MOTION_CUE_INVESTIGATE and motion is not None
                and t0 - motion[0] <= 1.5):
            # nothing the detectors recognise, but it keeps MOVING on a still camera:
            # an animal or a bird - zoom in so the detectors can see it properly
            mcx, mcy, mw, mh = motion[1]
            cands = [{"box": [mcx - mw / 2, mcy - mh / 2, mcx + mw / 2, mcy + mh / 2],
                      "det_confidence": 0.0, "species": "moving", "class_id": 0, "live_box": True}]
            save_review(frame, cands, "motion", pose, key=("m", round(pose[0], 2), int(mcx // 200), int(mcy // 200)))
        if not cands:
            return
        # named boxes first (13 of 15 real), then the strongest
        best = max(cands, key=lambda d: (d in named, d.get("det_confidence", 0.0)))
        x1, y1, x2, y2 = best["box"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # a detector box belongs to the ANALYSED frame, and the camera may have turned since:
        # while sweeping, and also when it stopped after that frame was taken.  23/10 09:28:
        # a bird seen at pan 0.305 on the way to scan_7, the result came 13 s later at pan
        # 0.282, and the look centred the box on the LIVE picture - 460 px beside the bird,
        # outside the 9.6x view.  A motion box is measured on the live picture.
        moving = not still
        if res is not None and res.get("frame") is not None and not best.get("live_box"):
            ref, spot_pan = res["frame"], res["pose"][0]
        else:
            ref, spot_pan = frame, pose[0]
        done = [s for s in getattr(S, "investigated", []) if t0 - s[3] < config.AUTO_INVESTIGATE_REPEAT_SECONDS]
        if any(abs(spot_pan - p) < 0.01 and abs(cx - x) < 150 and abs(cy - y) < 150 for p, x, y, _ in done):
            S.investigated = done
            return
        S.investigated = done + [(spot_pan, cx, cy, t0)]
        S.last_investigate = t0
        factor = None
        if best in named:
            # small and far: go close enough for the detectors (like an approach)
            size = max(x2 - x1, y2 - y1)
            factor = max(config.MANUAL_ZOOM_FACTOR,
                         min(config.APPROACH_MAX_FACTOR, config.APPROACH_TARGET_PX / max(1.0, size)))
        logger.info("Auto-investigate: something at (%.0f, %.0f) scores %.2f (%s) but is not confirmed - zooming in%s",
                    cx, cy, best.get("det_confidence", 0.0), best.get("species") or "?",
                    " from the sweep" if moving else "")
        if patrol is not None:
            patrol.motion.target = None
        if moving:
            camera.stop()                 # leave the sweep where it is
        self.manual = ManualLook(camera, (cx, cy), ref, pose, focus=focus,
                                 hold_seconds=config.AUTO_INVESTIGATE_HOLD_SECONDS, zoom_factor=factor)
        self.manual.auto = True
        tracker.manual_active = True      # the patrol must not move the camera this tick

    def _watch_operator(self, camera, tracker, patrol, pose, t0, S):
        """Camera moving although we sent nothing for a while -> a person steers it."""
        last_cmd = getattr(camera, "last_command_time", None)
        if not config.OPERATOR_DETECT or pose is None or last_cmd is None:
            return
        if tracker.locked or isinstance(self.manual, ManualLook):
            # we are steering (tracking / zooming in): our own moves and their coasting are
            # not a person (19/10 18:07: a buffalo lock was dropped as "moved by hand")
            S.op_anchor, S.op_anchor_t = pose, t0
            return
        anchor = getattr(S, "op_anchor", None)
        quiet_from = last_cmd + config.OPERATOR_QUIET_SECONDS      # our own motion is over by then
        if t0 < quiet_from or anchor is None or getattr(S, "op_anchor_t", -1e9) < quiet_from:
            # Compare only with a pose taken AFTER our own motion ended.  21/10 09:00: the loop
            # stalled 3 s during the patrol's tilt glide, the camera slid on under OUR command,
            # and the next tick compared with the pose from before the stall - "moved by hand",
            # the patrol stood still for 5 min.
            S.op_anchor, S.op_anchor_t = pose, t0
            return
        tilt_moved = (pose[1] > -0.999 or anchor[1] > -0.999) and abs(pose[1] - anchor[1]) > config.OPERATOR_MOVE_TILT
        if not (abs(pose[0] - anchor[0]) > config.OPERATOR_MOVE_PAN or tilt_moved
                or abs(pose[2] - anchor[2]) > config.OPERATOR_MOVE_ZOOM):
            return
        S.op_anchor, S.op_anchor_t = pose, t0
        if isinstance(self.manual, OperatorHold):
            self.manual.note_motion()
            return
        tracker.release_for_operator()
        if patrol is not None:
            patrol.motion.clear_fault()
        self.manual = OperatorHold(pose)

    @staticmethod
    def _motion_roi(cue, shape):
        """MOTION_ROI_SIDE-px square around a motion cue, kept inside the frame (native resolution)."""
        h, w = shape[:2]
        side = int(min(config.MOTION_ROI_SIDE, w, h))
        x1 = int(min(max(0.0, cue[0] - side / 2.0), w - side))
        y1 = int(min(max(0.0, cue[1] - side / 2.0), h - side))
        return (x1, y1, x1 + side, y1 + side)

    @staticmethod
    def _roi_for(box, shape):
        h, w = shape[:2]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        size = max(config.TRACK_ROI_MIN_SIZE, config.TRACK_ROI_SCALE * max(box[2] - box[0], box[3] - box[1]))
        x1, y1 = int(max(0, cx - size / 2)), int(max(0, cy - size / 2))
        x2, y2 = int(min(w, cx + size / 2)), int(min(h, cy + size / 2))
        return (x1, y1, x2, y2)


def main():
    setup_logging()
    if not config.CAMERA_PASSWORD:
        raise RuntimeError("Set WILDLIFE_CAMERA_PASSWORD in .env or the environment before starting the camera.")
    logger.info("Melia Serengeti Wildlife AI v%s | Python %s | ptz_mode=%s | device=%s",
                config.VERSION, platform.python_version(), config.PTZ_MOVE_MODE, config.DEVICE)

    rt = Runtime()
    rt.trace = PipelineTrace()
    logger.info("Connecting to camera at %s ...", config.CAMERA_IP)
    camera = OnvifPTZCamera(config.CAMERA_IP, config.CAMERA_ONVIF_PORT,
                            config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    publisher = RTMPPublisher()
    events = WildlifeEventEngine() if config.EVENTS_ENABLED else None
    grabber = worker = control = None
    try:
        stream_uri = camera.get_stream_uri()
        logger.info("RTSP stream acquired")
        auth_uri = add_rtsp_credentials(stream_uri, config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
        if config.RTSP_TRANSPORT:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                                  f"rtsp_transport;{config.RTSP_TRANSPORT}|buffer_size;102400")
        grabber = FrameGrabber(auth_uri, rt.stop, rt.metrics)
        grabber.start()
        t_wait = time.monotonic()
        while grabber.latest()[0] is None:
            if time.monotonic() - t_wait > 30:
                raise RuntimeError("Could not read frames from the RTSP stream")
            time.sleep(0.1)

        logger.info("Loading AI models...")
        from detection.pipeline import WildlifePipeline   # heavy import, after the stream is confirmed
        pipeline = WildlifePipeline()
        worker = DetectionWorker(pipeline, rt.stop)
        worker.start()

        focus = FocusController(camera, grabber.frame, pose_source=lambda: rt.last_pose)
        tracker = PTZTracker(camera)
        tracker.focus = focus
        if tracker.observation_patrol is not None:
            tracker.observation_patrol.focus = focus
        publisher.start()

        control = threading.Thread(target=rt.control_loop, name="control",
                                   args=(camera, grabber, worker, tracker, focus, events), daemon=True)
        control.start()

        # ---- display / publish loop (main thread) ----
        # Feed ffmpeg at a steady YOUTUBE_FPS (an unsteady feed makes YouTube stutter).
        fps = config.YOUTUBE_FPS if publisher.proc is not None else config.OVERLAY_FPS
        period = 1.0 / max(1.0, fps)
        need_overlay = publisher.proc is not None or config.SHOW_LIVE_PREVIEW
        win = "Serengeti Wildlife AI"
        if config.SHOW_LIVE_PREVIEW:
            cv2.namedWindow(win)

            def on_mouse(event, x, y, _flags, _param):
                if event == cv2.EVENT_LBUTTONDOWN:     # click an animal: centre + zoom on it
                    rt.click = (x, y)
            cv2.setMouseCallback(win, on_mouse)
            logger.info("Preview: CLICK an animal to centre and zoom on it | W = go to the water-hole now | "
                        "M = save frame of a missed animal for training | Q = quit")
        while not rt.stop.is_set():
            t0 = time.monotonic()
            frame, _, _ = grabber.latest()
            frame = privacy_filter(frame, rt.last_pose)      # stream, clips AND preview
            if frame is not None:
                if events:
                    with rt.events_lock:
                        events.push_frame(frame)
                if need_overlay:
                    with rt.lock:
                        ann = dict(rt.annotation)
                    snap = rt.metrics.snapshot()
                    # YouTube shows confirmed animals only; the local preview also
                    # shows what the AI is still checking (grey "checking..." boxes).
                    if publisher.proc is not None:
                        publisher.publish(draw_overlay(frame.copy(), ann["tracks"], tracker, ann["status"],
                                                       snap, ann["health"],
                                                       show_candidates=config.SHOW_CANDIDATES_ON_OVERLAY))
                    if config.SHOW_LIVE_PREVIEW:
                        preview = draw_overlay(frame.copy(), ann["tracks"], tracker, ann["status"], snap,
                                               ann["health"], show_candidates=config.SHOW_CANDIDATES_ON_PREVIEW)
                        cv2.imshow(win, preview)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        if key in (ord("m"), ord("M")):
                            save_missed_example(frame, rt.last_pose)
                        if key in (ord("w"), ord("W")):
                            rt.goto_water = True
            spare = period - (time.monotonic() - t0)
            if spare > 0:
                rt.stop.wait(spare)
    except KeyboardInterrupt:
        pass
    finally:
        rt.stop.set()
        for t in (control, worker, grabber):
            if t is not None:
                t.join(timeout=5)
        if events:
            events.close()
        publisher.stop()
        try:
            camera.stop()
        except Exception:
            pass
        camera.close()
        if config.SHOW_LIVE_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
