"""Central configuration for the Serengeti Wildlife Tracking System.

Secrets are read from environment variables.  Keep the wildlife safety arc
and other site-specific values here, but never commit camera passwords or
YouTube stream keys to source control.
"""

import os




def _load_local_dotenv():
    """Load a simple KEY=VALUE .env file without requiring python-dotenv."""
    path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                os.environ.setdefault(key, value)
    except OSError:
        pass


_load_local_dotenv()

def _env(name, default=""):
    return os.getenv(name, default)


def _env_bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Camera / network ---
CAMERA_IP = _env("WILDLIFE_CAMERA_IP", "10.140.0.12")
CAMERA_ONVIF_PORT = int(_env("WILDLIFE_CAMERA_ONVIF_PORT", "80"))
CAMERA_USERNAME = _env("WILDLIFE_CAMERA_USERNAME", "admin")
CAMERA_PASSWORD = _env("WILDLIFE_CAMERA_PASSWORD", "")   # set it in .env, never here
RTSP_TRANSPORT = _env("WILDLIFE_RTSP_TRANSPORT", "tcp")

# Camera orientation.  This SUNBA camera is mounted upside-down with EFlip=ON:
# it executes pan commands MIRRORED (ContinuousMove +x lowers the reported pan,
# AbsoluteMove x lands at reported -x) while GetStatus reports the normal pan.
# This was why the patrol drove into the arc edge and "stayed on one side", and
# why an AbsoluteMove once swung the camera to the hotel side.
# "auto" = value measured by tools/ptz_selftest.py, else read EFlip from the
# camera.  Force with "1" or "-1".
PTZ_PAN_COMMAND_SIGN = _env("WILDLIFE_PTZ_PAN_COMMAND_SIGN", "auto")
# GetStatus pan sign is unreliable: EVERY reply while the camera travels and
# ~5 % while it is still carry -pan (the magnitude is right).  A reading outside
# the arc whose mirror is inside it is sign-corrected; it is believed as real
# only if it stays unchanged (camera still) for this long.
PTZ_POSE_MIRROR_TOLERANCE = 0.05
PTZ_POSE_GLITCH_CONFIRM_SECONDS = 10.0

# --- Detection / classification ---
DEVICE = _env("WILDLIFE_DEVICE", "cpu")
# Lower thresholds improve recall on monkeys/birds in trees (MegaDetector often
# scores them weakly). False positives are reduced by multi-frame confirm + filters.
# v3: 0.10 was far too low - tree trunks, bushes, water glare and the camera's own
# on-screen text were all being called "animal".  Weak detections are still
# kept as *candidates*, but only become real animals after multi-frame
# confirmation (see TRACK_CONFIRM_* below) - so recall for far/small birds is kept
# without flooding the logs with false positives.
DETECTION_CONFIDENCE_THRESHOLD = 0.18
LOW_LIGHT_DETECTION_CONFIDENCE_THRESHOLD = 0.14
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.50

CLASSIFIER_BACKEND = _env("WILDLIFE_CLASSIFIER_BACKEND", "speciesnet")
SPECIESNET_MODEL = _env("WILDLIFE_SPECIESNET_MODEL", "kaggle:google/speciesnet/pyTorch/v4.0.3a/1")
SPECIESNET_COUNTRY = _env("WILDLIFE_SPECIESNET_COUNTRY", "TZA") or None
SPECIESNET_CONFIDENCE_THRESHOLD = 0.45
NON_ANIMAL_VETO_CONFIDENCE = 0.50
SPECIESNET_CACHE_SECONDS = 20.0   # reuse a label for an object at the same place (cleared when the view changes)
SPECIESNET_MAX_NEW_PER_FRAME = 3      # ~1.5 s per crop on this CPU; the rest come from the 20-s cache

# Optional site-specific YOLO model.  Disabled until the user's trained
# weights are supplied; the rest of the system works without it.
CUSTOM_YOLO_ENABLED = _env_bool("WILDLIFE_CUSTOM_YOLO_ENABLED", False)
CUSTOM_YOLO_MODEL_PATH = _env("WILDLIFE_CUSTOM_YOLO_MODEL_PATH", "")
CUSTOM_YOLO_CONFIDENCE = float(_env("WILDLIFE_CUSTOM_YOLO_CONFIDENCE", "0.18"))
CUSTOM_YOLO_IOU = float(_env("WILDLIFE_CUSTOM_YOLO_IOU", "0.50"))
CUSTOM_YOLO_FUSION_WEIGHT = float(_env("WILDLIFE_CUSTOM_YOLO_FUSION_WEIGHT", "0.60"))
# Inference size for the custom model.  If you trained on native-resolution
# 640 px tiles (training/build_dataset.py --tile) run it on the FULL frame at
# its native size (1920): small birds then keep the pixel size they had in training.
CUSTOM_YOLO_IMGSZ = int(_env("WILDLIFE_CUSTOM_YOLO_IMGSZ", "1920"))
# With the site model enabled it is the primary detector (see detection/pipeline.py)
MEGADETECTOR_WEIGHTS = "models/MDV6-yolov9-c.pt"   # local copy: no download at start
MEGADETECTOR_BACKGROUND_LOAD = True    # with the site model, start at once; MD joins when loaded
MEGADETECTOR_EVERY_N_WITH_CUSTOM = 3   # wide views: MegaDetector (1280 px, ~7 s on this i5) every N frames
# Zoomed views (water-hole, tracked animal): animals are big - smaller inputs are
# faster AND scored the water-hole zebras higher (0.79 vs 0.63 at 1280).
ZOOMED_VIEW_MIN_ZOOM = 0.10            # >= ~10x
FAR_TINY_BOX_PX = 40                   # wide view: smaller boxes are never verified by a species label alone
MEGADETECTOR_EVERY_N_ZOOMED = 2
MEGADETECTOR_EVERY_N_TRACKING = 2      # following a locked animal
MEGADETECTOR_EVERY_N_LOW_VIEW = 1      # camera looking down at the near ground: every frame
# Tracking pulses sized in pixels (tracking/ptz_tracker.py, 19/10).  The camera
# slows pan/tilt with the zoom, so each speed level moves the PICTURE at a fixed
# rate after a start-up delay - measured live: a 0.53 s fast pan pulse moved it 304 px,
# a 0.59 s fast tilt pulse 186 px (level rates from the pose measurements x 3.85 x px/unit).
TRACK_PAN_FAST, TRACK_PAN_FAST_PX_S = 0.30, 800.0
TRACK_PAN_SLOW, TRACK_PAN_SLOW_PX_S = 0.20, 68.0
TRACK_TILT_FAST, TRACK_TILT_FAST_PX_S = 0.25, 690.0
TRACK_TILT_SLOW, TRACK_TILT_SLOW_PX_S = 0.20, 49.0
TRACK_PAN_LATENCY, TRACK_TILT_LATENCY = 0.15, 0.32
TRACK_RATE_DROP_PER_ZOOM = 1.7         # picture rate x (1 - 1.7 x (zoom - 0.03)): 0.71 at 20x (measured)
TRACK_MIN_PULSE, TRACK_MAX_PULSE = 0.15, 0.9
TRACK_MOVE_FRACTION = 0.8              # move this much of the offset per AI result (no overshoot)
TRACK_LEAD = 0.6                       # aim this many AI-result intervals ahead of a walking animal
TRACK_TARGET_PX = 200                  # keep the animal about this big (user 19/10: closer)
TRACK_FAST_MOVER_FRACTION = 0.12       # moving more than this fraction of the width per result -> zoom out
# Frame-to-frame box tracker between AI results (tracking/box_flow.py)
FLOW_ENABLED = True
FLOW_WIDTH = 640                       # flow image width (1920 -> 640: cheap, ~5-10 ms)
FLOW_MAX_POINTS = 40
FLOW_MIN_POINTS = 6
FLOW_WIN = 15
FLOW_FB_MAX = 1.0                      # forward-backward error limit (small px)
FLOW_MAX_SPREAD = 3.0                  # points must move together (small px)
FLOW_INNER = 0.6                       # seed points in the inner 60 % of the box (its border is background)
FLOW_PTZ_INTERVAL = 0.6                # at most one visual-follow pulse per this many seconds
FLOW_KEEPALIVE_SECONDS = 20.0          # the visual tracker steers (and keeps the lock) at most this long after
                                       # the AI's last real sighting (21/10: unbounded -> a 42-min lock)
PIPELINE_TRACE_FILE = "logs/pipeline_trace.jsonl"
TRACK_PAN_IMAGE_SIGN = -1              # measured 19/10: pan up = picture moves right (centring tool, tracked bird)
TRACK_FRAME_GROUP = True               # frame the whole troop/herd around the target
TRACK_GROUP_RADIUS = 0.55
TRACK_GROUP_MIN_SCORE = 0.45           # unconfirmed boxes this strong also belong to the framed group
TRACK_GROUP_FILL_MIN = 0.40            # a framed group smaller than this share of the picture: zoom in
TRACK_GROUP_FILL_MAX = 0.80            # larger: zoom out (members would leave the picture)
TRACK_GROUP_FILL_TARGET = 0.60         # one-step approach to a far group: aim for this share
TRACK_ZOOM_STEP = 1.3                  # tracking zoom: one AbsoluteMove of x1.3 magnification per AI result
TRACK_ZOOM_IN_AFTER_OUT_SECONDS = 30.0 # no zooming in again this soon after a zoom out (26/10: pumping)
TRACK_FEEDFORWARD = True               # keep turning with a walking animal between AI results
TRACK_FF_PERIOD = 1.0                  # seconds between feedforward pulses
TRACK_FF_MIN_SIGHTINGS = 3             # feedforward / predictions only after the SAME animal was seen this often
TRACK_FF_MAX_LEAD = 0.5                # ...and never more than this share of the picture past its last sighting
TRACK_OTHERS_MEMORY = 45.0             # other animals seen this recently: a lock "hop" onto one of them is not motion
BOUNDARY_ZOOM_OUT_DX = 0.45            # (unused since 26/10: _boundary_zoom works from the picture edges)
BOUNDARY_LEAVE_MARGIN = 0.04           # held at the arc end: the followed animal this close to the edge -> zoom out
BOUNDARY_ZOOM_IN_MARGIN = 0.10         # ...zoom in only if, one step closer, it stays this far inside
BOUNDARY_LEAD_RESULTS = 1.5            # ...judged also where it will be this many AI results ahead
TRACK_FF_MIN_PX = 12                   # smaller expected drift: no pulse
TRACK_FF_WINDOW = 12.0                 # seconds of positions used for the walking speed
TRACK_FF_RESULT_SECONDS = 3.5          # ~AI interval (world speed -> "moves this far per result")              # fraction of the frame diagonal
MEGADETECTOR_ZOOMED_IMGSZ = 960        # 3.9 s instead of 6.7 s
CUSTOM_YOLO_ZOOMED_IMGSZ = 1920        # full resolution: small birds at the water-hole (only 0.3 s more)
ROI_SITE_IMAGE_SIZE = 640              # site model on the tracking region (~0.3 s)
MD_ONLY_MIN_CONFIDENCE = 0.35          # an MD box the site model does not confirm needs this
SKIP_TILES_WITH_CUSTOM = True          # MegaDetector tiles off: the site model tiles below are 10x cheaper
# Magnified site-model passes (detection/pipeline.py _custom_magnified):
WATER_MAGNIFY_ENABLED = True           # the water polygon, every frame, ~2x magnified (small birds)
WATER_MAGNIFY_IMGSZ = 1920            # 1.2 s per pass on this CPU (2560 = 2.3 s)
WATER_MAGNIFY_CONFIDENCE = 0.15
CUSTOM_TILES_PER_FRAME = 1             # while scanning: 1 of 6 tiles per frame (0.9 s), ~1.5x magnified; the slow sweep keeps a spot in view ~25 s
CUSTOM_TILE_IMGSZ = 1280
# Confirmation with the site model running: an object no classifier can name is
# an animal only if the site model itself saw it this many times (bushes, branches
# and logs fooled the generic detectors, not the site model).
CONFIRM_SITE_MODEL_HITS = 2
CONFIRM_STRONG_BOX = 0.60               # any box this strong seen twice = an animal (blank already vetoed)
CONFIRM_TWO_MODEL_MIN_BOX = 0.40        # MegaDetector box, seen twice...
CONFIRM_TWO_MODEL_MIN_SPECIES = 0.45    # ...+ SpeciesNet names an animal this sure
SITE_BIRD_VETO_EXEMPT = 0.45           # site-model "bird" this sure is not vetoed as "blank"
# Birds IN FLIGHT (tracking/flight.py, main._birds_in_flight).  24-26/10: 4 of 8 bird
# detections checked by eye were real birds, all flying past and seen by ONE AI result
# (confirmation needs two); the other 4 were trees/stumps, two of them at night.
FLIGHT_BIRDS_ENABLED = True
FLIGHT_MIN_SCORE = 0.40                # the four real flying birds scored 0.42-0.86
FLIGHT_MIN_RATIO = 3.0                 # box change vs the whole picture's change (aligned frames)
FLIGHT_MIN_DIFF = 6.0                  # and at least this many grey levels on average
FLIGHT_MIN_LOCAL = 2.0                 # ...and vs its surroundings (a tree in the wind changes all around)
FLIGHT_STAY_SHIFT = 0.35               # a shift of up to 0.35 box sizes that undoes the change...
FLIGHT_STAY_RESIDUAL = 0.5             # ...to half of it = a branch that swayed, not a bird that left
FLIGHT_SCALE = 0.5                     # frame ring: half-size grey copies...
FLIGHT_RING_FPS = 4.0                  # ...4 per second...
FLIGHT_RING_SECONDS = 16.0             # ...for 16 s (AI results come 5-13 s after the frame): ~33 MB
FLIGHT_NEIGHBOUR_MIN_S = 0.2           # frames this far before/after the analysed one are compared
FLIGHT_NEIGHBOUR_MAX_S = 1.0
NIGHT_DROP_BIRDS = True                # "birds" in night frames were stumps/bushes (2 of 2 checked,
                                       # plus the recurring 22/10 night false positive)
CUSTOM_YOLO_PERSON_NAMES = {"person", "human", "people"}
CUSTOM_YOLO_VEHICLE_NAMES = {"vehicle", "car", "truck", "bus", "motorcycle"}

# Distant wildlife enhancement.  The enhanced path is deliberately adaptive
# because the Intel NUC CPU can be overwhelmed by full tiled inference every frame.
# IMPORTANT: MegaDetector on CPU is ~5s/frame at 1280 — never run tiles every scan frame
# or the PTZ patrol cannot physically move.
ENHANCED_DETECTION_ENABLED = True
ENHANCED_DETECTION_MODE = "auto"  # auto | always | off
# MegaDetector letter-boxes every input to ~1280 px, so up-scaling the whole
# frame (old ENHANCED_UPSCALE) only wasted CPU.  Real gain for small birds comes
# from tiles that are *native resolution* -> 3 x 2 grid of the 1920x1080 frame.
ENHANCED_TILE_COLS = 3
ENHANCED_TILE_ROWS = 2
ENHANCED_TILE_OVERLAP = 0.25
ENHANCED_DETECTION_CONFIDENCE_THRESHOLD = 0.14
ENHANCED_NMS_IOU = 0.50
# While scanning/patrolling, only run full AI every N loop iterations so PTZ can move.
# Extra pure-PTZ patrol ticks between slow AI frames (keeps left/right motion alive).
# When already zoomed in (tracking or user-assisted), run AI every loop for better lock.

# Strict park wildlife mode: only animals/birds are tracked, logged and followed.
# Persons and vehicles are dropped before tracking / events / overlays.
IGNORE_PERSON_DETECTIONS = True
IGNORE_VEHICLE_DETECTIONS = True
# Drop detections that SpeciesNet/AI4G confidently labels as non-wildlife.
DROP_NON_WILDLIFE_SPECIES = True
EXCLUDED_PAN_RANGES = []

# --- Wildlife-safe pan arc / confirmed calibration ---
# High end extended slightly so the confirmed water-hole (pan≈0.7294) sits
# inside the software arc. User verified this still points at wildlife only.
ALLOWED_PAN_RANGE = (0.239861116, 0.735000000)
WILDLIFE_CENTER_PAN = 0.476055562
PAN_BOUNDARY_MARGIN = 0.03            # at the patrol zoom (3.85x); narrower in proportion when zoomed in
PAN_BOUNDARY_MARGIN_MIN = 0.010        # (the camera slows its pan in proportion to the zoom - 0.03 was
                                       # 1.2 picture widths at 14.7x and held zebras out of the centre)
# Measured on the SUNBA (wide zoom): speed <=0.10 does NOTHING, 0.15-0.22 crawls
# (~0.003/s), 0.30 moves ~0.038/s.  The old 0.22 cap meant live tracking could
# barely pan.  Pan speed also scales down with zoom on this camera.
BOUNDARY_PAN_MAX_SPEED = 0.40
PAN_RECOVERY_TOLERANCE = 0.01
PAN_RECOVERY_SPEED = 0.16          # lowest speed the camera actually executes
BOUNDARY_PAN_PULSE_SECONDS = 0.22
BOUNDARY_MOVE_TIMEOUT_SECONDS = 1.2

# --- Target acquisition / multi-object tracking ---
# 2 hits is enough once thresholds are lower; monkeys in foliage flicker a lot.
NEW_TARGET_CONFIRM_FRAMES = 1  # v3: multi-frame confirmation now happens in the object tracker
TRACKING_CONFIDENCE_THRESHOLD = 0.22
TARGET_STICKINESS_RADIUS = 0.30
TARGET_SWITCH_CONFIRM_FRAMES = 3
TARGET_MATCH_IOU = 0.02
TARGET_MOTION_PREDICTION_FRAMES = 4
TARGET_MAX_PREDICTED_RADIUS = 0.45
TARGET_VELOCITY_SMOOTHING = 0.65
# Hold lock longer through missed frames so we don't zoom-out and abandon monkeys.
TRACK_COAST_FRAMES = 14
TRACK_COAST_DECAY = 0.80
BOUNDARY_TARGET_RELEASE_FRAMES = 2
LOST_TARGET_FRAMES = 8

TRACK_ASSOC_MAX_DISTANCE = 0.18
TRACK_ASSOC_MIN_IOU = 0.05
TRACK_MAX_MISSES = 10
TRACK_CONFIRM_HITS = 3
TRACK_SPECIES_CONFIRM_HITS = 2
TRACK_VELOCITY_SMOOTHING = 0.70
TARGET_PRIORITY_CONFIDENCE_WEIGHT = 0.44
TARGET_PRIORITY_PERSISTENCE_WEIGHT = 0.20
TARGET_PRIORITY_SIZE_WEIGHT = 0.16
TARGET_PRIORITY_CENTER_WEIGHT = 0.10
TARGET_PRIORITY_SPECIES_WEIGHT = 0.10

# --- PTZ behaviour ---
FRAME_CENTER_DEADZONE = 0.06
# 26/10 17:25-17:49, zebras at the water-hole: 166 tilt pulses in 24 min for vertical offsets of
# +-0.09 (50 px) - up, down, up - and every move made the AI wait for a still picture (one
# result in 46 s: the lock was lost and the patrol took over).  Height in the picture matters
# far less than staying still: the vertical band is wider.
FRAME_CENTER_DEADZONE_Y = 0.20
# 29/10 18:13-18:19, impala group at the water-hole (Linux run): 70 pan pulses in 4 min, the
# direction flipping almost every pulse for offsets of only +-0.07..0.15 - the AI aim and the
# visual follow took turns correcting each other's overshoot, the picture shook and the lock was
# lost.  Once centred (or after crossing the centre) the camera now HOLDS until the animal is
# this far off-centre, and never turns back the other way sooner than TRACK_REVERSE_HOLD_SECONDS.
TRACK_HOLD_DEADZONE = 0.22
TRACK_REVERSE_HOLD_SECONDS = 3.0
TRACK_REVERSE_OVERRIDE = 0.45          # ...unless the animal is really leaving (this far off-centre)
PAN_TILT_GAIN = 0.70
MAX_PAN_TILT_SPEED = 0.30   # tracking: slow level for small offsets, ~7 deg/s only for a target far off-centre
ZOOM_GAIN = 0.45
MAX_ZOOM_SPEED = 0.42   # 18/10: 0.45-0.50 ramp up much faster (user: zoom too fast)
ZOOM_MIN_TRACKING_SPEED = 0.40   # measured: continuous zoom below ~0.35 does nothing
ZOOM_CENTER_TOLERANCE = 0.28
ZOOM_AREA_DEADZONE = 0.025
ZOOM_POSITION_MIN = 0.0
# Higher max zoom for distant park subjects (camera is set back from the mbuga).
# Still capped well below full optical range for safety and focus stability.
ZOOM_POSITION_MAX = 0.49   # ~47x: operator clicks may go this close (the lens reaches ~48x); automatic looks stop at LOOK_MAX_ZOOM
ZOOM_SEARCH_POSITION = 0.010
ZOOM_POSITION_MARGIN = 0.003
ZOOM_PULSE_SECONDS = 0.40
# Prefer a moderately filled frame so the tracker zooms in on distant animals.
TARGET_BOX_AREA_RATIO = 0.12
SEARCH_PAN_SPEED = 0.28
MAX_PAN_TILT_ACCEL = 0.50  # v3: results arrive every 1-3 s, ramping 0.12/step was far too sluggish
MAX_ZOOM_ACCEL = 0.35

# --- Smart patrol ---
PATROL_PRESET_NAMES = []
PATROL_DWELL_SECONDS = 20
SMART_PATROL_ENABLED = True
SMART_PATROL_STATIONS = 7
SMART_PATROL_TILT = -0.98
SMART_PATROL_POSITION_TOLERANCE = 0.012
# Short wide glance, then zoom; keep the sweep moving across the arc.
SMART_PATROL_WIDE_SECONDS = 2.5
SMART_PATROL_ZOOM_SECONDS = 7.0
# Observe zoom for normal stations (water-hole uses its own higher zoom).
SMART_PATROL_OBSERVE_ZOOM = 0.16
SMART_PATROL_HOTSPOT_DWELL_MULTIPLIER = 1.80
SMART_PATROL_ACTIVITY_DECAY = 0.995
# Longer continuous pan pulse while travelling between scan stations (still safe).
SMART_PATROL_TRAVEL_PULSE_SECONDS = 0.35
# Confirmed water-hole — main animal/bird gathering area.
# v3.1: re-taken 2026-10-16 from the pose the user aimed the camera at the water
# (pan 0.7275, zoom 0.145 = 14.8x).  The old zoom 0.32 (~30x) was so tight that
# focus was critical and heat haze softened the picture; 14.8x focuses reliably.
WATER_HOLE_POSITION = {
    "pan": 0.7275,
    "tilt": -1.0,
    "zoom": 0.145,
}
WATER_HOLE_DWELL_SECONDS = 60.0        # half at 21.8x, half at 41x (birds)


def _water_hole_override():
    """tools/set_water_hole.py saves the exact, centred water-hole pose here."""
    import json
    try:
        with open(os.path.join(os.path.dirname(__file__), "water_hole.json"), encoding="utf-8") as f:
            data = json.load(f)
        return {"pan": float(data["pan"]), "tilt": float(data["tilt"]), "zoom": float(data["zoom"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


WATER_HOLE_POSITION = _water_hole_override() or WATER_HOLE_POSITION

# --- Operational reliability ---
MAX_CONSECUTIVE_DROPPED_FRAMES = 30
STREAM_RECONNECT_DELAY_SECONDS = 2
CAMERA_STATUS_INTERVAL_SECONDS = 3

# --- Review / metrics ---
LOG_DIR = "logs"
SAVE_ANNOTATED_FRAMES = True
LOW_CONFIDENCE_REVIEW_DIR = "logs/needs_review"
METRICS_JSONL = "logs/metrics.jsonl"
EVENTS_CSV = "logs/events.csv"
SYSTEM_HEALTH_JSON = "logs/system_health.json"
SAVE_LOW_CONFIDENCE_EVERY_SECONDS = 2.0
LOW_CONFIDENCE_MIN_AREA_RATIO = 0.0004
# WILDLIFE_PREVIEW=0: no preview window (deploy/linux/start_tracker.sh, when no desktop is logged in)
SHOW_LIVE_PREVIEW = os.environ.get("WILDLIFE_PREVIEW", "1") != "0"

# --- Tourism / event highlights ---
EVENTS_ENABLED = True
EVENT_CLIPS_ENABLED = True
EVENT_CLIP_DIR = "logs/highlights"
EVENT_PREBUFFER_SECONDS = 3
EVENT_POSTBUFFER_SECONDS = 6
EVENT_MIN_CONFIDENCE = 0.55
EVENT_MIN_TRACK_HITS = 3
EVENT_COOLDOWN_SECONDS = 15
EVENT_MAX_CLIP_WIDTH = 1280
EVENT_CLIP_FPS = 10.0   # must match the rate frames are pushed (OVERLAY_FPS / YOUTUBE_FPS)

# --- Optional YouTube annotated publisher ---
YOUTUBE_ENABLED = _env_bool("WILDLIFE_YOUTUBE_ENABLED", False)
YOUTUBE_RTMP_BASE = _env("WILDLIFE_YOUTUBE_RTMP_BASE", "rtmp://a.rtmp.youtube.com/live2")
YOUTUBE_STREAM_KEY = _env("WILDLIFE_YOUTUBE_STREAM_KEY", "")
YOUTUBE_WIDTH = int(_env("WILDLIFE_YOUTUBE_WIDTH", "1280"))
YOUTUBE_HEIGHT = int(_env("WILDLIFE_YOUTUBE_HEIGHT", "720"))
YOUTUBE_FPS = float(_env("WILDLIFE_YOUTUBE_FPS", "15"))
YOUTUBE_BITRATE = _env("WILDLIFE_YOUTUBE_BITRATE", "4500k")
FFMPEG_BIN = _env("WILDLIFE_FFMPEG_BIN", "ffmpeg")

# --- Optional dashboard ---
DASHBOARD_HOST = _env("WILDLIFE_DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(_env("WILDLIFE_DASHBOARD_PORT", "8090"))


# =====================================================================
# v3 additions
# =====================================================================
VERSION = "3.1"

# --- Threaded runtime -------------------------------------------------
# Capture, AI and PTZ control now run in separate threads.  The camera is
# commanded CONTROL_LOOP_HZ times a second no matter how slow the CPU AI is
# (it used to move only once per AI frame = every 5-12 s -> "camera is stuck").
CONTROL_LOOP_HZ = 8.0
SLOW_TICK_SECONDS = 1.5                # a control tick / gap longer than this is logged with its cause
OVERLAY_FPS = 10.0      # 18/10: 15 -> 10, CPU for the AI (2-core i5)
# Detections older than this are never used to steer the camera.
DETECTION_MAX_AGE_SECONDS = 6.0
# Run a full-frame pass at least this often while tracking (other animals);
# in between, a fast ROI pass around the locked animal is used.
TRACK_FULLFRAME_EVERY_N = 1   # 19/10: every tracking pass looks at the whole frame (ROI passes missed walking baboons)
TRACK_ROI_SCALE = 3.2
TRACK_ROI_MIN_SIZE = 640
TRACK_ROI_MAX_PAN_SHIFT = 0.002        # camera turned more than this since the last result -> full frame
TRACK_ROI_MAX_ZOOM_SHIFT = 0.01
# Camera motion between two AI results, in pixels (multi_object_tracker): measured
# 19/10 on a buffalo tracked during the sweep - ~20000 px per pan unit at 3.85x.
PAN_PIXELS_PER_UNIT_AT_1X = 5200.0
TILT_PIXELS_PER_UNIT_AT_1X = 2600.0    # estimate (tilt range ~ half the pan range per unit)
TRACK_RELOCK_UNCONFIRMED = True        # locked target may be re-found as an unconfirmed track near its spot

# --- Only look when the picture is sharp -------------------------------
# Frames captured while the camera is panning / zooming / refocusing are
# motion-blurred and were a major source of "ghost" animals and vehicles.
DETECT_ONLY_WHEN_STILL = True
SETTLE_AFTER_PAN_SECONDS = 1.0
SETTLE_AFTER_ZOOM_SECONDS = 3.0   # measured: AF is sharp 2-3 s after the zoom stops

# --- On-screen-display / near-field masks (fractions of the frame) -----
# (x1, y1, x2, y2).  A detection whose centre falls in a mask is dropped.
# Defaults cover the camera's date/time stamp (top-left) and "Z: 11.08X"
# zoom label (bottom-right), both of which were detected as animals.
OSD_MASKS = [
    (0.00, 0.00, 0.30, 0.09),
    (0.78, 0.89, 1.00, 1.00),
]
# Things physically close to the camera (poles, tanks, staff, furniture).
# Set to [] if your camera never sees them.
NEAR_FIELD_MASKS = [
    (0.00, 0.93, 1.00, 1.00),
]
EDGE_SLIVER_MAX_PX = 28        # box touching frame edge and thinner than this
MIN_BOX_PIXELS = 9             # smaller than this (w or h) is noise
MAX_BOX_AREA_RATIO = 0.60      # an "animal" covering most of the frame is a smear
MIN_BOX_ASPECT = 0.12          # w/h and h/w limits for sanity
MAX_BOX_ASPECT = 8.0

# --- Confirmation / false-positive suppression -------------------------
# A track becomes a real animal when it has TRACK_CONFIRM_HITS hits AND at
# least one of: strong detector score, verified species, or sits in the
# water zone with a decent score.
STRONG_DETECTION_CONFIDENCE = 0.50   # v3.1: swaying branches/bushes reach 0.40-0.45
VERIFIED_SPECIES_CONFIDENCE = 0.60
VERIFIED_MIN_BOX_CONFIDENCE = 0.25     # identified (trusted classifier) + this box score = confirmed at once
WATER_ZONE_MIN_CONFIDENCE = 0.35     # reeds / bushes at the water edge
CONFIRM_MEAN_CONFIDENCE = 0.45
# Static false-positive memory (stumps, bushes, rocks seen again and again at
# the same place from the same camera pose, never moving, never identified).
STATIC_FP_ENABLED = True
STATIC_FP_MIN_HITS = 4
STATIC_FP_MIN_SECONDS = 25.0
STATIC_FP_MOVE_TOLERANCE = 0.035   # fraction of frame diagonal
STATIC_FP_POSE_TOLERANCE = (0.012, 0.04, 0.015)   # pan, tilt, zoom
STATIC_FP_FILE = "logs/static_false_positives.json"
STATIC_FP_ANIMAL_LABEL_MIN = 0.35      # a classifier animal label this sure -> never learned as clutter
# 26/10 17:19: zebras drinking at the water-hole were thrown away as "static clutter" - two
# spots learned there (today 13:59 and 23/10) were trusted for ever and checked BEFORE the
# classifier, so an animal standing on one never got a chance to be recognised.
STATIC_FP_FORGET_HOURS = 6.0           # a learned spot is trusted this long, then evaluated afresh
STATIC_FP_RECHECK_CONFIDENCE = 0.60    # a box this sure on a learned spot still goes to the classifier
STATIC_FP_SITE_MODEL_MIN = 0.45        # (unused since 19/10: site model alone is not animal evidence)
TOP_EDGE_PX = 4                        # prefilter "top_foliage": a box touching the top edge...
TOP_FOLIAGE_MIN_HEIGHT = 0.20          # ...at least this tall (fraction of the frame)...
TOP_FOLIAGE_MAX_BOTTOM = 0.45          # ...and ending in the upper part = tree crown / sky
# Safety stop (camera moved away from the glide target): retry instead of freezing
PTZ_FAULT_RETRY_SECONDS = 60.0
PTZ_FAULT_MAX_RETRIES = 3
# Operator takeover: the camera moves while the system sent nothing -> someone is
# steering it by hand.  Hold there (AI running) until it has been still this long.
# Motion cue (utils/motion_cue.py): something moving on its own (animal walking,
# bird flying) stops the sweep and - if it keeps moving while the camera is
# still - is zoomed in on, even when the detectors do not recognise it.
MOTION_CUE_ENABLED = True              # only on a STILL camera (a station, a pause, an operator view)
MOTION_CUE_WHILE_SWEEPING = False      # 21/10: 169 sweep stops for movement alone, 0 animals (wind in acacias)
MOTION_ROI_SIDE = 640                  # still view + movement: the next AI look is this crop at native resolution
MOTION_ROI_MAX_AGE = 10.0              # (21/10: MegaDetector scored 3 of 7 missed small primates >= 0.35 on such a
                                       #  crop; 10 s: the AI is busy 5-10 s, a 3-s cue was always stale)
MOTION_ROI_MIN_GAP = 6.0               #  crop, 0 of 7 on the whole frame; 2x upscaled crops scored 0 of 7)
MOTION_CUE_PERIOD = 0.33               # seconds between comparisons
MOTION_CUE_WIDTH = 320                 # comparison image width (cheap)
MOTION_CUE_MIN_DIFF = 14.0
MOTION_CUE_MIN_AREA = 4                # small-image pixels (~ a 25x25 px object in 1080p)
MOTION_CUE_MAX_AREA = 600
MOTION_CUE_MAX_BLOBS = 6               # more at once = wind / heat haze, ignore
MOTION_CUE_WINDOW = 2.0
MOTION_CUE_MATCH_PX = 12
MOTION_CUE_MIN_REPEATS = 2
MOTION_CUE_MIN_TRAVEL = 5.0            # small-image px the blob must move (~24 px in 1080p): swaying leaves stay put
MOTION_CUE_MAX_ZOOM = 0.06
MOTION_CUE_EDGE_MARGIN = 0.04          # ignore movement this close to the frame border (near-field foliage)
MOTION_CUE_INVESTIGATE_EVENTS = 3      # moving cues within 4 s on a still camera before zooming in
MOTION_CUE_INVESTIGATE = False         # 19/10: motion ALONE never zooms (a tree crown in the wind did) - only pauses
MOTION_CUE_MAX_EXTRA = 8.0             # motion may extend a stop by at most this many seconds
MOTION_MUTE_EPISODES = 2               # same spot moving on this many separate visits -> muted
MOTION_MUTE_EPISODE_GAP = 60.0         # seconds between cues that make them separate visits
MOTION_MUTE_SECONDS = 3600.0             # wider views only (heat haze moves everything when zoomed)
CAMERA_CALL_TIMEOUT = 8.0              # seconds: no ONVIF command may block the control loop longer
# A confirmed animal that is small in the picture is approached in one smooth
# centre + zoom move (tracker zoom pulses alone took minutes).
APPROACH_SMALL_TARGETS = True
APPROACH_MAX_PX = 80                   # box size (largest side) below which we go closer
APPROACH_TARGET_PX = 200               # aim for this size after zooming
APPROACH_MAX_FACTOR = 4.0
APPROACH_ONLY_BELOW_ZOOM = 0.12        # one approach per sighting: above ~12x the tracker's own zoom takes over
APPROACH_RELOCK_MIN_SCORE = 0.25       # after zooming in: a box this sure near the centre is our animal
INVESTIGATE_RELOCK_MIN_SCORE = 0.40    # after an automatic zoom-in on a half-seen object
APPROACH_RELOCK_RADIUS = 0.30          # "near the centre" (fraction of the frame width/height)
# Labels SpeciesNet gives far wild animals here (no livestock inside the park)
PARK_LABEL_MAP = {"domestic horse": "african buffalo", "domestic cattle": "african buffalo",
                  "wild turkey": "helmeted guineafowl", "domestic turkey": "helmeted guineafowl",
                  "domestic chicken": "helmeted guineafowl"}   # 19/10: kanga in a tree = "wild turkey 0.97"
OPERATOR_DETECT = True
OPERATOR_QUIET_SECONDS = 1.5           # no command of ours for this long...
OPERATOR_MOVE_PAN = 0.006              # ...and the pose changed by more than this
OPERATOR_MOVE_TILT = 0.015
OPERATOR_MOVE_ZOOM = 0.012
OPERATOR_RESUME_SECONDS = 300.0
OPERATOR_STARTUP_HOLD_SECONDS = 45.0   # started while not at a station: watch briefly, then patrol
# Auto-investigate (main._maybe_investigate): a candidate the AI keeps half-seeing
# on a still camera is centred and zoomed in on automatically (ManualLook).
AUTO_INVESTIGATE = True
AUTO_INVESTIGATE_MIN_SCORE = 0.30
AUTO_INVESTIGATE_SINGLE_SCORE = 0.40   # this sure: no need to wait for a second sighting
AUTO_INVESTIGATE_HOLD_SECONDS = 25.0   # automatic looks hold shorter than an operator click (90 s)
AUTO_INVESTIGATE_MAX_ZOOM = 0.15       # already >= ~15x: the animal is big enough - confirm it, do not zoom further
LOOK_MAX_ZOOM = 0.25                   # automatic looks / approaches stop at ~25x (41x made a bird blurry, 19/10)
TRACK_MAX_ZOOM = 0.28                  # the tracker never zooms in beyond ~27x (sharp, enough detail)
AUTO_INVESTIGATE_COOLDOWN = 60.0
AUTO_INVESTIGATE_REPEAT_SECONDS = 600.0
# 28/10 10:09-10:14: the operator held the camera on an elephant herd behind branches for 5 min
# (11.6x, ~25 px per animal).  MegaDetector saw nothing (also on native 640-px crops), the site
# model boxed them 8 times (0.18-0.50) and SpeciesNet called every box "blank" (0.997+), so
# nothing was ever looked at closer.  In an OPERATOR view a spot the classifier vetoes twice now
# gets one automatic closer look.  65 h of trace: 5 such repeats in operator views, 100 in patrol
# views (mostly the same bushes every hour) - so not in patrol views.
OPERATOR_VETO_LOOK = True
OPERATOR_VETO_LOOK_MIN_SCORE = 0.20
OPERATOR_VETO_REPEAT_SECONDS = 90.0    # the herd's spot came back after 56 s
SHOW_CANDIDATES_ON_OVERLAY = False     # YouTube stream: confirmed animals only
SHOW_CANDIDATES_ON_PREVIEW = True
# Click-to-zoom from the preview window (tracking/manual_look.py)
MANUAL_ZOOM_FACTOR = 2.5               # magnification x this (capped at ZOOM_POSITION_MAX)
MANUAL_HOLD_SECONDS = 90.0             # watch the zoomed view with the AI before the patrol resumes
MANUAL_CENTER_TOLERANCE = 0.06         # fraction of the frame width
MANUAL_CENTER_MAX_SECONDS = 25.0
MANUAL_PULSE_SPEED = 0.20              # slow level
MANUAL_SETTLE_SECONDS = 0.6
CLICK_REPEAT_SECONDS = 1.5             # a click this soon on (almost) the same spot does not restart the look
OPERATOR_RELOCK_MIN_SCORE = 0.15       # after an operator click: a box this weak near the centre is the animal
LOCKED_MIN_SCORE = 0.15                # an animal already being followed is kept down to this box score
OVERLAY_MAX_PAN_SHIFT = 0.05           # boxes follow the camera up to this pan change, then disappear

# --- PTZ motion (closed loop, with position feedback) -------------------
# "auto": try AbsoluteMove first, verify the camera really moved; if not,
#         remember it and drive with closed-loop ContinuousMove instead.
PTZ_MOVE_MODE = "auto"            # auto | absolute | continuous
PTZ_ABSOLUTE_PROGRESS_TIMEOUT = 3.0
PTZ_ABSOLUTE_MIN_PROGRESS = 0.004
PTZ_ABSOLUTE_FAILS_BEFORE_FALLBACK = 2
PTZ_CAPABILITIES_FILE = "logs/ptz_capabilities.json"
PTZ_CONTINUOUS_TIMEOUT = 1.0      # camera auto-stops after this; it only knows WHOLE seconds (<1 s = ~2 s!)
PTZ_PAN_KP = 3.2
PTZ_PAN_MIN_SPEED = 0.20          # slow cinematic level (0.15-0.222 all = ~0.6 deg/s); >=0.224 = ~7 deg/s
PTZ_PAN_MAX_SPEED = 0.35
PTZ_TILT_MIN_SPEED = 0.16
PTZ_TILT_MAX_SPEED = 0.25
PTZ_ZOOM_KP = 2.5
PTZ_ZOOM_MIN_SPEED = 0.40
PTZ_ZOOM_MAX_SPEED = 0.42
PTZ_PAN_TOLERANCE = 0.012
PTZ_TILT_TOLERANCE = 0.05         # tilt is secondary; -0.98 vs -1.00 must count as arrived
PTZ_ZOOM_TOLERANCE = 0.008
# Patrol stations may be driven closer to the hard edge than live tracking.
PATROL_EDGE_MARGIN = 0.004
PATROL_EDGE_SLOW_ZONE = 0.02      # taper speed inside this distance of an edge
PATROL_EDGE_SLOW_SPEED = 0.26    # still the smooth ~8 deg/s level (<=0.22 only crawls)
PTZ_STALL_SECONDS = 10.0
# Cinematic movement (v3.1): the camera ignores AbsoluteMove speed and jumps at
# ~36 deg/s, so pan TRAVEL uses a constant slow ContinuousMove (0.24-0.30 all
# give ~0.046/s = ~8 deg/s on this camera; 0.40 = ~12 deg/s, 0.50 = ~16 deg/s).
# Zoom uses AbsoluteMove, which runs at the lens motor's own gentle speed.
PTZ_CINEMATIC = True
# 17/10 measurement (wide zoom): ANY speed 0.15-0.222 = 0.0034/s (~0.6 deg/s,
# smooth), 0.224-0.30 = ~0.04/s (~7 deg/s).  Nothing in between.  The user found
# 7 deg/s too fast; 0.6 deg/s is the slow cinematic glide v2 had.  The camera
# also slows pan when zoomed in (proportional speed).
PTZ_CINEMATIC_PAN_SPEED = 0.20
PTZ_CINEMATIC_STOP_LEAD = 0.008   # measured slide after Stop 0.0050+-0.0005, + half a control tick
PTZ_CINEMATIC_COAST_MAX = 0.010   # worst slide assumed when the target is near an arc edge
PTZ_CINEMATIC_EDGE_KEEPOUT = 0.002
PTZ_CINEMATIC_SETTLE_SECONDS = 0.7
PTZ_CINEMATIC_FINAL_TOL = 0.003   # residual above this gets a tiny (<1 deg) precise AbsoluteMove
PTZ_CINEMATIC_REFRESH_SECONDS = 0.4
PTZ_WRONG_WAY_TOLERANCE = 0.006    # error growing this much while gliding = safety stop
# Distance a single 1-second ContinuousMove at PTZ_CINEMATIC_PAN_SPEED travels
# when the camera is ALREADY gliding, slide included (measured 0.055-0.057 on
# 16/10; from standstill it is only 0.0388).  The last part of every trip is one
# such timed glide, so the camera stops itself by its own clock.
PTZ_CINEMATIC_FINAL_GLIDE = 0.004   # 1-s command at the slow level, already gliding (0.0034/s + slide)
PTZ_CINEMATIC_REST_GLIDE = 0.003     # same, from standstill
PTZ_CINEMATIC_GLIDE_WINDOW = 0.0015   # final glide is started when it lands within this of the target
PTZ_CINEMATIC_EDGE_ZONE = 0.05        # target this close to an arc edge (water-hole, scan_7) ...
PTZ_CINEMATIC_EDGE_SHORTFALL = 0.003  # ... the glide lands this far short (never across)
PTZ_TRACE_CSV = "logs/ptz_trace.csv"
PTZ_TRACE_EVERY_SECONDS = 1.0

# --- Patrol schedule ------------------------------------------------------
# Visit the water-hole every N stations; the water-hole is where animals and
# birds actually are, so it gets more of the camera's time.
WATER_HOLE_EVERY_N_STATIONS = 0   # the slow sweep passes the water-hole (arc end) every round trip
# Scan stations closer than this to the water-hole are dropped: at wide zoom the
# water-hole view already covers them, and a hop that short could only be made
# with a fast AbsoluteMove jump (not cinematic).
SMART_PATROL_WATER_EXCLUSION = 0.05
# Starting (or restarting after a lock) this close to a station: go straight to
# its pose instead of zooming out first.
PATROL_DIRECT_START_PAN = 0.03
# Never leave an observation before the detector produced this many results
# on a settled frame (the CPU AI can take 3-8 s per frame).
PATROL_MIN_RESULTS_WIDE = 1       # the AI already watched the whole sweep on the way
# Scan stations are zoomed only when it is worth it (water-hole, a candidate the
# AI is verifying, or a hotspot); set True to zoom at every station.
PATROL_ZOOM_EVERY_STATION = False
# --- Station presets (17/10) ---------------------------------------------------
# The plain and the water-hole lie ABOVE tilt -1.0, which this camera can neither
# report nor accept in AbsoluteMove.  Each station is therefore stored as a
# preset inside the camera (tools/build_ptz_presets.py); the patrol glides to the
# station's pan (pan-only moves keep the tilt) and finishes with GotoPreset.
PTZ_PRESETS_FILE = "ptz_presets.json"
PRESET_MAX_APPROACH_PAN = 0.03       # GotoPreset only this close to the preset (safety)
# Patrol zoom glide (presets mode): AbsoluteMove zoom steps at a constant
# magnification rate.  0.30/s: 3.85x -> 21.8x in ~6 s, 30x -> 3.85x in ~7 s
# (one zoom command alone does 3.85x -> 21.8x in ~3 s, whatever speed is sent).
PTZ_ZOOM_GLIDE = True
PTZ_ZOOM_GLIDE_RATE = 0.30             # ln(magnification) per second
PTZ_ZOOM_GLIDE_STEP_SECONDS = 0.35
PTZ_ZOOM_MAG_PER_UNIT = 95.0           # measured: zoom 0.03 = 3.85x, 0.2193 = 21.8x, 0.30 = ~29.5x
PATROL_SWEEP_ZOOM = 0.03             # TRAVEL zoom 3.85x: the camera pans slower when zoomed in (user: keep the old speed)
# Station views: all presets 3.85x again (user 19/10: a zoom-in at every stop looked like
# "it saw something").  Zoom-ins now only mean something: water-hole, AI/motion look, click.
# Extra preset-only stations: name -> seconds watched (ptz_presets.json).  The LOW
# row (near ground below the plain, where baboons rest) was chosen 19/10 so that no
# view shows the lodge pool / guest areas (pan 0.70 does - left out).
PATROL_EXTRA_STATIONS = {"low_5": 12.0, "low_7": 12.0}   # 21/10: 3 s gave the AI (5-10 s per frame) one look -
                                                         # small monkeys crossed a still low view unseen
PATROL_LOW_MIN_RESULTS = 2             # 22/10: the low view takes 10-14 s per AI frame (MegaDetector every frame):
                                       # 12 s was ONE look - the low stations wait for two
# (baboon_spot is no stop any more - user 19/10: the low-row sweep passes over it, that is enough)
# (low_2/low_3/low_4 removed 19/10: next to the pool - guest loungers at the frame edge)
PATROL_TWO_ROWS = True                 # out along the plain row, back along the low row
# Guest privacy (utils/privacy.py): pan range where looking down (tilt reading
# above max_tilt) shows the lodge pool, loungers and towels.  Such views are
# blurred on the stream/clips/preview, never analysed or saved, the patrol lifts
# the camera before gliding through them and the tracker will not tilt into them.
PRIVACY_ZONES = [{"pan": (0.425, 0.80), "max_tilt": -0.97},      # 19/10: widened - whole pool side
                 {"pan": (0.58, 0.80), "max_tilt": 9.9,           # next to the pool, wider than ~3.4x
                  "min_zoom": 0.025}]                             # it shows at the bottom even at -1.0
# (27/10: both up to 0.80 - the zoomed-in arc now reaches 0.775, see PRIVACY_EDGE_PAN)
# Zoom-dependent end of the arc (user 27/10: "ndio ongeza ila pima" - extend it, but measure).
# Measured 27/10 08:23-08:27 with the camera at pans 0.735-0.785, zooms 0.025/0.10/0.144 and
# the lowest allowed tilt (-0.972): the lodge deck (planks, railing, chairs) starts at pan
# ~0.79 at the bottom-left of the picture; pool towels show only at the very bottom when
# tilted down at the widest zoom (already PRIVACY_ZONES).  So the camera may turn past 0.735
# only while zoomed in so far that the picture's LEFT edge stays PRIVACY_EDGE_SAFETY short
# of the deck: 0.735 at 3.9x (as before), ~0.766 at 10.5x, ~0.771 at 14.7x, 0.775 at most
# (beyond: the near tree with the lodge lanterns, no wildlife view).
PRIVACY_EDGE_PAN = 0.79
PRIVACY_EDGE_SAFETY = 0.005
PRIVACY_PX_PER_PAN_AT_1X = 4740.0      # measured 27/10: 160 px per 0.01 pan at 3.375x
PRIVACY_FRAME_WIDTH = 1920
PAN_HIGH_ZOOMED_MAX = 0.775
# Tilt glide before a station preset (measured 19/10: tilt has two speed levels -
# <=0.22 = 0.0049/s, >=0.25 = 0.069/s; GotoPreset swings it in under a second)
PATROL_TILT_GLIDE = True
PATROL_TILT_GLIDE_MIN = 0.02           # smaller differences: the preset alone
PATROL_TILT_SLOW_BAND = 0.04           # within this of the target: the slow (cinematic) level
PATROL_TILT_FAST_SPEED = 0.25
PATROL_TILT_SLOW_SPEED = 0.20
PATROL_TILT_FAST_LEAD = 0.035          # stop this early at the fast level (it slides on)
PATROL_TILT_SLOW_LEAD = 0.004
PATROL_TILT_GLIDE_MAX_SECONDS = 20.0             # ~3.9x: closer than 2.2x, the plain still fills the frame
SETTLE_AFTER_PRESET_SECONDS = 3.0
# Seconds of tilt-up (speed 0.3, 0.4-s steps) from tilt -1.0 that frame the plain
# at each scan station - chosen from calibration pictures 17/10.
STATION_TILT_UP_SECONDS = {"scan_2": 1.6, "scan_3": 1.6, "scan_4": 1.2,
                           "scan_5": 0.8, "scan_6": 1.0, "scan_7": 0.0}   # scan_6 re-aimed 19/10: pan 0.385, 1.0 s up (past the big tree)
# After half the water-hole dwell, zoom closer on the (centred) water for small birds.
WATER_HOLE_CLOSE_ZOOM = 0.42           # ~41x (19/10): the whole water-hole still fits, small birds ~2x bigger than at 21.8x
# Let the AI analyse frames while the camera glides slowly between stations
# (0.6 deg/s: no motion blur) - every part of the arc is searched, not only stations.
PATROL_DETECT_WHILE_SWEEPING = True
# Stop-and-verify: a candidate seen while sweeping pauses the glide so the AI can
# confirm (or reject) it on still frames, then the sweep continues.
PATROL_PAUSE_ON_CANDIDATE = True
PATROL_PAUSE_MIN_SCORE = 0.25            # 19/10: more sensitive (warthog/baboons were passed)
PATROL_VERIFY_SECONDS = 20.0           # longest pause
PATROL_VERIFY_MIN_RESULTS = 2          # AI results without the candidate before the sweep goes on
PATROL_VERIFY_MOTION_SECONDS = 3.0     # a pause for movement alone (no AI candidate) is this short
PATROL_VERIFY_MIN_SECONDS = 6.0        # shortest pause (>= one more AI result)
PATROL_PAUSE_REPEAT_PAN = 0.02         # no second pause this close to a recent one...
PATROL_PAUSE_REPEAT_SECONDS = 300.0    # ...within this time
# The camera moved by itself while observing (e.g. its own park/idle action):
# re-aim at the station when the pose drifts more than this.
PATROL_HOLD_PAN_TOLERANCE = 0.015
PATROL_HOLD_ZOOM_TOLERANCE = 0.03
PATROL_ZOOM_HOTSPOT_MIN = 0.3     # activity score / 5 (about 1.5 confirmed sightings)
PATROL_MIN_RESULTS_ZOOM = 2
PATROL_OBSERVE_MAX_WAIT = 30.0
LOST_TARGET_SECONDS = 10.0
LOST_TARGET_MIN_RESULTS = 3            # ...and only after this many AI results without the animal
LOST_TARGET_MAX_SECONDS = 40.0         # hard cap if the AI stalls
LOST_HOLD_SECONDS = 60.0               # after losing a target the camera stays on that view this long
LOST_HOLD_SECONDS_GROUP = 180.0        # ...a herd (behind trees, still there) this long
LOST_HOLD_GROUP_MIN = 3                # animals framed together that count as a herd

# --- Water zone -------------------------------------------------------------
WATER_ROI_FILE = "water_roi.json"
WATER_ZONE_PRIORITY_BONUS = 0.25
WATER_ZONE_ROI_PASS = True          # extra full-resolution pass over the water ROI
WATER_ZONE_ROI_CONFIDENCE = 0.12

# --- Focus (ONVIF Imaging service) -------------------------------------------
# Measured on the SUNBA lens: AUTO refocuses by itself 2-3 s after each zoom;
# at 14.8x a focus error of 0.01 makes the picture 10-25x softer.  See camera/focus.py.
FOCUS_CONTROL_ENABLED = True
FOCUS_AUTOFOCUS_MODE = "AUTO"       # put the lens in continuous autofocus
FOCUS_REFOCUS_AFTER_ZOOM = True
FOCUS_SHARPNESS_MIN_RATIO = 0.55    # search when sharpness < ratio * best seen at this pose
FOCUS_HUNT_MAX_SECONDS = 12.0
FOCUS_SETTLE_MAX_SECONDS = 5.0      # wait this long at most for AF to finish
FOCUS_SEARCH_STEP = 0.004           # absolute-focus step of the software search
FOCUS_MOVE_WAIT_SECONDS = 0.6
FOCUS_BEST_DECAY = 0.995            # per second: "best sharpness" halves in ~2.3 min (haze, light)
FOCUS_NUDGE_SPEED = 0.25            # (legacy continuous nudge, unused by the v3.1 controller)
FOCUS_NUDGE_SECONDS = 0.35
# Never send a soft frame to the AI: it must reach this fraction of the best
# sharpness recently seen at the same pan/zoom.
DETECT_REQUIRE_SHARP_FRAME = True
DETECT_MIN_SHARPNESS_RATIO = 0.45
SOFT_FRAME_MAX_BLOCK_SECONDS = 15.0  # fail-open: never starve the AI longer than this

# --- Review queue / training data ---------------------------------------------
# Save FULL frames + YOLO pre-labels (not tiny crops) so they can be corrected
# and used for training.  One save per track per REVIEW_MIN_SECONDS_PER_TRACK.
REVIEW_SAVE_FULL_FRAMES = True
REVIEW_MIN_SECONDS_PER_TRACK = 45.0
REVIEW_DIR = "logs/review"
LOG_UNCONFIRMED_DETECTIONS = False

# Tiles are time-boxed: they are processed water-hole first, then round-robin,
# until this budget is spent, so a slow CPU never stalls the system.
# Measured on the i5-7200U: MegaDetector needs ~5.5 s per 1280 px pass (full frame
# or one tile).  4 s = one tile per frame -> ~11 s per AI result.
ENHANCED_TIME_BUDGET_SECONDS = 4.0
# CPU threads for the AI (torch).  This i5 has 4: leave 2 for camera control,
# video capture and the overlay - with all 4 the PTZ loop froze for 2-3 s.
AI_TORCH_THREADS = 2
# A camera found slightly outside the arc (e.g. moved by hand) is brought back
# with one precise AbsoluteMove; farther out it is left alone (safety).
PAN_ARC_RECOVERY_MAX = 0.03
# Optional: override MegaDetector input size for fast ROI (tracking) passes.
# Only honoured if the installed PytorchWildlife exposes IMAGE_SIZE.
ROI_DETECTOR_IMAGE_SIZE = 640
