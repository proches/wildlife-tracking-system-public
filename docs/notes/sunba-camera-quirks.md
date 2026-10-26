---
name: sunba-camera-quirks
description: "Measured PTZ/focus behaviour of the SUNBA camera at 10.140.0.12 (EFlip mirrored pan, status sign glitches, speed dead-zone, focus)"
metadata:
  node_type: memory
  type: project
  originSessionId: 7dd85fa5-55e5-4829-b010-059ecc851521
  modified: 2026-10-21T11:02:51.533Z
---

Measured on the real camera 2026-10-16 (SUNBA Performance-Series, ONVIF):

- EFlip=ON (ceiling mount): pan COMMANDS are mirrored vs GetStatus. AbsoluteMove x lands at reported -x; ContinuousMove +x lowers reported pan. Handled by `pan_command_sign=-1` in camera/onvif_camera.py (stored in logs/ptz_capabilities.json by tools/ptz_selftest.py).
- GetStatus pan sign is unreliable: EVERY read while travelling, ~5% when idle, return -pan (magnitude exact). MoveStatus says IDLE even while moving — don't trust it.
- Pan speed dead-zone: <=0.10 no motion, 0.15-0.22 crawl (~0.003/s), 0.30 ~0.038/s at wide zoom; speed also scales down with zoom.
- AbsoluteMove IGNORES any speed (always ~36 deg/s jump). User wants CINEMATIC movement: pan travel uses constant ContinuousMove 0.30 (~8 deg/s; 0.24-0.30 identical, 0.40=12, 0.50=16 deg/s), stop early, precise small AbsoluteMove finish. Zoom via AbsoluteMove is gentle (lens motor).
- ContinuousMove Timeout is in WHOLE seconds; <1 s behaves like ~2 s (0.5 s moved 0.096). A 1 s command while already gliding at 0.30 travels 0.056 (from rest 0.0388); slide after Stop 0.005. Cinematic finish = one device-timed 1 s glide (PC latency from training/AI caused late Stops and arc overshoot).
- AbsoluteMove must not carry a PanTilt speed (empty PanTiltSpeedSpace). AbsoluteMove is precise (~0.0001) once the sign is right.
- Focus: AUTO refocuses 2-3 s after a zoom stops; MANUAL->AUTO switch does NOT refocus; at 14.8x a 0.01 focus error = 10-25x softer. Imaging supports absolute focus 0..1.
- Wildlife arc ALLOWED_PAN_RANGE (0.2399, 0.735); water-hole the user chose: pan 0.7275, zoom 0.145. Wrong-sign AbsoluteMove swings the camera to the hotel side — never send absolute pan moves without a measured sign.

- 17/10/2026 incident: unit tests wrote the REAL logs/ptz_capabilities.json (sign +1) and the tracker then drove the camera to the hotel side. Fixed: tests/__init__.py redirects all state files; the camera's live EFlip beats the file; MotionController safety-stops if the error grows while gliding. Also the AI used all 4 CPU threads and froze the PTZ loop 2-3 s -> AI_TORCH_THREADS=2; every cinematic command is sized so 1 s of travel can never pass the target.

- TILT is clamped: status reads -1.0 over a wide range and AbsoluteMove cannot go above -1.0, but the plain and water-hole are ABOVE it. Solution 17/10: camera presets (ptz_presets.json, tokens 1=water 0.723/0.2193, 2-7 scan stations at zoom 0.03 with calibrated tilt-up), pan-only glides keep tilt, zoom_only() keeps tilt, GotoPreset only within 0.03 pan. Camera also moves/zooms by itself when idle (park action?).
- Speed levels at wide zoom: 0.15-0.222 = 0.0034/s (slow, used), 0.224-0.30 = ~0.04/s; pan rate scales down with zoom (proportional).
- 18/10/2026: ZOOM ignores every speed for AbsoluteMove (3.85x->21.8x in ~3.4 s); continuous zoom-in has one accelerating speed (0.36-0.40 same), zoom-out -0.38 is slow (~0.2 ln/s). Magnification = 1 + 95*zoom. Slow zoom = stepped zoom_only glide (PTZ_ZOOM_GLIDE_RATE 0.30 ln/s). Water preset 1 re-centred (pan 0.7249) with tools/center_water_preset.py; water_wide preset token 8 (same pan/tilt, zoom 0.03). Pan guard blocks + pulses within 0.03 of the arc end (use edge_margin for nudges).
- 18/10/2026: with the USER'S APPROVAL main stream changed 60 fps -> 15 fps (GOP 120 -> 30) via ONVIF; old values in logs/camera_encoder_backup.json. Decode 124% -> 40% of a core; AI 12 s -> ~4 s/frame. PytorchWildlife MD inference size is predictor.args.imgsz (IMAGE_SIZE does nothing). A browser live-view "WebPlugin_IPC_NB" eats ~1 core.
- 19/10/2026: TILT also has two speed levels: <=0.22 = 0.0049/s, >=0.25 = 0.069/s; + command raises (reading falls). Tilt BELOW the clamp (looking down, readable > -1.0) works with AbsoluteMove. Low patrol row presets 10-13,15 + baboon_spot 9 (tilts -0.80..-0.93): pan 0.70 and lower tilts show the LODGE POOL/guest loungers - never use them (privacy, YouTube).
- 19/10/2026: site model (round 1) is blind to baboons; MegaDetector sees them 0.42-0.70. Round-2 training scheduled as Windows task "MeliaWildlifeTrainingRound2" (training/round2.py, log logs/round2.log); accepts the new model only if not worse than the old one.

**Why:** these caused "camera stuck on one side / zooms out and never returns" and several test-induced swings toward the hotel.
**How to apply:** any PTZ test must zoom out first, probe pan direction with tiny pulses using the predicted sign, stop on first motion, and check pose against the arc every read. See [[wildlife-tracker-hardware]].
- 19/10/2026 evening: TRACKER PAN SIGN WAS WRONG since the start (pan up = picture moves RIGHT; target right of centre needs pan DOWN) - verified live with tools in scratchpad dirtest (tracker pulses must move a fake target toward centre). Picture rates per level are ~zoom-invariant: pan fast 800 px/s (+0.15 s latency), tilt fast 690 px/s (+0.32 s), ~30 % slower at 20x. Always verify motion direction LIVE before trusting tracking changes; tests/test_world_scenarios.py is the end-to-end harness.
- 21/10/2026: a 0.4-s continuous ZOOM pulse at speed 0.4 coasts far past Stop, and further the higher the zoom: +0.09 at 17x, 26x -> 42x. Tracking zoom therefore uses one zoom_only AbsoluteMove step (x1.3 magnification, capped at TRACK_MAX_ZOOM), never continuous pulses near the limit.
- 19/10/2026 night: tracking is world-based for pan (animal pan from box + capture pose; feedforward pulses each 1 s between AI results; aim at predicted CURRENT position). Feedforward pulses must NOT set view_changed (that blocked all AI results). Privacy also covers wide zoom (<0.025) at pan 0.58-0.745. Old .venv (py3.14) deleted; only .venv313 exists.
