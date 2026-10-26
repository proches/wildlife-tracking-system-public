---
name: detection-tracking-evidence
description: "Measured 20/10/2026: AI cadence, MD-only threshold eval (no gain), recovery-hold AI starvation bug, how to verify detection/tracking claims"
metadata:
  node_type: memory
  type: project
  originSessionId: 7dd85fa5-55e5-4829-b010-059ecc851521
  modified: 2026-10-23T09:18:09.582Z
---

Measured facts from the 20/10/2026 detection + tracking review:

- AI result every 4-5.5 s on the i5 (inference 3.6-3.8 s). Between results the camera follows the box with tracking/box_flow.py (optical flow on every frame, re-anchored by each AI result). One state machine in main._update_state (PATROL / CANDIDATE / ACQUIRING / TRACKING / TARGET_LOST / RECOVERY / OPERATOR) logs "STATE A -> B: reason"; logs/pipeline_trace.jsonl records AI results and tracker PTZ commands.
- Detection eval set: tools/eval_set_20oct.json (65 real frames) + tools/eval_detection.py. Lowering MD_ONLY_MIN_CONFIDENCE 0.35 -> 0.20 recovered 0/13 of the operator-seen, AI-missed primates (weak MD boxes are vetoed by SpeciesNet "blank") and cost +0.5 s/frame, so the threshold stays 0.35. Site model round 1 scores 0/13 on them; MD tiles find 5/13 but cost 19.5 s/frame.
- Bug found by tests/test_tracking_spec.py and confirmed in logs/ptz_trace.csv (ai_busy=0 for the whole 60-s hold in 4 of 4 holds, 19-20/10): after a lost target the parked patrol (phase "bootstrap") counted as camera motion and closed the AI gate during RECOVERY. Fixed 20/10 in main._tick.
- Round-2 training ran 20/10 20:30 -> 21/10 06:32 and was NOT accepted: mAP50 0.640 vs 0.663 for the current model, and 3x more false boxes on empty site tiles. The 13 M-key primate frames got no teacher labels (MegaDetector, the teacher, cannot see them), so they were left out by design. Teacher-labelled training cannot fix the primate blind spot; those frames need hand-drawn boxes.
- 21/10 06:59-07:41 field bug, fixed the same day: a vanished buffalo stayed locked for 42 min with 2223 tilt pulses into an unmoving picture. Cause: main._visual_follow re-anchored on every AI result (also empty ones) and refreshed the flow keepalive. It now re-anchors only when PTZTracker.target_in_result is set, and stops steering FLOW_KEEPALIVE_SECONDS after the last real sighting (tests/test_tracking_spec.py test_16). A "pulse had no effect" guard based on phaseCorrelate was tried and removed: at 25x the unwindowed correlation reports a confident (0, 0) while the camera moves, and that dropped a walking animal.

- 21/10 09:00: the patrol stood still 5 min with no target. A 3-s control-loop stall during the patrol's own tilt glide made main._watch_operator compare against a pose taken before the stall, so our own move counted as "moved by hand". Fixed: the reference pose must be taken after our motion ended (op_anchor_t; test_17/17b). Measured cause of the stalls: the SUNBA answers GetStatus slowly (~1.7-2 s) about once a minute at second :15-:24 (ptz_trace.csv gaps + "Slow control tick ... camera pose 1.7s" logs). All ONVIF calls share one lock (OnvifPTZCamera._run), so a slow call blocks every command.
- 21/10 sweep audit (204 sweep stops, 19/10 16:20 to 21/10 09:38). Motion-only stops: 169, 0 animals, so the motion cue now acts only on a still camera (MOTION_CUE_WHILE_SWEEPING False). Of the 64 hand-checked candidates (fixture SWEEP_STOPS_21_10 in tests/test_tracking_spec.py):
  - boxes SpeciesNet NAMES were real 13 of 15 in daylight;
  - generic boxes ("unidentified", bare "bird") during the sweep were a followable animal 1 of 16 times (acacias);
  - at night, 0 of 11.
  The sweep now stops and zooms only for named daylight boxes (main.worth_a_closer_look).
- Motion ROI probe on 7 known-location missed primates: MegaDetector scored 3 of 7 >= 0.35 on a 640 px native-resolution crop, 0 of 7 on the full frame, and 0 of 7 on 2x upscaled crops. Hence MOTION_ROI_*: a still-view motion cue sends the next AI job to that crop.
- Do not run heavy tests at normal priority while the tracker runs: on 21/10 it doubled AI time (9-13 s per frame) and small monkeys were missed. The tracker alone uses about 3.2 of the 4 threads. Use Idle priority.
- 22/10 17:38-17:50, waterbuck at the right end of the low row (pan about 0.25), all detected fine but left. Three causes, all fixed in tracking/ptz_tracker.py and main._visual_follow; regression tests Field22OctTests in tests/test_tracking_spec.py:
  - The arc-boundary rule released the lock 4 times ("scanning inward"). The tracker now holds at the arc edge.
  - The lock hopped between group members (each hop read as running), and feedforward drove the camera from pan 0.27 to 0.53 with no animal in view.
    - Velocity is now reset only when the new box lies closer to ANOTHER recently seen animal (TRACK_OTHERS_MEMORY 45 s) than to the expected position. The first attempt, a reset on every track-id change, broke walking animals: the MOT renumbered a walker 8 times in 44 results, and the world tests dropped to 18-24 % in-frame.
    - Feedforward is trusted only after 3 sightings, is slow-level only, and is capped at 0.5 picture width past the last sighting.
    - At the arc end the camera holds, widens the view as the animal nears the picture edge, and skips the approach. The walking-left sim went from 76 % to 99 % in-frame.
  - The flow was re-anchored at a bogus world prediction, so it followed a waving branch. It is now re-anchored on the AI box, camera-shifted.
- 23/10 09:28, a bird at scan_7 (right end) was left. Fixed in main._maybe_investigate; tests test_18e/18f.
  - The analysed frame was taken while travelling (pan 0.305). The result arrived 13 s later, with the camera still at pan 0.282, so the "still" branch used the LIVE frame as the ManualLook reference for a box from the old frame.
  - The look aimed 460 px off (camera_shift; PAN_PIXELS_PER_UNIT_AT_1X 5200), zoomed to 9.6x with the bird out of view, and gave up after 2 empty results.
  - Rule now: a detector box always uses res["frame"] as the reference; a motion box uses the live frame. phaseCorrelate recovered the 460 px shift on the real frame (resp 0.68).
  - The user took my visible test window (MELIA_WINDOW_TEST) for a tracker restart. Avoid visible test windows while they watch.
- 23/10 12:02, "the camera jerks instead of settling": a numpy.float32 pulse came from the optical-flow box.
  - The camera's timedelta() raised for every flow pulse >= 0.6 s (pulse + 0.6 > BOUNDARY_MOVE_TIMEOUT_SECONDS 1.2), and each failure called stop().
  - 640 failed moves 21/10 11:53 to 23/10, all on large corrections; small ones worked.
  - Fixed with float() at the camera, in PTZTracker._command and in follow_box (tests test_22, test_float32_pulse_from_optical_flow_is_sent).
  - SimCamera never built a timedelta, so the simulations could not see this. Grep tracker.log for "PTZ command failed" when tracking looks jerky.
- Birds: only 12 "bird" detections in 26 h (21-22/10). 2 were confirmed at the water-hole; most were one recurring night false positive at pan 0.385.
- Round 3 training (training/round3.py, logs/round3.log) adds:
  - hand-verified crops (training/verified_21oct.json: 25 animals, 7 birds, 35 negatives);
  - targeted LILA species;
  - frames since 21/10.

  The first run (22/10) never finished. It is re-run 23/10 20:30 with a deadline; see [[night-training-jobs]].
- Teacher labels contradict hand checks: of the 67 verified pictures, 11 animals/birds had a teacher label of "nothing", and 3 non-animals had a teacher box. Teacher "empty" labels on review-candidate frames are therefore not trustworthy negatives.
**Why:** the user kept seeing "found the animal, froze, lost it, back to patrol"; these were the measured causes, not guesses.
**How to apply:** before touching detection thresholds, run tools/eval_detection.py with the tracker stopped and compare against the table in its docstring (also use it to judge the round-2 model). For a lost-target complaint, read the STATE lines in tracker.log and ai_busy in ptz_trace.csv first. Related: [[sunba-camera-quirks]], [[wildlife-tracker-hardware]].
