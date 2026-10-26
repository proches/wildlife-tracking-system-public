---
name: night-training-jobs
description: "How the unattended night training + morning watchdog run (tasks, sizes, timings measured 20-23/10/2026) and why round 3 failed the first time"
metadata:
  node_type: memory
  type: project
  originSessionId: 7dd85fa5-55e5-4829-b010-059ecc851521
  modified: 2026-10-25T06:44:04.366Z
---

Scheduled tasks (Interactive principal; action wscript.exe //B tools\run_hidden.vbs <bat>, so there is NO console that can be closed):
- MeliaWildlifeTrainingRound3: one-shot 23/10/2026 20:30. It runs training\run_round3.bat, i.e. `training\round3.py --reuse --deadline 05:30`.
- MeliaWildlifeTrackerWatchdog: every 15 min since 23/10 15:05, tools\tracker_watchdog.py, log logs\watchdog.log.
  - It starts main.py if absent.
  - It restarts main.py if frozen: running more than 5 min with no logs\ptz_trace.csv line (1/s heartbeat) for 5 min.
  - It waits for finalize_model.py.
  - It stops other night jobs only 05:30-12:00.
  - It started the tracker through Task Scheduler on 23/10 14:52 (tested).
- 23/10 13:45-14:43 the tracker froze for 1 h with no error, most likely console QuickEdit (HKCU\Console QuickEdit=1). utils/logger.disable_console_quick_edit now clears it at start (verified via AttachConsole: mode 0x01b7).

Measured on this i5:
- Training is about 0.43 s per training image (3.4-3.5 s per batch of 8) plus about 0.2 s per validation image per epoch.
- Round 2 ran 20 min past its --time, and finalize took 18 min. Hence round3.training_hours stops 1 h before the deadline.
- The teacher (MD+SpeciesNet at 1280) is about 9 s per frame. 4048 frames took 10 h.

Round 3 first run (22-23/10) failed for these reasons:
- The harvest took every clip frame, and the set reached 26 000 tiles (3.1 h per epoch).
- The console was closed at about 07:46 (0xC000013A).
- The kill skipped `finally: start_tracker()`, and the tracker was off until 08:32.

The re-run set is 9 020 train / 2 457 val (counted, not yet trained):
- every 6th clip frame (training_data/site3s);
- empty-keep 0.3 on train only;
- a per-folder time split;
- training_data/site3v hand crops override teacher frames of the same second (14 of 67 contradicted the teacher).

Round 3 re-run (23-24/10) completed: 7 epochs in 8.05 h, finalize 21 min, tracker back 05:14.
- It was NOT accepted: mAP50 0.690 vs 0.718. Animals were equal, birds 0.679 vs 0.738, and vegetation false boxes 107 vs 121 (better).
- Ultralytics optimizer=auto had picked AdamW lr 0.001667 with warmup 3/7 epochs. Validation mAP50 peaked after epoch 1 (0.759) and then fell.

Round 4 (25/10 20:30, task MeliaWildlifeTrainingRound4, training/round4.py, logs/round4.log):
- preset cpu_finetune (AdamW lr 3e-4, warmup 0.5, save_period 1); best epoch by mAP50 (best_epoch_weights: results.csv epoch k is weights/epoch{k-1}.pt);
- lila4_yolo: 3 700 new Snapshot Serengeti images, 1 500 birds, 1 239 small-only, skipping the 4 817 already downloaded;
- finalize reports by_source (site/lila) and requires site mAP50 not worse;
- MARGIN 1.4 h.

Notes:
- Ultralytics --time overrides --epochs.
- finalize class_result(k) indexes ap_class_index, not class ids (subset crash fixed 25/10). finalize --no-install exists for safe tests.
- Snapshot Serengeti bbox subset: about 146k human boxes. Species file categories: gazellethomsons, gazellegrants, topi, hartebeest, eland ... (61).

**Why:** the user saw "cmd haikuwa ina-run" in the morning. The system must be up by day; a night job must never leave it off.
**How to apply:**
- Before any new night job, size it with the measured rates and keep the deadline margin.
- Launch it hidden.
- Check logs\round3.log, logs\watchdog.log and models\melia_wildlife_report.json in the morning.

Related: [[detection-tracking-evidence]], [[wildlife-tracker-hardware]].
