---
name: wildlife-tracker-hardware
description: "Runtime machine and environment limits for the wildlife tracker (CPU-only i5, Python 3.14, no SpeciesNet, backups)"
metadata:
  node_type: memory
  type: project
  originSessionId: 7dd85fa5-55e5-4829-b010-059ecc851521
  modified: 2026-10-20T15:36:12.675Z
---

- Runtime PC = this desktop: Intel i5-7200U, no GPU. MegaDetector V6 ~5.5 s per 1280 px pass; model load ~3 min. Training must go to Google Colab (training/colab_train.ipynb).
- Python 3.13.15 installed (user scope) on 2026-10-16; `.venv313` has SpeciesNet 5.0.5 + ultralytics and is the runtime (run_tracker.bat). The old `.venv` (3.14, no SpeciesNet) was deleted on 2026-10-19. PytorchWildlife pulls `yolov5`, which monkeypatches cv2.imwrite to 2 args - use utils/imageio.write_jpeg.
- Training data: training_data/ (LILA Snapshot Serengeti subset 3472 imgs + 199 site frames, teacher-labelled with SpeciesNet blank veto). First CPU training run (output really in runs/detect/runs/wildlife/melia_v1 - ultralytics prefixes runs/detect) started 2026-10-16 ~17:20 (preset cpu, 14 h cap); training/finalize_model.py gates and enables it.
- Not a git repo. `../wild_v3` = untouched copy of v3.0 code; `../wild` = old v2 run with logs (needs_review crops, one highlight mp4).
- The user writes in Swahili; replies go in Swahili.

Related: [[sunba-camera-quirks]]
