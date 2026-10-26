"""Validate safety and deployment configuration before a live run."""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as: python tools/validate_deployment.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config


def main():
    problems = []
    low, high = config.ALLOWED_PAN_RANGE
    if not (0.0 <= low < high <= 1.0):
        problems.append("ALLOWED_PAN_RANGE is invalid")
    if not config.CAMERA_PASSWORD:
        problems.append("WILDLIFE_CAMERA_PASSWORD is not set")
    if config.YOUTUBE_ENABLED and not config.YOUTUBE_STREAM_KEY:
        problems.append("YouTube enabled but WILDLIFE_YOUTUBE_STREAM_KEY is missing")
    if config.CUSTOM_YOLO_ENABLED and not Path(config.CUSTOM_YOLO_MODEL_PATH).exists():
        problems.append(f"Custom YOLO model missing: {config.CUSTOM_YOLO_MODEL_PATH}")

    calibration = _ROOT / "camera_boundaries.calibration.json"
    if calibration.exists():
        data = json.loads(calibration.read_text(encoding="utf-8"))
        water = data.get("water_hole")
        if water and not (low <= water.get("pan", -1) <= high):
            problems.append(
                f"Water-hole calibration pan={water.get('pan')} is outside software wildlife arc [{low}, {high}]. "
                "Do not silently widen the arc; re-measure a safe water-hole position."
            )

    if problems:
        print("DEPLOYMENT CHECK: FAIL")
        for problem in problems:
            print(" -", problem)
        return 1
    print("DEPLOYMENT CHECK: PASS")
    print("Pan arc:", config.ALLOWED_PAN_RANGE)
    print("Classifier:", config.CLASSIFIER_BACKEND)
    print("Custom YOLO:", config.CUSTOM_YOLO_ENABLED)
    print("YouTube publisher:", config.YOUTUBE_ENABLED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
