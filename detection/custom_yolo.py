"""Optional site-specific YOLO detector.

The base system remains fully functional without Ultralytics or a custom
weight file. When enabled, this detector supplies a second, local model that
can be trained on the Melia/Serengeti camera's actual viewpoints.
"""

import logging
import os

import config

logger = logging.getLogger("wildlife_tracker.custom_yolo")


class CustomYOLODetector:
    def __init__(self, device=config.DEVICE):
        self.model = None
        self.names = {}
        path = config.CUSTOM_YOLO_MODEL_PATH
        if not config.CUSTOM_YOLO_ENABLED:
            return
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Custom YOLO weights not found: {path!r}")
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Ultralytics is not installed; install requirements-custom-yolo.txt") from exc
        self.model = YOLO(path)
        self.names = getattr(self.model, "names", {}) or {}
        logger.info("Custom YOLO loaded: %s", path)

    @property
    def enabled(self):
        return self.model is not None

    def detect(self, frame_bgr, imgsz=None, conf=None):
        """``imgsz`` larger than the image = magnification (small far animals/birds)."""
        if not self.enabled:
            return []
        results = self.model.predict(
            source=frame_bgr,
            conf=config.CUSTOM_YOLO_CONFIDENCE if conf is None else conf,
            iou=config.CUSTOM_YOLO_IOU,
            imgsz=config.CUSTOM_YOLO_IMGSZ if imgsz is None else int(imgsz),
            device=config.DEVICE,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        out = []
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            raw_name = self.names.get(cls_id, str(cls_id))
            name = str(raw_name).strip().lower()
            if name in config.CUSTOM_YOLO_PERSON_NAMES:
                mapped_class = 1
            elif name in config.CUSTOM_YOLO_VEHICLE_NAMES:
                mapped_class = 2
            else:
                mapped_class = 0
            out.append({
                "box": [float(v) for v in xyxy],
                "class_id": mapped_class,
                "det_confidence": conf,
                "species": None,
                "species_confidence": None,
                "source": "custom_yolo",
                "source_label": name,
            })
        return out
