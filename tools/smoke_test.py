"""Smoke test for the AI architecture without a real camera or model weights."""

import sys
import types
from pathlib import Path

# Ensure project root is on sys.path when running as: python tools/smoke_test.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sup = types.ModuleType("supervision")
sys.modules.setdefault("supervision", sup)
pw = types.ModuleType("PytorchWildlife")
pwm = types.ModuleType("PytorchWildlife.models")
pwd = types.ModuleType("PytorchWildlife.models.detection")
pwc = types.ModuleType("PytorchWildlife.models.classification")
pwd.MegaDetectorV6 = object
pwc.AI4GSnapshotSerengeti = object
sys.modules.update({
    "PytorchWildlife": pw,
    "PytorchWildlife.models": pwm,
    "PytorchWildlife.models.detection": pwd,
    "PytorchWildlife.models.classification": pwc,
})
onvif = types.ModuleType("onvif")
onvif.ONVIFCamera = object
sys.modules["onvif"] = onvif

from detection.pipeline import WildlifePipeline
from tracking.multi_object_tracker import MultiObjectTracker
from tracking.ptz_tracker import PTZTracker
from utils.runtime_metrics import RuntimeMetrics

items = [
    {"box": [100, 100, 180, 180], "class_id": 0, "det_confidence": .8, "species": "unidentified", "species_confidence": 0},
    {"box": [500, 120, 580, 200], "class_id": 0, "det_confidence": .9, "species": "plains zebra", "species_confidence": .88},
]
tracker = MultiObjectTracker()
out1 = tracker.update(items, 1000, 800)
out2 = tracker.update([dict(items[0], box=[110, 105, 190, 185]), dict(items[1], box=[515, 122, 595, 202])], 1000, 800)
assert len({d["track_id"] for d in out2}) == 2

pipe = WildlifePipeline.__new__(WildlifePipeline)
assert pipe._class_aware_nms([
    {"box":[0,0,100,100],"class_id":0,"det_confidence":.8},
    {"box":[5,5,100,100],"class_id":0,"det_confidence":.7}], 0.5).__len__() == 1

m = RuntimeMetrics(); m.record_frame(.1, .02, out2)
assert m.snapshot()["frames"] == 1

print("SMOKE TEST: PASS")
