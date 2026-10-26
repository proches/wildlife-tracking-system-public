"""Camera/stream health snapshot for unattended deployments."""

import json
import os
from datetime import datetime, timezone

import config


def write_health(camera, stream_ok=True, extra=None):
    status = camera.get_status() if camera is not None else None
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stream_ok": bool(stream_ok),
        "camera_status_ok": status is not None,
    }
    if status is not None:
        try:
            data["pan"] = float(status.PanTilt.x)
            data["tilt"] = float(status.PanTilt.y)
            data["zoom"] = float(status.Zoom.x)
        except (AttributeError, TypeError, ValueError):
            data["camera_status_ok"] = False
    if extra:
        data.update(extra)
    os.makedirs(os.path.dirname(config.SYSTEM_HEALTH_JSON) or ".", exist_ok=True)
    with open(config.SYSTEM_HEALTH_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data
