"""Draw the water-hole region once; the system uses it forever after.

    python tools/calibrate_water_roi.py            # moves camera to the water-hole pose, wide then zoomed
    python tools/calibrate_water_roi.py --here     # do NOT move the camera; use its current pose

For each pose a snapshot opens.  Click the corners of the water (and the muddy
bank where animals stand), press ENTER to accept / 'r' to redo / ESC to skip.
The polygons are saved with the camera pose they were drawn at in water_roi.json:
the ROI is only active when the camera is back at (nearly) that pose.

Why: colour-based "find the water" is unreliable on sunlit savanna (dry grass
is as bright as a puddle), but a one-time human-drawn polygon is exact.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from camera.onvif_camera import OnvifPTZCamera
from main import add_rtsp_credentials


def grab(uri, warm=15):
    cap = cv2.VideoCapture(uri)
    frame = None
    for _ in range(warm):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


def draw(frame, label):
    h, w = frame.shape[:2]
    pts = []
    win = f"Water ROI - {label}"
    scale = 1280.0 / w
    view = cv2.resize(frame, None, fx=scale, fy=scale)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = view.copy()
        for p in pts:
            cv2.circle(img, p, 4, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.polylines(img, [__import__("numpy").array(pts)], len(pts) >= 3, (0, 255, 255), 2)
        cv2.putText(img, "click water outline | ENTER=accept  r=redo  ESC=skip", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and len(pts) >= 3:
            cv2.destroyWindow(win)
            return [[x / view.shape[1], y / view.shape[0]] for x, y in pts]
        if k == ord("r"):
            pts.clear()
        if k == 27:
            cv2.destroyWindow(win)
            return None


def settle(cam, seconds=5.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--here", action="store_true")
    args = ap.parse_args()
    cam = OnvifPTZCamera(config.CAMERA_IP, config.CAMERA_ONVIF_PORT, config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    uri = add_rtsp_credentials(cam.get_stream_uri(), config.CAMERA_USERNAME, config.CAMERA_PASSWORD)
    zones = []
    try:
        poses = []
        if args.here:
            poses.append(("current view", None))
        else:
            w = config.WATER_HOLE_POSITION
            poses.append(("water-hole WIDE", (w["pan"], w["tilt"], config.ZOOM_SEARCH_POSITION + config.ZOOM_POSITION_MARGIN)))
            poses.append(("water-hole ZOOMED", (w["pan"], w["tilt"], w["zoom"])))
        for label, pose in poses:
            if pose is not None:
                cam.goto_position(*pose)
                settle(cam, 8.0)
            real = cam.get_pose()
            frame = grab(uri)
            if frame is None or real is None:
                print("Could not read frame/pose for", label)
                continue
            poly = draw(frame, label)
            if poly:
                zones.append({"name": label.replace(" ", "_").lower(),
                              "pose": {"pan": real[0], "tilt": real[1], "zoom": real[2]},
                              "polygon": poly})
                print(f"saved zone '{label}' at pose {real}")
    finally:
        cam.close()
        cv2.destroyAllWindows()
    if zones:
        with open(config.WATER_ROI_FILE, "w", encoding="utf-8") as f:
            json.dump({"zones": zones}, f, indent=2)
        print("Wrote", config.WATER_ROI_FILE)
    else:
        print("Nothing saved.")


if __name__ == "__main__":
    main()
