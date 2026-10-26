"""Fast keyboard label reviewer (OpenCV window).

    python training/review_labels.py logs/review          # frames saved by the running system
    python training/review_labels.py dataset_raw/images

Every image is shown with its draft boxes (green = animal, orange = bird).

    SPACE / RIGHT  save + next        b / LEFT  save + previous
    a   draw a new box with the mouse (then ENTER); uses the current class
    d   delete the LAST box           c   clear all boxes  (= "nothing here": a HARD NEGATIVE - very valuable!)
    f   flip class of the last box (animal <-> bird)
    1 / 2  set current class: 1 animal, 2 bird
    z   zoom in on the last box (check species / tiny birds)      q  save + quit

Images with an empty .txt are kept as hard negatives (tree trunks, glare, text
overlays...).  Images you never touch keep their draft boxes, so only *wrong*
drafts need fixing.  A '.reviewed' marker is written so build_dataset.py only
uses images you have actually looked at.
"""

import os
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.labels_io import CLASSES, label_path, list_images, read_labels, write_labels, xyxy_to_yolo, yolo_to_xyxy

COLORS = [(0, 200, 0), (0, 165, 255)]


def marker(img_path):
    return os.path.splitext(img_path)[0] + ".reviewed"


def render(frame, boxes, cur_cls, idx, total, name):
    h, w = frame.shape[:2]
    view = frame.copy()
    for i, (cls, cx, cy, bw, bh) in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in yolo_to_xyxy((cx, cy, bw, bh), w, h)]
        cv2.rectangle(view, (x1, y1), (x2, y2), COLORS[cls % 2], 2)
        cv2.putText(view, f"{i}:{CLASSES[cls % len(CLASSES)]}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls % 2], 2)
    s = 1400.0 / w
    view = cv2.resize(view, None, fx=s, fy=s)
    cv2.rectangle(view, (0, 0), (view.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(view, f"[{idx + 1}/{total}] {name} | boxes={len(boxes)} | class={CLASSES[cur_cls]} | "
                      "a=add d=del c=clear f=flip z=zoom SPACE=next q=quit",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return view


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    images = list_images(sys.argv[1])
    if not images:
        print("No images found")
        return
    i, cur_cls = 0, 0
    while 0 <= i < len(images):
        path = images[i]
        frame = cv2.imread(path)
        if frame is None:
            i += 1
            continue
        h, w = frame.shape[:2]
        boxes = read_labels(label_path(path))
        while True:
            cv2.imshow("review", render(frame, boxes, cur_cls, i, len(images), os.path.basename(path)))
            k = cv2.waitKey(0) & 0xFF
            if k in (ord(" "), 83, ord("n")):
                step = 1
                break
            if k in (ord("b"), 81):
                step = -1
                break
            if k == ord("q"):
                step = 0
                break
            if k == ord("1"):
                cur_cls = 0
            elif k == ord("2"):
                cur_cls = 1
            elif k == ord("d") and boxes:
                boxes.pop()
            elif k == ord("c"):
                boxes = []
            elif k == ord("f") and boxes:
                c, *rest = boxes[-1]
                boxes[-1] = ((c + 1) % len(CLASSES), *rest)
            elif k == ord("z") and boxes:
                x1, y1, x2, y2 = [int(v) for v in yolo_to_xyxy(boxes[-1][1:], w, h)]
                pad = max(40, int(0.6 * max(x2 - x1, y2 - y1)))
                crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                if crop.size:
                    cv2.imshow("zoom", cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
                    cv2.waitKey(0)
                    cv2.destroyWindow("zoom")
            elif k == ord("a"):
                r = cv2.selectROI("review", render(frame, boxes, cur_cls, i, len(images), "draw box, ENTER"), False)
                if r[2] > 3 and r[3] > 3:
                    s = w / 1400.0
                    box = (r[0] * s, r[1] * s, (r[0] + r[2]) * s, (r[1] + r[3]) * s)
                    boxes.append((cur_cls,) + xyxy_to_yolo(box, w, h))
        write_labels(label_path(path), boxes)
        open(marker(path), "w").close()
        if step == 0:
            break
        i += step
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
