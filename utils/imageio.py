"""JPEG writing that cannot be broken by other libraries.

PytorchWildlife imports the ``yolov5`` package, which REPLACES ``cv2.imwrite``
process-wide with a 2-argument version (no quality parameter).  After the AI is
loaded, every ``cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, q])`` raised
TypeError.  ``cv2.imencode`` is left alone, so encode + write ourselves (this also
handles non-ASCII paths on Windows).
"""

import os

import cv2


def write_jpeg(path, img, quality=92):
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1] or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except OSError:
        return False
