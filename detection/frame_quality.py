"""Cheap frame-quality signals used to tune inference without changing models."""

import cv2
import numpy as np


class FrameQuality:
    def analyze(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0
        contrast = float(np.std(gray)) / 128.0
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        h, w = gray.shape[:2]
        return {
            "brightness": brightness,
            "contrast": min(2.0, contrast),
            "blur": blur,
            "width": int(w),
            "height": int(h),
            "low_light": brightness < 0.20,
            "very_dark": brightness < 0.10,
            "blurred": blur < 35.0,
        }
