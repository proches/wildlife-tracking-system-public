"""Optional low-latency YouTube RTMP publisher for annotated video."""

import logging
import os
import subprocess

import cv2

import config

logger = logging.getLogger("wildlife_tracker.youtube")


class RTMPPublisher:
    def __init__(self):
        self.proc = None
        self.width = config.YOUTUBE_WIDTH
        self.height = config.YOUTUBE_HEIGHT
        self.enabled = config.YOUTUBE_ENABLED and bool(config.YOUTUBE_STREAM_KEY)
        if self.enabled:
            self.url = f"{config.YOUTUBE_RTMP_BASE.rstrip('/')}/{config.YOUTUBE_STREAM_KEY}"

    def start(self):
        if not self.enabled:
            return False
        cmd = [
            config.FFMPEG_BIN, "-loglevel", "warning",
            "-use_wallclock_as_timestamps", "1",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-i", "-",
            "-r", str(config.YOUTUBE_FPS),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-b:v", config.YOUTUBE_BITRATE,
            "-maxrate", config.YOUTUBE_BITRATE, "-bufsize", "9000k",
            "-g", str(max(2, int(config.YOUTUBE_FPS * 2))),
            "-f", "flv", self.url,
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception as exc:
            logger.warning("YouTube RTMP publisher unavailable: %s", exc)
            self.proc = None
            return False
        logger.info("YouTube annotated publisher started")
        return True

    def publish(self, frame):
        if self.proc is None or self.proc.stdin is None:
            return False
        resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        try:
            self.proc.stdin.write(resized.tobytes())
            return True
        except (BrokenPipeError, OSError):
            logger.warning("YouTube RTMP pipe closed")
            self.stop()
            return False

    def stop(self):
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None
