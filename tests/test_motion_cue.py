"""19/10: a moving warthog/baboons must stop the sweep even if the detectors miss them."""

import unittest

import cv2
import numpy as np

from utils.motion_cue import MotionCue

H, W = 1080, 1920


def scene(rng):
    small = rng.integers(60, 200, (140, 250), dtype=np.uint8)
    big = cv2.resize(small, (2500, 1400), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


class MotionCueTests(unittest.TestCase):
    def run_seq(self, n, pan_px_per_step=0.0, animal=None, wind=False, seed=0):
        rng = np.random.default_rng(seed)
        bg = scene(rng)
        cue = MotionCue()
        hits = []
        for i in range(n):
            ox = int(200 + i * pan_px_per_step)
            f = bg[100:100 + H, ox:ox + W].copy()
            if animal is not None:
                ax, ay = animal[0] + i * animal[1], animal[2]
                cv2.rectangle(f, (int(ax), ay), (int(ax) + 60, ay + 45), (20, 20, 20), -1)
            if wind:
                for _ in range(40):
                    x, y = rng.integers(0, W - 40), rng.integers(0, H - 40)
                    f[y:y + 30, x:x + 30] = rng.integers(0, 255)
            r = cue.update(f, i * 0.4, camera_moving=pan_px_per_step != 0.0)
            if r is not None:
                hits.append(r)
        return hits

    def test_camera_pan_alone_is_not_motion(self):
        self.assertEqual(self.run_seq(10, pan_px_per_step=12.0), [])

    def test_walking_animal_on_still_camera(self):
        hits = self.run_seq(8, animal=(800, 18, 600))
        self.assertTrue(hits)
        self.assertLess(abs(hits[-1][1] - 622), 80)

    def test_walking_animal_while_panning(self):
        hits = self.run_seq(10, pan_px_per_step=12.0, animal=(700, 25, 500))
        self.assertTrue(hits)

    def test_leaves_swaying_in_place_are_ignored(self):
        """19/10: the big tree at scan_6 was zoomed in on."""
        rng = np.random.default_rng(3)
        bg = scene(rng)
        cue = MotionCue()
        hits = []
        for i in range(12):
            f = bg[100:100 + H, 200:200 + W].copy()
            dx = 10 if i % 2 else -10                         # a branch swinging back and forth
            cv2.rectangle(f, (1500 + dx, 300), (1560 + dx, 350), (30, 120, 30), -1)
            r = cue.update(f, i * 0.4, camera_moving=False)
            if r is not None:
                hits.append(r)
        self.assertEqual(hits, [])

    def test_wind_moving_everything_is_ignored(self):
        self.assertEqual(self.run_seq(8, wind=True), [])


if __name__ == "__main__":
    unittest.main()
