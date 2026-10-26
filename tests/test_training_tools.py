import os
import tempfile
import unittest

import cv2
import numpy as np

from training.build_dataset import split_by_time, tile_labels, tile_positions, emit_tiles, image_time
from training.eval_by_size import evaluate
from training.labels_io import read_labels, write_labels, xyxy_to_yolo, yolo_to_xyxy


class TilingTests(unittest.TestCase):
    def test_tile_positions_cover_whole_axis_with_overlap(self):
        pos = tile_positions(1920, 640, 0.2)
        self.assertEqual(pos[0], 0)
        self.assertEqual(pos[-1], 1920 - 640)
        self.assertTrue(all(b - a <= 512 for a, b in zip(pos, pos[1:])))
        self.assertEqual(tile_positions(500, 640, 0.2), [0])

    def test_small_bird_keeps_native_pixel_size_inside_tile(self):
        bird = [(1, 1000, 500, 1014, 512)]          # 14 x 12 px bird
        out = tile_labels(bird, 800, 300, 640, 640)
        self.assertEqual(len(out), 1)
        cls, cx, cy, w, h = out[0]
        self.assertAlmostEqual(w * 640, 14, places=3)
        self.assertAlmostEqual(h * 640, 12, places=3)

    def test_box_mostly_outside_tile_is_dropped_partially_inside_is_clipped(self):
        self.assertEqual(tile_labels([(0, 600, 100, 700, 200)], 0, 0, 640, 640, 0.5), [])   # 40% visible
        out = tile_labels([(0, 560, 100, 660, 200)], 0, 0, 640, 640, 0.5)                  # 80% visible
        self.assertEqual(len(out), 1)
        x1, y1, x2, y2 = yolo_to_xyxy(out[0][1:], 640, 640)
        self.assertAlmostEqual(x2, 640, places=3)

    def test_emit_tiles_writes_images_and_labels(self):
        d = tempfile.mkdtemp()
        img = np.zeros((1080, 1920, 3), np.uint8)
        p = os.path.join(d, "a.jpg")
        cv2.imwrite(p, img)
        oi, ol = os.path.join(d, "i"), os.path.join(d, "l")
        os.makedirs(oi)
        os.makedirs(ol)
        labels = [(0,) + xyxy_to_yolo((1000, 500, 1060, 540), 1920, 1080)]
        import random
        n = emit_tiles(p, labels, oi, ol, "a", 640, 0.2, 0.0, random.Random(1))
        self.assertGreaterEqual(n, 1)
        with_boxes = [f for f in os.listdir(ol) if read_labels(os.path.join(ol, f))]
        self.assertTrue(with_boxes)


class SplitTests(unittest.TestCase):
    def test_neighbouring_frames_never_straddle_train_and_val(self):
        items = []
        t = 1_700_000_000.0
        for block in range(10):                       # 10 sightings, 20 frames each, 5 s apart
            for k in range(20):
                items.append((f"b{block}_f{k}", t + block * 3600 + k * 5))
        train, val = split_by_time(items, 0.2)
        train_blocks = {p.split("_")[0] for p in train}
        val_blocks = {p.split("_")[0] for p in val}
        self.assertFalse(train_blocks & val_blocks)
        self.assertTrue(val)

    def test_filename_timestamp_is_parsed(self):
        from datetime import datetime
        expected = datetime(2026, 10, 8, 11, 11, 25).timestamp()
        self.assertEqual(image_time("20261008T111125_445925Z_unidentified.jpg"), expected)


class LabelIOTests(unittest.TestCase):
    def test_round_trip(self):
        p = os.path.join(tempfile.mkdtemp(), "x.txt")
        write_labels(p, [(1, 0.5, 0.5, 0.1, 0.2)])
        self.assertEqual(read_labels(p), [(1, 0.5, 0.5, 0.1, 0.2)])


class EvalBySizeTests(unittest.TestCase):
    def test_recall_is_reported_per_size_bucket(self):
        d = tempfile.mkdtemp()
        gt, pr = os.path.join(d, "gt"), os.path.join(d, "pr")
        os.makedirs(gt)
        os.makedirs(pr)
        W, H = 1920, 1080
        big = xyxy_to_yolo((100, 100, 300, 250), W, H)      # large, found
        bird = xyxy_to_yolo((900, 500, 912, 510), W, H)      # tiny, missed
        write_labels(os.path.join(gt, "f.txt"), [(0,) + big, (1,) + bird])
        with open(os.path.join(pr, "f.txt"), "w") as f:
            f.write("0 %f %f %f %f 0.9\n" % big)
            f.write("0 0.1 0.9 0.02 0.02 0.9\n")             # a false positive
        stats, tp, fp = evaluate(gt, pr, W, H)
        self.assertEqual(stats["tiny  (<16px)"], [1, 0])
        self.assertEqual(stats["large (>=96px)"], [1, 1])
        self.assertEqual((tp, fp), (1, 1))


def _frame(folder, name, boxes=(), size=(720, 1280)):
    """A grey frame with YOLO labels (boxes in pixels); returns its path."""
    os.makedirs(folder, exist_ok=True)
    h, w = size
    path = os.path.join(folder, name)
    cv2.imwrite(path, np.full((h, w, 3), 90, np.uint8))
    write_labels(os.path.splitext(path)[0] + ".txt", [(0,) + xyxy_to_yolo(b, w, h) for b in boxes])
    return path


def _assemble(*args):
    import sys
    from unittest import mock
    from training import assemble_training_set as asm
    with mock.patch.object(sys, "argv", ["assemble"] + list(args)), mock.patch("builtins.print"):
        asm.main()


class AssembleRound3Tests(unittest.TestCase):
    """23/10: the hand-checked crops win over the teacher, empty tiles are thinned for
    training only, and every site folder gives its own share of validation."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.out = os.path.join(self.d, "ds")
        self.no_lila = os.path.join(self.d, "no_lila")

    def _names(self, kind, split):
        return sorted(os.listdir(os.path.join(self.out, kind, split)))

    def test_hand_checked_crop_replaces_the_teacher_frame_of_the_same_second(self):
        site, ver = os.path.join(self.d, "site"), os.path.join(self.d, "ver")
        # the teacher said "nothing" where the operator saw primates (field case 20/10 12:21:50)
        _frame(site, "20261020T122150_283512Z_candidate.jpg")
        for k in range(12):                            # other frames, an hour apart
            _frame(site, f"20261021T{k + 6:02d}0000_000000Z_negative.jpg")
        _frame(ver, "20261020T122150_283512Z_verified.jpg", [(300, 300, 340, 330)], size=(640, 640))
        _assemble("--out", self.out, "--lila", self.no_lila, "--site", site, "--verified", ver,
                  "--site-repeat", "1", "--empty-keep", "1.0", "--val", "0.12")
        everything = self._names("images", "train") + self._names("images", "val")
        self.assertFalse([n for n in everything if "122150_283512Z_candidate" in n])
        crops = [n for n in self._names("labels", "train") if "_verified" in n]
        self.assertEqual(len(crops), 3)                # --verified-repeat 3, all in training
        for n in crops:
            self.assertTrue(read_labels(os.path.join(self.out, "labels", "train", n)))
        self.assertFalse([n for n in self._names("images", "val") if "_verified" in n])

    def test_empty_tiles_thinned_for_training_but_all_kept_for_validation(self):
        site = os.path.join(self.d, "site")
        for k in range(20):                            # 20 empty frames an hour apart: 20 time blocks
            _frame(site, f"20261021T{k:02d}0000_000000Z_negative.jpg")
        _assemble("--out", self.out, "--lila", self.no_lila, "--site", site,
                  "--site-repeat", "1", "--empty-keep", "0.0", "--val", "0.12")
        self.assertEqual(self._names("images", "train"), [])
        val_frames = {n.split("_r0_")[0] for n in self._names("images", "val")}
        self.assertGreaterEqual(len(val_frames), 2)
        self.assertEqual(len(self._names("images", "val")), 6 * len(val_frames))   # 3 x 2 tiles each

    def test_every_site_folder_gives_its_own_validation_share(self):
        a, b = os.path.join(self.d, "a"), os.path.join(self.d, "b")
        for k in range(10):
            _frame(a, f"20261018T{k:02d}0000_000000Z_negative.jpg", [(100, 100, 160, 150)])
            _frame(b, f"20261022T{k:02d}0000_000000Z_negative.jpg", [(100, 100, 160, 150)])
        _assemble("--out", self.out, "--lila", self.no_lila, "--site", a, b,
                  "--site-repeat", "1", "--val", "0.12")
        val = self._names("images", "val")
        self.assertTrue([n for n in val if "20261018T" in n])
        self.assertTrue([n for n in val if "20261022T" in n])

    def test_the_same_frame_in_two_folders_is_used_once(self):
        a, b = os.path.join(self.d, "a"), os.path.join(self.d, "b")
        _frame(a, "20261019T075443_missed.jpg", [(100, 100, 160, 150)])
        _frame(b, "20261019T075443_missed.jpg", [(100, 100, 160, 150)])
        _assemble("--out", self.out, "--lila", self.no_lila, "--site", a, b, "--site-repeat", "1")
        names = self._names("images", "train") + self._names("images", "val")
        self.assertEqual(len([n for n in names if "075443_missed_r0_x0_y0" in n]), 1)


class Round4Tests(unittest.TestCase):
    def test_the_epoch_with_the_best_map50_is_checked_not_the_last(self):
        """Round 3: mAP50 0.759 after epoch 1, 0.736 at the end - the end went to the check."""
        from training.round4 import best_epoch_weights
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "weights"))
        rows = [(1, 0.759), (2, 0.717), (3, 0.718), (4, 0.736)]
        with open(os.path.join(d, "results.csv"), "w", encoding="utf-8") as f:
            f.write("                  epoch,                   time,      metrics/mAP50(B)\n")
            for e, m in rows:
                f.write(f"{e},{100 * e},{m}\n")
        for e, _ in rows:                                    # the trainer names them epoch0 .. epoch3
            open(os.path.join(d, "weights", f"epoch{e - 1}.pt"), "wb").close()
        open(os.path.join(d, "weights", "best.pt"), "wb").close()
        w, epoch, m = best_epoch_weights(d)
        self.assertEqual((os.path.basename(w), epoch, m), ("epoch0.pt", 1, 0.759))

    def test_without_epoch_files_best_pt_is_used(self):
        from training.round4 import best_epoch_weights
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "weights"))
        open(os.path.join(d, "weights", "best.pt"), "wb").close()
        self.assertEqual(os.path.basename(best_epoch_weights(d)[0]), "best.pt")
        self.assertIsNone(best_epoch_weights(tempfile.mkdtemp()))

    def test_the_watchdog_leaves_round_4_alone_at_night(self):
        from tools.tracker_watchdog import plan
        r4 = (5120, r'"C:\x\python.exe" training\round4.py --deadline 05:30', 900.0)
        self.assertEqual(plan([r4]), ("start", [5120]))      # main() leaves it alone before 05:30

    def test_the_fine_tune_preset_is_gentle(self):
        from training.train_custom_yolo import PRESETS
        p = PRESETS["cpu_finetune"]
        self.assertEqual(p["optimizer"], "AdamW")
        self.assertLessEqual(p["lr0"], 0.0005)               # round 3 ran at 0.001667
        self.assertLessEqual(p["warmup_epochs"], 1.0)        # round 3: 3 of 7 epochs
        self.assertEqual(p["save_period"], 1)


class RestoreTests(unittest.TestCase):
    def test_hand_checked_crops_survive_a_pc_without_logs(self):
        """25/10: the project goes to GitHub without logs/ (videos, review frames).  On a
        restored PC build_site_verified.py found no review frame and crashed, so every
        training round would stop; the crops it made are in the repository."""
        import json
        import subprocess
        import sys
        d = tempfile.mkdtemp()
        out = os.path.join(d, "site3v")
        crop = _frame(out, "20261020T122150_283512Z_verified.jpg", [(300, 300, 340, 330)], size=(640, 640))
        lst = os.path.join(d, "verified.json")
        with open(lst, "w", encoding="utf-8") as f:
            json.dump({"items": [{"review": "logs/review/20261020T122150_283512Z_candidate.json_GONE",
                                  "verdict": "animal"},
                                 {"review": "logs/review/20991231T000000_000000Z_candidate.json_GONE",
                                  "verdict": "no"}]}, f)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, os.path.join(root, "training", "build_site_verified.py"),
                            "--list", lst, "--out", out], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("'kept': 1", r.stdout)
        self.assertIn("'skipped': 1", r.stdout)
        self.assertTrue(read_labels(os.path.splitext(crop)[0] + ".txt"))


class LilaDownloadTests(unittest.TestCase):
    def test_no_download_starts_after_the_time_limit(self):
        """25/10: the lodge link kept dropping; the night's download must stop at its limit
        (and the first version of this check used time without importing it)."""
        import time as _time
        from training.build_lila_serengeti import fetch
        d = tempfile.mkdtemp()
        item = ("S1/B04/B04_R1/S1_B04_R1_PICT0001", "S1/B04/B04_R1/S1_B04_R1_PICT0001.JPG", 0,
                [[10, 10, 50, 50]], 2048, 1536)
        self.assertFalse(fetch(item, d, d, 960, deadline=_time.time() - 1))
        self.assertEqual(os.listdir(d), [])
        open(os.path.join(d, "S1_B04_B04_R1_S1_B04_R1_PICT0001.jpg"), "wb").close()
        self.assertTrue(fetch(item, d, d, 960, deadline=_time.time() - 1))   # already here


class ConsoleTests(unittest.TestCase):
    def test_quick_edit_switch_never_raises_without_a_console(self):
        from utils.logger import disable_console_quick_edit
        self.assertIn(disable_console_quick_edit(), (True, False))


class Round3PlanTests(unittest.TestCase):
    def test_clip_frames_are_thinned_and_unlabelled_frames_left_out(self):
        from unittest import mock
        from training import round3
        d = tempfile.mkdtemp()
        src, dst = os.path.join(d, "src"), os.path.join(d, "dst")
        for k in range(12):
            _frame(src, f"20261022T1000{k:02d}_clip{k:04d}.jpg", [(10, 10, 50, 50)], size=(64, 64))
        _frame(src, "20261022T110000_000001Z_candidate.jpg", size=(64, 64))
        unsure = _frame(src, "20261022T120000_000002Z_candidate.jpg", size=(64, 64))
        os.remove(os.path.splitext(unsure)[0] + ".txt")       # the teacher was unsure: no label file
        with mock.patch.object(round3, "LOG", os.path.join(d, "round3.log")), mock.patch("builtins.print"):
            kept = round3.subsample_site3h(src, dst, clip_every=6)
        jpgs = sorted(f for f in os.listdir(dst) if f.endswith(".jpg"))
        self.assertEqual(kept, 3)                             # the labelled review frame + clips 6 and 12
        self.assertEqual(len(jpgs), 3)
        self.assertNotIn("20261022T120000_000002Z_candidate.jpg", jpgs)
        for f in jpgs:
            self.assertTrue(os.path.exists(os.path.join(dst, f[:-4] + ".txt")))

    def test_training_stops_an_hour_before_the_deadline(self):
        import time as _time
        from training.round3 import training_hours
        start = _time.mktime((2026, 10, 23, 20, 40, 0, 0, 0, -1))
        self.assertAlmostEqual(training_hours("05:30", 12.0, now=start), 8 + 50 / 60 - 1.0, places=3)
        self.assertAlmostEqual(training_hours("05:30", 6.0, now=start), 6.0, places=3)
        late = _time.mktime((2026, 10, 24, 4, 50, 0, 0, 0, -1))
        self.assertLess(training_hours("05:30", 12.0, now=late), 1.0)     # -> no training tonight


class WatchdogTests(unittest.TestCase):
    TRACKER = (4508, r'".venv313\Scripts\python.exe"  main.py')
    LSP = (7864, r'"c:\Users\Melia Serengeti User\.vscode\extensions\ms-python.flake8\bundled\tool\lsp_server.py"')
    ROUND3 = (1248, r'"C:\Users\Melia Serengeti User\Desktop\wildlife_tracker\.venv313\Scripts\python.exe" '
                    r'training\round3.py --reuse --deadline 05:30')
    TRAIN = (2348, r'"C:\x\python.exe" training/train_custom_yolo.py --data training_data/dataset3/data.yaml')
    FINAL = (3001, r'"C:\x\python.exe" training/finalize_model.py --weights runs\best.pt')

    def test_tracker_running_means_nothing_to_do(self):
        from tools.tracker_watchdog import plan
        self.assertEqual(plan([self.LSP, self.TRACKER, self.ROUND3]), ("ok", []))

    def test_an_overrunning_night_job_is_stopped_before_the_tracker_starts(self):
        from tools.tracker_watchdog import plan
        self.assertEqual(plan([self.LSP, self.ROUND3, self.TRAIN]), ("start", [1248, 2348]))

    def test_a_running_model_check_is_waited_for(self):
        from tools.tracker_watchdog import plan
        self.assertEqual(plan([self.ROUND3, self.FINAL]), ("wait", [3001]))

    def test_nothing_running_starts_the_tracker_and_editor_helpers_are_left_alone(self):
        from tools.tracker_watchdog import plan
        self.assertEqual(plan([self.LSP]), ("start", []))
        self.assertEqual(plan([]), ("start", []))

    def test_a_tracker_without_heartbeat_is_frozen_but_a_starting_one_is_not(self):
        """23/10 13:45-14:43: the tracker froze for an hour with no error (control loop and
        log stopped in the same second); ptz_trace.csv gets a line every second."""
        from tools.tracker_watchdog import frozen
        now = 100_000.0
        running = [(4508, self.TRACKER[1], 3600.0), (7864, self.LSP[1], 9000.0)]
        self.assertTrue(frozen(running, now - 58 * 60, now))           # the 23/10 case
        self.assertFalse(frozen(running, now - 2, now))                # healthy: a line 2 s ago
        starting = [(4508, self.TRACKER[1], 90.0)]                      # loading its models
        self.assertFalse(frozen(starting, now - 10 * 3600, now))       # old file from before it
        self.assertFalse(frozen([(7864, self.LSP[1], 9000.0)], now - 3600, now))   # no tracker at all

    def test_night_jobs_are_stopped_only_after_the_training_deadline(self):
        import time as _time
        from tools.tracker_watchdog import morning

        def at(h, m):
            return _time.localtime(_time.mktime((2026, 10, 24, h, m, 0, 0, 0, -1)))
        self.assertFalse(morning(at(20, 35)))      # training just started
        self.assertFalse(morning(at(5, 5)))        # before the 05:30 deadline
        self.assertTrue(morning(at(5, 35)))
        self.assertTrue(morning(at(6, 0)))
        self.assertFalse(morning(at(12, 5)))

    def test_plan_reads_processes_with_their_age(self):
        from tools.tracker_watchdog import plan
        self.assertEqual(plan([(1248, self.ROUND3[1], 600.0), (2348, self.TRAIN[1], 500.0)]),
                         ("start", [1248, 2348]))


if __name__ == "__main__":
    unittest.main()
