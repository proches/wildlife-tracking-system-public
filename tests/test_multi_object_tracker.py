import unittest

from tracking.multi_object_tracker import MultiObjectTracker


class MultiObjectTrackerTests(unittest.TestCase):
    def test_persistent_id_survives_motion_and_species_becomes_stable(self):
        tracker = MultiObjectTracker()
        first = tracker.update([
            {"box": [100, 100, 180, 180], "class_id": 0, "det_confidence": 0.80,
             "species": "unidentified", "species_confidence": 0.0}
        ], 1000, 1000)
        tid = first[0]["track_id"]
        out = tracker.update([
            {"box": [112, 104, 192, 184], "class_id": 0, "det_confidence": 0.82,
             "species": "plains zebra", "species_confidence": 0.84}
        ], 1000, 1000)
        self.assertEqual(out[0]["track_id"], tid)
        self.assertEqual(out[0]["species"], "unidentified")
        out = tracker.update([
            {"box": [125, 108, 205, 188], "class_id": 0, "det_confidence": 0.84,
             "species": "plains zebra", "species_confidence": 0.88}
        ], 1000, 1000)
        self.assertEqual(out[0]["track_id"], tid)
        self.assertEqual(out[0]["species"], "plains zebra")
        self.assertGreater(out[0]["priority_score"], 0.0)

    def test_two_nearby_animals_keep_separate_ids(self):
        tracker = MultiObjectTracker()
        out = tracker.update([
            {"box": [100, 100, 180, 180], "class_id": 0, "det_confidence": 0.8,
             "species": "unidentified", "species_confidence": 0},
            {"box": [400, 100, 480, 180], "class_id": 0, "det_confidence": 0.9,
             "species": "unidentified", "species_confidence": 0},
        ], 1000, 1000)
        ids = {d["track_id"] for d in out}
        self.assertEqual(len(ids), 2)


if __name__ == "__main__":
    unittest.main()
