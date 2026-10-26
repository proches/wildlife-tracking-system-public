"""Integration checks against the downloaded SpeciesNet geofence resources."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock

MODEL = Path(os.environ.get('SPECIESNET_TEST_MODEL', str(
    Path.home() / '.cache/kagglehub/models/google/speciesnet/pyTorch/v4.0.3a/1')))


@unittest.skipUnless(importlib.util.find_spec('speciesnet') and (MODEL / 'info.json').exists(),
                     'Requires SpeciesNet and downloaded model resources')
class TanzaniaGeofenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from detection.speciesnet_geofence import SpeciesNetGeofence
        cls.filter = SpeciesNetGeofence(str(MODEL), 'TZA')
        import json
        info = json.loads((MODEL / 'info.json').read_text())
        cls.labels = {
            label.rsplit(';', 1)[-1]: label
            for label in (MODEL / info['classifier_labels']).read_text().splitlines()
        }

    def result(self, name, score=0.9):
        return {'classes': [self.labels[name]], 'scores': [score]}

    def test_tanzania_species_kept(self):
        for name in ('plains zebra', 'african buffalo', 'lion'):
            with self.subTest(name=name):
                self.assertEqual(self.filter.apply(self.result(name)), (self.labels[name], 0.9))

    def test_blocked_species_rolls_up_without_renaming_to_local_species(self):
        label, score = self.filter.apply(self.result('white-tailed deer'))
        self.assertNotEqual(label, self.labels['white-tailed deer'])
        self.assertTrue(label.endswith(';cervidae family'))
        self.assertEqual(score, 0.9)

    def test_non_animal_veto_labels_unchanged(self):
        for name in ('blank', 'human', 'vehicle'):
            self.assertEqual(self.filter.apply(self.result(name)), (self.labels[name], 0.9))

    def test_disabled_retains_raw_species(self):
        from detection.speciesnet_geofence import SpeciesNetGeofence
        disabled = SpeciesNetGeofence(None, None)
        self.assertEqual(disabled.apply(self.result('white-tailed deer')),
                         (self.labels['white-tailed deer'], 0.9))

    def test_unknown_country_rejected(self):
        from detection.speciesnet_geofence import SpeciesNetGeofence
        with self.assertRaises(ValueError):
            SpeciesNetGeofence(str(MODEL), 'ZZZ')

    def test_missing_predictions_unidentified(self):
        self.assertEqual(self.filter.apply({}), ('unidentified', 0.0))
        self.assertEqual(self.filter.apply({'classes': ['x'], 'scores': []}),
                         ('unidentified', 0.0))

    def test_live_backend_uses_filter_and_keeps_threshold(self):
        import numpy as np
        import PIL.Image
        from detection.pipeline import _SpeciesNetBackend, WildlifePipeline
        backend = _SpeciesNetBackend.__new__(_SpeciesNetBackend)
        backend._Image = PIL.Image
        backend.classifier = Mock()
        backend.classifier.predict.return_value = {
            'classifications': self.result('white-tailed deer', 0.4)}
        backend.geofence = self.filter
        backend.threshold = 0.5
        crop = np.zeros((32, 32, 3), dtype=np.uint8)
        self.assertEqual(backend.classify(crop), ('cervidae family', 0.4))
        pipeline = WildlifePipeline.__new__(WildlifePipeline)
        pipeline.backend = backend
        pipeline._warned_classify = False
        self.assertEqual(pipeline._classify(crop), ('unidentified', 0.4))
