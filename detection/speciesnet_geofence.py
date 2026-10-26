"""SpeciesNet's official country filter for already-detected animal crops."""
import logging

logger = logging.getLogger("wildlife_tracker.detection")


class SpeciesNetGeofence:
    def __init__(self, model_path, country):
        self.country = country
        self.ensemble = None
        if country is None:
            logger.info("SpeciesNet geographic filtering disabled")
            return
        from speciesnet.ensemble import SpeciesNetEnsemble
        from speciesnet.geofence_utils import geofence_animal_classification

        self.ensemble = SpeciesNetEnsemble(model_path, geofence=True)
        # Validate against countries actually represented in the bundled rules,
        # so a typo cannot silently filter against an unknown location.
        countries = {
            code
            for rules in self.ensemble.geofence_map.values()
            for rule in ("allow", "block")
            for code in (rules.get(rule) or {})
        }
        if country not in countries:
            raise ValueError(f"Unknown SpeciesNet geofence country: {country!r}")
        self._filter = geofence_animal_classification
        logger.info("SpeciesNet geographic filtering active: %s", country)

    def apply(self, classifications):
        labels = classifications.get("classes", [])
        scores = classifications.get("scores", [])
        if not labels or len(labels) != len(scores):
            return "unidentified", 0.0
        label, score = labels[0], float(scores[0])
        # Preserve the existing blank/human/vehicle veto and confidence rules.
        name = label.rsplit(";", 1)[-1].strip()
        if self.ensemble is None or name in {"blank", "human", "vehicle"}:
            return label, score
        label, score, source = self._filter(
            labels=labels, scores=scores, country=self.country,
            admin1_region=None, taxonomy_map=self.ensemble.taxonomy_map,
            geofence_map=self.ensemble.geofence_map, enable_geofence=True,
        )
        label = getattr(label, "value", label)
        if "geofence" in source:
            logger.debug("SpeciesNet %s: %s -> %s (%s)",
                         self.country, labels[0], label, source)
        return label, float(score)
