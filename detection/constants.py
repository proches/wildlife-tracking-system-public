"""Shared detection constants (no heavy imports, safe to import anywhere)."""

CLASS_ANIMAL = 0
CLASS_PERSON = 1
CLASS_VEHICLE = 2
CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}

# Labels that must never be tracked or treated as wildlife in the park.
NON_ANIMAL_LABELS = {
    "blank", "vehicle", "human", "person", "people", "car", "truck",
    "bus", "motorcycle", "bicycle", "boat", "airplane", "train",
}

# Labels a classifier may output when it is NOT sure what the crop is.  They
# never count as "verified wildlife".
UNVERIFIED_LABELS = {"", "unidentified", "other", "unknown", "animal", "mammal", "bird"}

# Livestock / pets: not expected inside the national park.  On this camera
# SpeciesNet said "domestic horse 0.69" and "domestic cattle 0.66" on tiny far
# acacias at the horizon - such a label alone never verifies an animal.
DOMESTIC_LABELS = {
    "domestic horse", "domestic cattle", "domestic dog", "domestic cat", "domestic goat",
    "domestic sheep", "domestic donkey", "domestic pig", "domestic chicken", "donkey",
    "horse", "cattle", "cow", "goat", "sheep", "dog", "cat",
}
