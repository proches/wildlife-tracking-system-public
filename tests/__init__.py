"""Test package.

Tests must NEVER touch the live system's state files.  On 17/10/2026 a simulated
camera (pan sign +1) wrote logs/ptz_capabilities.json, the real tracker then
loaded that sign and drove the real camera to the hotel side.  Every state file
is redirected to a throw-away directory before any test module is imported.
"""
import os
import tempfile

import config

_TMP = tempfile.mkdtemp(prefix="wildlife_tests_")
config.PTZ_CAPABILITIES_FILE = os.path.join(_TMP, "ptz_capabilities.json")
config.STATIC_FP_FILE = os.path.join(_TMP, "static_false_positives.json")
config.PTZ_TRACE_CSV = os.path.join(_TMP, "ptz_trace.csv")
config.WATER_ROI_FILE = os.path.join(_TMP, "water_roi.json")
config.PTZ_PRESETS_FILE = os.path.join(_TMP, "ptz_presets.json")
