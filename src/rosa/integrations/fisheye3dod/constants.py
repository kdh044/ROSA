"""Immutable constants for the released Fisheye3DOD protocol."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OFFICIAL_REPO = Path(
    os.environ.get(
        "FISHEYE3DOD_OFFICIAL_ROOT",
        PROJECT_ROOT / "external" / "fisheye3dod",
    )
).resolve()

CLASSES = ("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist")
CAMERAS = (
    "fisheye_camera_front",
    "fisheye_camera_left",
    "fisheye_camera_right",
    "fisheye_camera_rear",
)
CAMERA_IMAGE_SIZE = (800, 800)
CAMERA_FOV_DEGREES = 220.0

OFFICIAL_AP_DISTANCES = (0.5, 1.0, 2.0, 4.0)
OFFICIAL_TP_DISTANCE = 2.0
OFFICIAL_CLASS_RANGES = {
    "Car": 48,
    "Van": 48,
    "Truck": 48,
    "Bus": 48,
    "Pedestrian": 36,
    "Cyclist": 42,
}
OFFICIAL_POINT_CLOUD_RANGE = (
    -48.0,
    -48.0,
    -5.0,
    48.0,
    48.0,
    5.0,
)

# The released evaluator declares 50 but does not truncate its predictions.
# The paired detector arms therefore retain the common 300-query budget.
RELEASED_DECLARED_MAX_BOXES = 50
PAIRED_QUERY_BUDGET = 300

DEFAULT_SEED = 3407
EXPECTED_TRAIN_FRAMES = 10_080
EXPECTED_VAL_FRAMES = 4_464
EXPECTED_SCENES = 144
EXPECTED_TRAIN_FRAMES_PER_SCENE = 70
EXPECTED_VAL_FRAMES_PER_SCENE = 31


__all__ = [
    "CAMERAS",
    "CAMERA_FOV_DEGREES",
    "CAMERA_IMAGE_SIZE",
    "CLASSES",
    "DEFAULT_OFFICIAL_REPO",
    "DEFAULT_SEED",
    "EXPECTED_SCENES",
    "EXPECTED_TRAIN_FRAMES",
    "EXPECTED_TRAIN_FRAMES_PER_SCENE",
    "EXPECTED_VAL_FRAMES",
    "EXPECTED_VAL_FRAMES_PER_SCENE",
    "OFFICIAL_AP_DISTANCES",
    "OFFICIAL_CLASS_RANGES",
    "OFFICIAL_POINT_CLOUD_RANGE",
    "OFFICIAL_TP_DISTANCE",
    "PAIRED_QUERY_BUDGET",
    "RELEASED_DECLARED_MAX_BOXES",
]
