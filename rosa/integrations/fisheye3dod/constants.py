"""Fisheye3DOD benchmark constants."""

from __future__ import annotations

import os
from pathlib import Path

# Checkout of https://github.com/weiyangdaren/Fisheye3DOD (evaluator, OCam model).
DEFAULT_OFFICIAL_REPO = Path(
    os.environ.get(
        "FISHEYE3DOD_OFFICIAL_ROOT",
        Path(__file__).resolve().parents[3] / "external" / "fisheye3dod",
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

# Evaluation range of each class (m) and the official point-cloud range.
OFFICIAL_CLASS_RANGES = {
    "Car": 48,
    "Van": 48,
    "Truck": 48,
    "Bus": 48,
    "Pedestrian": 36,
    "Cyclist": 42,
}
OFFICIAL_POINT_CLOUD_RANGE = (-48.0, -48.0, -5.0, 48.0, 48.0, 5.0)
