"""Optional adapters for the released Fisheye3DOD schema.

This package initializer is dependency-light.  MMDetection3D registrations
live in ``dataset``, ``model``, and ``metric`` and must be imported explicitly
by the training config.
"""

from .constants import (
    CAMERAS,
    CLASSES,
    DEFAULT_SEED,
    OFFICIAL_AP_DISTANCES,
    OFFICIAL_TP_DISTANCE,
)

__all__ = [
    "CAMERAS",
    "CLASSES",
    "DEFAULT_SEED",
    "OFFICIAL_AP_DISTANCES",
    "OFFICIAL_TP_DISTANCE",
]
