"""FishBEVOD/DAPETR-compatible KITTI-360 integration."""

from .backbone import register_fishbevod_vovnet
from .constants import CAMERAS, CLASSES
from .dataset import (
    DAPETRGlobalRotScale,
    DAPETRImageAugment,
    KITTI360MixedCameraDataset,
    KITTI360ObjectRangeFilter,
    LoadKITTI360MixedCameraImages,
)
from .metric import KITTI360DAPETRMetric

__all__ = [
    "CAMERAS",
    "CLASSES",
    "DAPETRGlobalRotScale",
    "DAPETRImageAugment",
    "KITTI360MixedCameraDataset",
    "KITTI360DAPETRMetric",
    "KITTI360ObjectRangeFilter",
    "LoadKITTI360MixedCameraImages",
    "register_fishbevod_vovnet",
]
