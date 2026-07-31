"""ROSA: calibrated multiview refinement for native fisheye 3D detection."""

from .refiner import (
    ObservationEncoding,
    PointwiseROSA,
    ROSAConfig,
    ROSAOutput,
    ROSARefiner,
)
from .rfl import SphericalRayFeatureLifting

__version__ = "0.1.0"

__all__ = [
    "ObservationEncoding",
    "PointwiseROSA",
    "ROSAConfig",
    "ROSAOutput",
    "ROSARefiner",
    "SphericalRayFeatureLifting",
]
