"""ROSA: Ray-aware Observation Selection and Aggregation for fisheye 3D detection."""

from .projection import Projector, OCamProjection
from .decoder import (
    ROSADecoderConfig,
    ROSADecoderOutput,
    BoxRefinement,
    BoxState,
    ROSADecoder,
)
from .losses import RayAuxiliaryLoss, RayAuxiliaryLossConfig
from .rfl import SphericalRayFeatureLifting

__version__ = "1.0.0"

__all__ = [
    "RayAuxiliaryLoss",
    "ROSADecoderConfig",
    "ROSADecoderOutput",
    "BoxRefinement",
    "BoxState",
    "ROSADecoder",
    "Projector",
    "OCamProjection",
    "RayAuxiliaryLossConfig",
    "SphericalRayFeatureLifting",
]
