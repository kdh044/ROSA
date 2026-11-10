"""ROSA: Ray-aware Observation Selection and Aggregation for fisheye 3D detection."""

from .projection import NativeFisheyeProjector, OCamFisheyeProjection
from .decoder import (
    HybridBoxDecoderConfig,
    HybridBoxDecoderOutput,
    HybridBoxRefinement,
    HybridBoxState,
    HybridSparseBoxDecoder,
)
from .losses import FisheyeHybridRayAuxiliaryLoss, RayAuxiliaryLossConfig
from .rfl import SphericalRayFeatureLifting

__version__ = "1.0.0"

__all__ = [
    "FisheyeHybridRayAuxiliaryLoss",
    "HybridBoxDecoderConfig",
    "HybridBoxDecoderOutput",
    "HybridBoxRefinement",
    "HybridBoxState",
    "HybridSparseBoxDecoder",
    "NativeFisheyeProjector",
    "OCamFisheyeProjection",
    "RayAuxiliaryLossConfig",
    "SphericalRayFeatureLifting",
]
