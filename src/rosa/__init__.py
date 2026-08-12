"""ROSA: calibrated multiview refinement for native fisheye 3D detection."""

from .refiner import (
    ObservationEncoding,
    PointwiseROSA,
    ROSAConfig,
    ROSAOutput,
    ROSARefiner,
)
from .rfl import SphericalRayFeatureLifting
from .fisheye_decoder import (
    FisheyeRayDecoder,
    FisheyeRayDecoderConfig,
    FisheyeRayDecoderOutput,
    SphericalRayQueryBank,
)
from .fisheye_box_decoder import (
    FisheyeBoxDecoderConfig,
    FisheyeBoxDecoderOutput,
    FisheyeBoxKeypointGenerator,
    FisheyeSparseBoxDecoder,
    RayGuidedBoxInitializer,
)
from .fisheye_evidence import (
    FisheyeEvidenceConfig,
    FisheyeObservationGeometry,
    UnifiedFisheyeEvidenceAttention,
)
from .fisheye_projection import (
    FisheyeProjectionModel,
    NativeFisheyeProjector,
    OCamFisheyeProjection,
)
from .ray_losses import FisheyeRayAuxiliaryLoss, RayAuxiliaryLossConfig
from .ray_state import (
    FisheyeRayStateRefinement,
    RayRefinementConfig,
    RayState,
)

__version__ = "0.1.0"

__all__ = [
    "ObservationEncoding",
    "PointwiseROSA",
    "FisheyeEvidenceConfig",
    "FisheyeBoxDecoderConfig",
    "FisheyeBoxDecoderOutput",
    "FisheyeBoxKeypointGenerator",
    "FisheyeObservationGeometry",
    "FisheyeProjectionModel",
    "FisheyeRayAuxiliaryLoss",
    "FisheyeRayDecoder",
    "FisheyeRayDecoderConfig",
    "FisheyeRayDecoderOutput",
    "FisheyeRayStateRefinement",
    "FisheyeSparseBoxDecoder",
    "NativeFisheyeProjector",
    "OCamFisheyeProjection",
    "ROSAConfig",
    "ROSAOutput",
    "ROSARefiner",
    "RayAuxiliaryLossConfig",
    "RayGuidedBoxInitializer",
    "RayRefinementConfig",
    "RayState",
    "SphericalRayQueryBank",
    "SphericalRayFeatureLifting",
    "UnifiedFisheyeEvidenceAttention",
]
