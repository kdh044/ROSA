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
from .hybrid_box_decoder import (
    FisheyeBoxDecoderV4Config,
    FisheyeBoxDecoderV4Output,
    FisheyeSparseBoxDecoderV4,
    HybridBoxRefinement,
    HybridBoxState,
)
from .fisheye_projection import (
    FisheyeProjectionModel,
    NativeFisheyeProjector,
    OCamFisheyeProjection,
)
from .ray_losses import (
    FisheyeHybridRayAuxiliaryLoss,
    FisheyeRayAuxiliaryLoss,
    RayAuxiliaryLossConfig,
)
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
    "FisheyeBoxDecoderV4Config",
    "FisheyeBoxDecoderV4Output",
    "FisheyeObservationGeometry",
    "FisheyeProjectionModel",
    "FisheyeHybridRayAuxiliaryLoss",
    "FisheyeRayAuxiliaryLoss",
    "FisheyeRayDecoder",
    "FisheyeRayDecoderConfig",
    "FisheyeRayDecoderOutput",
    "FisheyeRayStateRefinement",
    "FisheyeSparseBoxDecoder",
    "FisheyeSparseBoxDecoderV4",
    "NativeFisheyeProjector",
    "OCamFisheyeProjection",
    "ROSAConfig",
    "ROSAOutput",
    "ROSARefiner",
    "RayAuxiliaryLossConfig",
    "RayGuidedBoxInitializer",
    "RayRefinementConfig",
    "RayState",
    "HybridBoxRefinement",
    "HybridBoxState",
    "SphericalRayQueryBank",
    "SphericalRayFeatureLifting",
    "UnifiedFisheyeEvidenceAttention",
]
