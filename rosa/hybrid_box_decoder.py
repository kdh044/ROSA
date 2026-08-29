"""Hybrid-state fisheye box decoder without hand-set physical update bounds.

The center remains ray-aligned as direction plus log-range.  Shape, yaw, and
velocity stay in the detector's global frame so a center-bearing correction
cannot rotate the box or its motion implicitly.  There is no tanh clamp or
hand-set maximum update.  Optional evidence prefusion and sigmoid update gates
are learned from features, while range uncertainty is an auxiliary per-stage
readout rather than recurrent box state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .fisheye_box_decoder import (
    FisheyeBoxEvidenceAttention,
    FisheyeBoxEvidenceOutput,
    FisheyeBoxKeypointGenerator,
    project_box_keypoints,
    sample_box_keypoints_fpn,
)
from .fisheye_evidence import FisheyeEvidenceConfig, UnifiedFisheyeEvidenceAttention
from .fisheye_projection import (
    NativeFisheyeProjector,
    sample_native_fisheye_fpn,
)
from .rfl import SphericalRayFeatureLifting
from .ray_state import normalize_direction, rotate_sin_cos, spherical_exp_map
from .sampling import FusedBilinearSampler


def _global_yaw(value: Tensor) -> Tensor:
    """Normalize a global sin/cos carrier with an identity fallback."""

    if value.shape[-1] != 2:
        raise ValueError("yaw must end in [sin, cos]")
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    identity = value.new_tensor((0.0, 1.0))
    return torch.where(norm > 1e-6, value / norm.clamp_min(1e-6), identity)


@dataclass(frozen=True)
class HybridBoxState:
    """Ray-aligned center with globally framed box attributes."""

    direction: Tensor
    log_range: Tensor
    log_size: Tensor
    yaw: Tensor
    velocity: Tensor

    def validate(self) -> "HybridBoxState":
        expected = {
            "direction": 3,
            "log_range": 1,
            "log_size": 3,
            "yaw": 2,
            "velocity": 3,
        }
        leading = self.direction.shape[:-1]
        for name, width in expected.items():
            value = getattr(self, name)
            if value.shape[:-1] != leading or value.shape[-1] != width:
                raise ValueError(f"invalid hybrid-state {name} shape")
        return self

    @property
    def range(self) -> Tensor:
        return self.log_range.exp()

    @property
    def center(self) -> Tensor:
        return normalize_direction(self.direction) * self.range

    @classmethod
    def from_sparse_anchor(cls, anchor: Tensor) -> "HybridBoxState":
        if anchor.ndim != 3 or anchor.shape[-1] != 11:
            raise ValueError("anchor must be [B,Q,11]")
        center = anchor[..., :3]
        distance = torch.linalg.vector_norm(center, dim=-1, keepdim=True)
        return cls(
            direction=normalize_direction(center),
            log_range=distance.clamp_min(1e-6).log(),
            log_size=anchor[..., 3:6],
            yaw=_global_yaw(anchor[..., 6:8]),
            velocity=anchor[..., 8:11],
        ).validate()

    def to_sparse_anchor(self) -> Tensor:
        self.validate()
        return torch.cat(
            (
                self.center,
                self.log_size,
                _global_yaw(self.yaw),
                self.velocity,
            ),
            dim=-1,
        )


def hybrid_log_angular_extent(state: HybridBoxState) -> Tensor:
    """Metric box footprint expressed as horizontal/vertical visual angle."""

    state.validate()
    size = state.log_size.exp()
    distance = state.range.clamp_min(1e-6)
    horizontal_size = size[..., :2].amax(dim=-1, keepdim=True)
    horizontal = 2.0 * torch.atan(0.5 * horizontal_size / distance)
    vertical = 2.0 * torch.atan(0.5 * size[..., 2:3] / distance)
    return torch.cat((horizontal, vertical), dim=-1).clamp_min(1e-6).log()


@dataclass(frozen=True)
class FisheyeBoxDecoderV4Config:
    """Architecture, dataset, and execution choices for the V4 family."""

    embed_dims: int = 256
    num_queries: int = 300
    num_layers: int = 6
    num_classes: int = 6
    num_heads: int = 8
    attention_dims: int = 64
    ffn_dims: int = 512
    num_cameras: int = 4
    level_strides: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0)
    num_learnable_points: int = 6
    fixed_keypoint_scale: float = 0.45
    point_chunk_size: int = 2
    checkpoint_point_chunks: bool = False
    use_fused_daf: bool = True
    camera_ray_rfl: bool = False
    camera_ray_low_rank_rfl: bool = False
    camera_ray_low_rank_rfl_layers: tuple[int, ...] | None = None
    camera_rfl_before_rosa: bool = False
    streaming_camera_rfl_cuda: bool = False
    predict_range_uncertainty: bool = False
    detach_range_uncertainty_input: bool = False
    state_conditioned_uncertainty: bool = False
    state_conditioned_refinement: bool = False
    learned_update_gate: bool = False
    c0_additive_size_update: bool = False
    c0_additive_size_update_layers: tuple[int, ...] | None = None
    state_conditioned_task_heads: bool = False
    sparse4d_task_heads: bool = False
    sparse4d_task_routing: bool = False
    sparse4d_refinement_trunk: bool = False
    dedicated_size_head: bool = False
    sparse4d_size_head_layers: tuple[int, ...] | None = None
    dedicated_size_gate: bool = False
    dedicated_size_gate_layers: tuple[int, ...] | None = None
    d0_center_prefusion: bool = False
    structured_face_contrast: bool = False
    structured_face_refinement: bool = False
    structured_face_refinement_layers: tuple[int, ...] | None = None
    structured_face_refinement_targets: tuple[str, ...] = ("size", "yaw")
    separate_face_size_residual: bool = False
    separate_face_yaw_residual: bool = False
    learned_face_yaw_gate: bool = False
    symmetric_face_size_evidence: bool = False
    yaw_update_mode: str = "rotation"
    pre_d0_query_evidence: bool = False
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_layers != 6:
            raise ValueError("V4 keeps the matched D0-D5 schedule")
        if self.embed_dims <= 0 or self.embed_dims % self.num_heads != 0:
            raise ValueError("invalid decoder width/head count")
        if self.attention_dims <= 0 or self.attention_dims % self.num_heads != 0:
            raise ValueError("invalid attention width/head count")
        if self.num_queries <= 0 or self.num_classes <= 0 or self.num_cameras <= 0:
            raise ValueError("query, class, and camera counts must be positive")
        if self.num_learnable_points < 0 or self.point_chunk_size <= 0:
            raise ValueError("invalid point configuration")
        if not 0.0 < self.fixed_keypoint_scale <= 0.5:
            raise ValueError("fixed_keypoint_scale must lie in (0, 0.5]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid dropout")
        if self.camera_ray_low_rank_rfl and not self.use_fused_daf:
            raise ValueError("low-rank camera RFL requires fused DAF")
        if self.camera_rfl_before_rosa and not self.use_fused_daf:
            raise ValueError("camera-wise RFL-before-ROSA requires fused DAF")
        if self.streaming_camera_rfl_cuda and not self.camera_rfl_before_rosa:
            raise ValueError(
                "streaming camera CUDA requires camera-wise RFL-before-ROSA"
            )
        if self.camera_rfl_before_rosa and (
            self.camera_ray_rfl or self.camera_ray_low_rank_rfl
        ):
            raise ValueError(
                "camera-wise RFL-before-ROSA is a separate camera RFL ablation"
            )
        if self.camera_ray_low_rank_rfl_layers is not None:
            if not self.camera_ray_low_rank_rfl:
                raise ValueError(
                    "low-rank camera RFL layers require the camera RFL branch"
                )
            if not self.camera_ray_low_rank_rfl_layers:
                raise ValueError("low-rank camera RFL layers cannot be empty")
            if len(set(self.camera_ray_low_rank_rfl_layers)) != len(
                self.camera_ray_low_rank_rfl_layers
            ):
                raise ValueError("low-rank camera RFL layers must be unique")
            if any(
                layer < 0 or layer >= self.num_layers
                for layer in self.camera_ray_low_rank_rfl_layers
            ):
                raise ValueError("low-rank camera RFL layer is outside D0-D5")
        if self.detach_range_uncertainty_input and not self.predict_range_uncertainty:
            raise ValueError(
                "detached uncertainty input requires an uncertainty readout"
            )
        if self.state_conditioned_uncertainty and not (
            self.predict_range_uncertainty
            and self.detach_range_uncertainty_input
        ):
            raise ValueError(
                "state-conditioned uncertainty requires a detached uncertainty "
                "readout"
            )
        if (
            self.c0_additive_size_update
            or self.c0_additive_size_update_layers is not None
        ) and not self.learned_update_gate:
            raise ValueError(
                "C0 additive size update requires the remaining learned update gates"
            )
        if self.c0_additive_size_update_layers is not None:
            if not self.c0_additive_size_update_layers:
                raise ValueError("C0 additive size layers cannot be empty")
            if len(set(self.c0_additive_size_update_layers)) != len(
                self.c0_additive_size_update_layers
            ):
                raise ValueError("C0 additive size layers must be unique")
            if any(
                layer < 0 or layer >= self.num_layers
                for layer in self.c0_additive_size_update_layers
            ):
                raise ValueError("C0 additive size layer is outside D0-D5")
        if self.dedicated_size_gate and not self.dedicated_size_head:
            raise ValueError(
                "a dedicated size gate requires the dedicated size head"
            )
        if self.sparse4d_size_head_layers is not None:
            if not self.dedicated_size_head:
                raise ValueError(
                    "Sparse4D-capacity size layers require a dedicated size head"
                )
            if not self.sparse4d_size_head_layers:
                raise ValueError("Sparse4D-capacity size layers cannot be empty")
            if len(set(self.sparse4d_size_head_layers)) != len(
                self.sparse4d_size_head_layers
            ):
                raise ValueError("Sparse4D-capacity size layers must be unique")
            if any(
                layer < 0 or layer >= self.num_layers
                for layer in self.sparse4d_size_head_layers
            ):
                raise ValueError(
                    "Sparse4D-capacity size layer is outside D0-D5"
                )
        if self.dedicated_size_gate_layers is not None:
            if not self.dedicated_size_gate:
                raise ValueError(
                    "dedicated size gate layers require a dedicated size gate"
                )
            if not self.dedicated_size_gate_layers:
                raise ValueError("dedicated size gate layers cannot be empty")
            if len(set(self.dedicated_size_gate_layers)) != len(
                self.dedicated_size_gate_layers
            ):
                raise ValueError("dedicated size gate layers must be unique")
            if any(
                layer < 0 or layer >= self.num_layers
                for layer in self.dedicated_size_gate_layers
            ):
                raise ValueError("dedicated size gate layer is outside D0-D5")
        if self.sparse4d_task_routing and not self.state_conditioned_task_heads:
            raise ValueError(
                "Sparse4D task routing requires a state-conditioned quality input"
            )
        if self.structured_face_contrast and self.structured_face_refinement:
            raise ValueError(
                "signed face evidence must use either shared contrast or "
                "targeted refinement, not both"
            )
        if self.structured_face_refinement_layers is not None:
            if not self.structured_face_refinement:
                raise ValueError(
                    "face refinement layers require targeted face refinement"
                )
            if not self.structured_face_refinement_layers:
                raise ValueError("face refinement layers cannot be empty")
            if len(set(self.structured_face_refinement_layers)) != len(
                self.structured_face_refinement_layers
            ):
                raise ValueError("face refinement layers must be unique")
            if any(
                layer < 0 or layer >= self.num_layers
                for layer in self.structured_face_refinement_layers
            ):
                raise ValueError("face refinement layer is outside D0-D5")
        valid_face_targets = {"size", "yaw"}
        if not self.structured_face_refinement_targets:
            raise ValueError("face refinement targets cannot be empty")
        if len(set(self.structured_face_refinement_targets)) != len(
            self.structured_face_refinement_targets
        ):
            raise ValueError("face refinement targets must be unique")
        if not set(self.structured_face_refinement_targets) <= valid_face_targets:
            raise ValueError("face refinement target must be size or yaw")
        if self.separate_face_yaw_residual and not (
            self.structured_face_refinement
            and "yaw" in self.structured_face_refinement_targets
        ):
            raise ValueError(
                "separate face-yaw residual requires targeted yaw refinement"
            )
        if self.separate_face_size_residual and not (
            self.structured_face_refinement
            and "size" in self.structured_face_refinement_targets
        ):
            raise ValueError(
                "separate face-size residual requires targeted size refinement"
            )
        if self.learned_face_yaw_gate and not self.separate_face_yaw_residual:
            raise ValueError(
                "learned face-yaw gate requires a separate face-yaw residual"
            )
        if self.symmetric_face_size_evidence and not (
            self.structured_face_refinement
            and "size" in self.structured_face_refinement_targets
        ):
            raise ValueError(
                "symmetric face-size evidence requires targeted size refinement"
            )
        if self.yaw_update_mode not in ("rotation", "vector_residual"):
            raise ValueError("unknown yaw update mode")

    @property
    def num_fixed_points(self) -> int:
        return 7

    @property
    def num_points(self) -> int:
        return self.num_fixed_points + self.num_learnable_points


class HybridBoxPositionEncoder(nn.Module):
    """Encode ray center plus globally framed shape, yaw, and velocity."""

    input_dims = 18  # SH9 + log-range1 + log-size3 + yaw2 + velocity3

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )

    def forward(self, state: HybridBoxState) -> Tensor:
        state.validate()
        inputs = torch.cat(
            (
                SphericalRayFeatureLifting.real_sh_degree_2(state.direction),
                state.log_range,
                state.log_size,
                _global_yaw(state.yaw),
                state.velocity,
            ),
            dim=-1,
        )
        return self.encoder(inputs)


class FisheyeCenterQueryInitializer(nn.Module):
    """Add one calibrated centre-ray observation without changing the box.

    This isolates the useful observation path from V3's pre-D0 initializer.
    It updates only the query carrier; direction, range, size, yaw, and velocity
    remain the exact C0 anchor until D0 performs its normal 13-point update.
    """

    def __init__(
        self,
        config: FisheyeBoxDecoderV4Config,
        projector: NativeFisheyeProjector,
        sampler: nn.Module,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.sampler = sampler
        self.evidence = UnifiedFisheyeEvidenceAttention(
            FisheyeEvidenceConfig(
                embed_dims=config.embed_dims,
                num_heads=config.num_heads,
                attention_dims=config.attention_dims,
                num_cameras=config.num_cameras,
                level_strides=config.level_strides,
                attention_dropout=config.dropout,
                projection_dropout=config.dropout,
                predict_angular_extent=False,
            )
        )
        self.norm = nn.LayerNorm(config.embed_dims)

    def forward(
        self,
        query: Tensor,
        state: HybridBoxState,
        fpn: Sequence[Tensor],
        metadata: dict,
    ) -> Tensor:
        state.validate()
        projection = self.projector(
            state.direction,
            metadata,
            points=state.center,
        )
        sampled = sample_native_fisheye_fpn(
            fpn,
            projection.uv,
            sampler=self.sampler,
        )
        evidence = self.evidence(
            query,
            sampled,
            projection.geometry,
            log_angular_extent=hybrid_log_angular_extent(state),
        )
        return self.norm(evidence.query)


class HybridBoxRefinement(nn.Module):
    """Zero-initialized unconstrained residuals in meaningful coordinates."""

    output_dims = 10  # default rotation mode, retained for V4 compatibility

    def __init__(
        self,
        embed_dims: int,
        *,
        state_conditioned: bool = False,
        learned_update_gate: bool = False,
        c0_additive_size_update: bool = False,
        yaw_update_mode: str = "rotation",
        sparse4d_trunk: bool = False,
        dedicated_size_head: bool = False,
        sparse4d_size_head: bool = False,
        dedicated_size_gate: bool = False,
        structured_face_refinement: bool = False,
        structured_face_refinement_targets: tuple[str, ...] = ("size", "yaw"),
        separate_face_size_residual: bool = False,
        separate_face_yaw_residual: bool = False,
        learned_face_yaw_gate: bool = False,
        symmetric_face_size_evidence: bool = False,
    ) -> None:
        super().__init__()
        self.state_conditioned = bool(state_conditioned)
        self.learned_update_gate = bool(learned_update_gate)
        self.c0_additive_size_update = bool(c0_additive_size_update)
        if self.c0_additive_size_update and not self.learned_update_gate:
            raise ValueError(
                "C0 additive size update requires the remaining learned update gates"
            )
        if yaw_update_mode not in ("rotation", "vector_residual"):
            raise ValueError("unknown yaw update mode")
        self.yaw_update_mode = yaw_update_mode
        self.output_dims = 10 if yaw_update_mode == "rotation" else 11
        self.dedicated_size_head = bool(dedicated_size_head)
        self.sparse4d_size_head = bool(sparse4d_size_head)
        self.dedicated_size_gate = bool(dedicated_size_gate)
        if self.sparse4d_size_head and not self.dedicated_size_head:
            raise ValueError(
                "a Sparse4D-capacity size head requires the dedicated size head"
            )
        if self.dedicated_size_gate and not self.dedicated_size_head:
            raise ValueError(
                "a dedicated size gate requires the dedicated size head"
            )
        self.structured_face_refinement = bool(structured_face_refinement)
        self.structured_face_refinement_targets = frozenset(
            structured_face_refinement_targets
        )
        self.separate_face_size_residual = bool(separate_face_size_residual)
        self.separate_face_yaw_residual = bool(separate_face_yaw_residual)
        self.learned_face_yaw_gate = bool(learned_face_yaw_gate)
        self.symmetric_face_size_evidence = bool(symmetric_face_size_evidence)
        if self.separate_face_size_residual and not (
            self.structured_face_refinement
            and "size" in self.structured_face_refinement_targets
        ):
            raise ValueError(
                "separate face-size residual requires targeted size refinement"
            )
        if self.separate_face_yaw_residual and not (
            self.structured_face_refinement
            and "yaw" in self.structured_face_refinement_targets
        ):
            raise ValueError(
                "separate face-yaw residual requires targeted yaw refinement"
            )
        if self.learned_face_yaw_gate and not self.separate_face_yaw_residual:
            raise ValueError(
                "learned face-yaw gate requires a separate face-yaw residual"
            )
        if self.symmetric_face_size_evidence and not (
            self.structured_face_refinement
            and "size" in self.structured_face_refinement_targets
        ):
            raise ValueError(
                "symmetric face-size evidence requires targeted size refinement"
            )
        hidden_layers: list[nn.Module] = []
        if sparse4d_trunk:
            for _ in range(2):
                for _ in range(2):
                    hidden_layers.extend(
                        (nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True))
                    )
                hidden_layers.append(nn.LayerNorm(embed_dims))
        else:
            hidden_layers.extend(
                (
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(embed_dims),
                )
            )
        self.trunk = nn.Sequential(
            *hidden_layers,
            nn.Linear(embed_dims, self.output_dims),
        )
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)
        if self.dedicated_size_head:
            # Give extent an independent representation. Forking preserves
            # every established parameter's RNG state, and the zero output
            # layer keeps the initial decoder function exactly unchanged.
            with torch.random.fork_rng(devices=[]):
                size_hidden: list[nn.Module] = []
                if self.sparse4d_size_head:
                    # Match Sparse4D's regression-head capacity, but only for
                    # the terminal extent expert selected by the decoder.  It
                    # remains independent of centre/yaw/velocity features.
                    for _ in range(2):
                        for _ in range(2):
                            size_hidden.extend(
                                (
                                    nn.Linear(embed_dims, embed_dims),
                                    nn.ReLU(inplace=True),
                                )
                            )
                        size_hidden.append(nn.LayerNorm(embed_dims))
                else:
                    size_hidden.extend(
                        (
                            nn.Linear(embed_dims, embed_dims),
                            nn.ReLU(inplace=True),
                            nn.LayerNorm(embed_dims),
                        )
                    )
                self.size_trunk = nn.Sequential(
                    *size_hidden,
                    nn.Linear(embed_dims, 3),
                )
            nn.init.zeros_(self.size_trunk[-1].weight)
            nn.init.zeros_(self.size_trunk[-1].bias)
            if self.dedicated_size_gate:
                # Predict size trust from the same task-specific representation
                # as the residual.  Forking keeps every established layer's
                # initialization identical, and zero initialization reproduces
                # the shared gate's initial sigmoid value of 0.5.
                with torch.random.fork_rng(devices=[]):
                    self.size_gate = nn.Linear(embed_dims, 1)
                nn.init.zeros_(self.size_gate.weight)
                nn.init.zeros_(self.size_gate.bias)
            else:
                self.size_gate = None
        else:
            self.size_trunk = None
            self.size_gate = None
        self.update_gate = (
            nn.Linear(embed_dims, 4) if self.learned_update_gate else None
        )
        if self.update_gate is not None:
            nn.init.zeros_(self.update_gate.weight)
            nn.init.zeros_(self.update_gate.bias)
        face_output_dims = 4 if yaw_update_mode == "rotation" else 5
        # The targeted arm must remain a controlled, same-seed extension of
        # the baseline.  ``nn.Linear`` samples an initialization even though
        # we immediately reset it to zero; without the fork that consumption
        # would shift every later decoder layer's baseline initialization.
        if self.structured_face_refinement:
            with torch.random.fork_rng(devices=[]):
                self.face_refinement = nn.Linear(
                    3 * embed_dims,
                    face_output_dims,
                    bias=False,
                )
        else:
            self.face_refinement = None
        if self.face_refinement is not None:
            nn.init.zeros_(self.face_refinement.weight)
        if self.learned_face_yaw_gate:
            with torch.random.fork_rng(devices=[]):
                self.face_yaw_gate = nn.Linear(embed_dims, 1)
            nn.init.zeros_(self.face_yaw_gate.weight)
            nn.init.zeros_(self.face_yaw_gate.bias)
        else:
            self.face_yaw_gate = None

    def forward(
        self,
        feature: Tensor,
        state: HybridBoxState,
        state_feature: Tensor | None = None,
        signed_face_features: Tensor | None = None,
        symmetric_face_features: Tensor | None = None,
    ) -> HybridBoxState:
        state.validate()
        if self.state_conditioned:
            if state_feature is None or state_feature.shape != feature.shape:
                raise ValueError(
                    "state-conditioned refinement requires a matching state feature"
                )
            refinement_feature = feature + state_feature
        else:
            refinement_feature = feature
        raw = self.trunk(refinement_feature.float()).to(feature.dtype)
        dedicated_size_feature = None
        if self.size_trunk is not None:
            dedicated_size_feature = self.size_trunk[:-1](
                refinement_feature.float()
            )
            dedicated_size = self.size_trunk[-1](dedicated_size_feature).to(
                feature.dtype
            )
            raw = torch.cat(
                (raw[..., :3], dedicated_size, raw[..., 6:]),
                dim=-1,
            )
        separate_face_size = None
        separate_face_yaw = None
        if self.face_refinement is not None:
            expected = feature.shape[:-1] + (3, feature.shape[-1])
            if signed_face_features is None or signed_face_features.shape != expected:
                raise ValueError(
                    "structured face refinement requires [B,Q,3,embed_dims]"
                )
            face_raw = self.face_refinement(
                signed_face_features.flatten(-2).float()
            ).to(feature.dtype)
            if self.symmetric_face_size_evidence:
                if (
                    symmetric_face_features is None
                    or symmetric_face_features.shape != expected
                ):
                    raise ValueError(
                        "symmetric size refinement requires "
                        "[B,Q,3,embed_dims]"
                    )
                symmetric_size = F.linear(
                    symmetric_face_features.flatten(-2).float(),
                    self.face_refinement.weight[:3].float(),
                ).to(feature.dtype)
                face_raw = torch.cat((symmetric_size, face_raw[..., 3:]), dim=-1)
            if self.yaw_update_mode == "rotation":
                face_size = (
                    face_raw[..., :3]
                    if "size" in self.structured_face_refinement_targets
                    else 0.0
                )
                face_yaw = (
                    face_raw[..., 3:4]
                    if "yaw" in self.structured_face_refinement_targets
                    else 0.0
                )
                if self.separate_face_size_residual:
                    separate_face_size = face_size
                    face_size = 0.0
                if self.separate_face_yaw_residual:
                    separate_face_yaw = face_yaw
                    face_yaw = 0.0
                raw = torch.cat(
                    (
                        raw[..., :3],
                        raw[..., 3:6] + face_size,
                        raw[..., 6:7] + face_yaw,
                        raw[..., 7:],
                    ),
                    dim=-1,
                )
            else:
                face_size = (
                    face_raw[..., :3]
                    if "size" in self.structured_face_refinement_targets
                    else 0.0
                )
                face_yaw = (
                    face_raw[..., 3:5]
                    if "yaw" in self.structured_face_refinement_targets
                    else 0.0
                )
                if self.separate_face_size_residual:
                    separate_face_size = face_size
                    face_size = 0.0
                if self.separate_face_yaw_residual:
                    separate_face_yaw = face_yaw
                    face_yaw = 0.0
                raw = torch.cat(
                    (
                        raw[..., :3],
                        raw[..., 3:6] + face_size,
                        raw[..., 6:8] + face_yaw,
                        raw[..., 8:],
                    ),
                    dim=-1,
                )
        use_fused_v49_state_update = (
            not torch.is_grad_enabled()
            and os.getenv("ROSA_DISABLE_FUSED_STATE_UPDATE", "0") != "1"
            and feature.is_cuda
            and self.yaw_update_mode == "rotation"
            and self.update_gate is not None
            and self.size_trunk is not None
            and self.size_gate is None
            and self.face_refinement is not None
            and separate_face_size is None
            and separate_face_yaw is not None
            and self.face_yaw_gate is not None
            and not self.c0_additive_size_update
        )
        if use_fused_v49_state_update:
            from projects.mmdet3d_plugin.ops import (
                v49_hybrid_state_update_inference,
            )

            update_gate_logits = self.update_gate(refinement_feature.float())
            face_yaw_gate_logits = self.face_yaw_gate(
                refinement_feature.float()
            )
            updated = v49_hybrid_state_update_inference(
                state.direction,
                state.log_range,
                state.log_size,
                state.yaw,
                state.velocity,
                raw,
                update_gate_logits,
                separate_face_yaw,
                face_yaw_gate_logits,
            )
            return HybridBoxState(*updated).validate()
        if self.update_gate is None:
            center_gate = size_gate = yaw_gate = velocity_gate = 1.0
        else:
            gates = (
                self.update_gate(refinement_feature.float()).sigmoid().to(feature.dtype)
            )
            center_gate = gates[..., 0:1]
            size_gate = gates[..., 1:2]
            yaw_gate = gates[..., 2:3]
            velocity_gate = gates[..., 3:4]
            if self.c0_additive_size_update:
                # Sparse4D applies the log-size residual additively.  This
                # controlled path removes only the learned multiplicative
                # attenuation on shape; it adds no tuned scale or bound and
                # retains learned trust for centre, yaw, and velocity.
                size_gate = 1.0
        if self.size_gate is not None:
            if dedicated_size_feature is None:
                raise RuntimeError("dedicated size gate has no size representation")
            size_gate = self.size_gate(dedicated_size_feature).sigmoid().to(
                feature.dtype
            )
        size_residual = raw[..., 3:6] * size_gate
        if separate_face_size is not None:
            size_residual = size_residual + separate_face_size
        if separate_face_yaw is not None and self.face_yaw_gate is not None:
            face_yaw_gate = self.face_yaw_gate(
                refinement_feature.float()
            ).sigmoid().to(feature.dtype)
            separate_face_yaw = separate_face_yaw * face_yaw_gate
        if self.yaw_update_mode == "rotation":
            yaw_residual = raw[..., 6:7] * yaw_gate
            if separate_face_yaw is not None:
                yaw_residual = yaw_residual + separate_face_yaw
            yaw = rotate_sin_cos(_global_yaw(state.yaw), yaw_residual)
            velocity_residual = raw[..., 7:10]
        else:
            # Match the established Sparse4D yaw geometry: update the two
            # components in their global frame, then project back to S1.  The
            # residual is learned and unconstrained; no maximum angle is used.
            yaw_residual = raw[..., 6:8] * yaw_gate
            if separate_face_yaw is not None:
                yaw_residual = yaw_residual + separate_face_yaw
            yaw = _global_yaw(_global_yaw(state.yaw) + yaw_residual)
            velocity_residual = raw[..., 8:11]
        return HybridBoxState(
            direction=spherical_exp_map(state.direction, raw[..., 0:2] * center_gate),
            log_range=state.log_range + raw[..., 2:3] * center_gate,
            log_size=state.log_size + size_residual,
            yaw=yaw,
            velocity=state.velocity + velocity_residual * velocity_gate,
        ).validate()


class FisheyeSparseBoxDecoderV4Layer(nn.Module):
    """One D0-D5 observation and hybrid-state refinement stage."""

    def __init__(
        self,
        config: FisheyeBoxDecoderV4Config,
        projector: NativeFisheyeProjector,
        sampler: nn.Module,
        *,
        center_prefusion: bool = False,
        structured_face_refinement: bool | None = None,
        sparse4d_size_head: bool = False,
        dedicated_size_gate: bool | None = None,
        c0_additive_size_update: bool | None = None,
        camera_ray_low_rank_rfl: bool | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.projector = projector
        self.sampler = sampler
        self.position_encoder = HybridBoxPositionEncoder(config.embed_dims)
        self.self_attention = nn.MultiheadAttention(
            config.embed_dims,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(config.embed_dims)
        self.keypoints = FisheyeBoxKeypointGenerator(
            config.embed_dims,
            config.num_learnable_points,
            config.fixed_keypoint_scale,
        )
        targeted_face_refinement = (
            config.structured_face_refinement
            if structured_face_refinement is None
            else bool(structured_face_refinement)
        )
        targeted_size_gate = (
            config.dedicated_size_gate
            if dedicated_size_gate is None
            else bool(dedicated_size_gate)
        )
        targeted_additive_size = (
            config.c0_additive_size_update
            if c0_additive_size_update is None
            else bool(c0_additive_size_update)
        )
        self.evidence = FisheyeBoxEvidenceAttention(
            config,
            center_prefusion=center_prefusion,
            return_signed_faces=targeted_face_refinement,
            return_symmetric_faces=(
                targeted_face_refinement
                and config.symmetric_face_size_evidence
            ),
            camera_ray_low_rank_rfl=camera_ray_low_rank_rfl,
        )
        self.evidence_norm = nn.LayerNorm(config.embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dims, config.ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dims, config.embed_dims),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.embed_dims)
        self.refinement = HybridBoxRefinement(
            config.embed_dims,
            state_conditioned=config.state_conditioned_refinement,
            learned_update_gate=config.learned_update_gate,
            c0_additive_size_update=targeted_additive_size,
            yaw_update_mode=config.yaw_update_mode,
            sparse4d_trunk=config.sparse4d_refinement_trunk,
            dedicated_size_head=config.dedicated_size_head,
            sparse4d_size_head=sparse4d_size_head,
            dedicated_size_gate=targeted_size_gate,
            structured_face_refinement=targeted_face_refinement,
            structured_face_refinement_targets=(
                config.structured_face_refinement_targets
            ),
            separate_face_size_residual=(
                config.separate_face_size_residual
                and targeted_face_refinement
            ),
            separate_face_yaw_residual=(
                config.separate_face_yaw_residual
                and targeted_face_refinement
            ),
            learned_face_yaw_gate=(
                config.learned_face_yaw_gate
                and targeted_face_refinement
            ),
            symmetric_face_size_evidence=(
                config.symmetric_face_size_evidence
                and targeted_face_refinement
            ),
        )

    def forward(
        self,
        query: Tensor,
        state: HybridBoxState,
        fpn: Sequence[Tensor],
        metadata: dict,
        self_attention_mask: Tensor | None = None,
        formatted_fpn: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> tuple[
        Tensor,
        HybridBoxState,
        FisheyeBoxEvidenceOutput,
        Tensor,
        Tensor,
    ]:
        use_fused_projection = (
            self.config.use_fused_daf
            and not self.training
            and not torch.is_grad_enabled()
            and state.direction.is_cuda
            and os.getenv("ROSA_DISABLE_FUSED_PROJECTION", "0") != "1"
        )
        use_fused_state_keypoints = (
            use_fused_projection
            and os.getenv("ROSA_DISABLE_FUSED_STATE_KEYPOINTS", "0") != "1"
        )
        if use_fused_state_keypoints:
            from projects.mmdet3d_plugin.ops import (
                v49_state_geometry_inputs_inference,
            )

            position_inputs, extent = v49_state_geometry_inputs_inference(
                state.direction,
                state.log_range,
                state.log_size,
                state.yaw,
                state.velocity,
            )
            position = self.position_encoder.encoder(position_inputs)
        else:
            position = self.position_encoder(state)
            extent = hybrid_log_angular_extent(state)
        attended, _ = self.self_attention(
            query + position,
            query + position,
            query,
            attn_mask=self_attention_mask,
            need_weights=False,
        )
        query = self.self_norm(query + attended)
        precomputed_geometry = None
        if use_fused_state_keypoints:
            from projects.mmdet3d_plugin.ops import (
                v49_build_keypoints_project_geometry_inference,
            )
            from .fisheye_box_decoder import BoxPointProjection
            from .fisheye_evidence import FisheyeObservationGeometry

            batch, queries = query.shape[:2]
            points = self.keypoints.num_points
            cameras = self.config.num_cameras
            raw_fov = metadata.get(
                "fisheye_lens_fov_deg", metadata.get("ocam_lens_fov_deg", 220.0)
            )
            fov = torch.as_tensor(
                raw_fov, device=query.device, dtype=query.dtype
            )
            if fov.numel() == 1:
                fov = fov.reshape(1, 1).expand(batch, cameras)
            elif fov.ndim == 1 and fov.shape[0] == batch:
                fov = fov[:, None].expand(batch, cameras)
            elif tuple(fov.shape) != (batch, cameras):
                raise ValueError("lens FOV must be scalar, [B], or [B,C]")
            learned_keypoints = self.keypoints.learnable_fc(query.float())
            (
                keypoints_flat,
                point_rays_flat,
                uv,
                rays,
                valid,
                descriptor,
                compatibility,
            ) = v49_build_keypoints_project_geometry_inference(
                    state.direction, state.log_range, state.log_size, state.yaw,
                    learned_keypoints, metadata["lidar2cam"],
                    metadata["image_wh"], metadata["ocam_intrinsic"],
                    metadata["ocam_pol"], torch.deg2rad(fov * 0.5), extent,
                    self.evidence.level_strides, points,
                    self.config.fixed_keypoint_scale,
                )
            keypoints = keypoints_flat.reshape(batch, queries, points, 3)
            point_rays = point_rays_flat.reshape(batch, queries, points, 3)
            empty = rays.new_empty((0,))
            geometry = FisheyeObservationGeometry(
                rays=rays.reshape(batch, queries, points, cameras, 3),
                tangent=empty, jacobian=empty, lens_margin=empty,
                overlap=empty,
                valid=valid.reshape(batch, queries, points, cameras),
            )
            projection = BoxPointProjection(
                uv=uv.reshape(batch, queries, points, cameras, 2),
                geometry=geometry,
            )
            precomputed_geometry = (descriptor, compatibility)
        elif use_fused_projection:
            from projects.mmdet3d_plugin.ops import v49_project_geometry_inference
            from .fisheye_box_decoder import BoxPointProjection
            from .fisheye_evidence import FisheyeObservationGeometry

            keypoints = self.keypoints(query, state)
            point_rays = normalize_direction(keypoints)
            batch, queries, points = keypoints.shape[:3]
            cameras = self.config.num_cameras
            raw_fov = metadata.get(
                "fisheye_lens_fov_deg", metadata.get("ocam_lens_fov_deg", 220.0)
            )
            fov = torch.as_tensor(raw_fov, device=query.device, dtype=query.dtype)
            if fov.numel() == 1:
                fov = fov.reshape(1, 1).expand(batch, cameras)
            elif fov.ndim == 1 and fov.shape[0] == batch:
                fov = fov[:, None].expand(batch, cameras)
            elif tuple(fov.shape) != (batch, cameras):
                raise ValueError("lens FOV must be scalar, [B], or [B,C]")
            uv, rays, valid, descriptor, compatibility = (
                v49_project_geometry_inference(
                    keypoints.reshape(batch, queries * points, 3),
                    metadata["lidar2cam"], metadata["image_wh"],
                    metadata["ocam_intrinsic"], metadata["ocam_pol"],
                    torch.deg2rad(fov * 0.5), extent,
                    self.evidence.level_strides, points,
                )
            )
            empty = rays.new_empty((0,))
            geometry = FisheyeObservationGeometry(
                rays=rays.reshape(batch, queries, points, cameras, 3),
                tangent=empty, jacobian=empty, lens_margin=empty,
                overlap=empty,
                valid=valid.reshape(batch, queries, points, cameras),
            )
            projection = BoxPointProjection(
                uv=uv.reshape(batch, queries, points, cameras, 2),
                geometry=geometry,
            )
            precomputed_geometry = (descriptor, compatibility)
        else:
            keypoints = self.keypoints(query, state)
            point_rays = normalize_direction(keypoints)
            projection = project_box_keypoints(self.projector, keypoints, metadata)
        if self.config.use_fused_daf:
            if formatted_fpn is None:
                raise RuntimeError("fused V4 layer requires formatted FPN features")
            evidence = self.evidence.forward_fused_daf(
                query,
                projection,
                extent,
                point_rays,
                formatted_fpn,
                precomputed_geometry,
            )
        elif self.training and self.config.checkpoint_point_chunks:
            evidence = self.evidence.forward_streaming_training(
                query,
                fpn,
                projection,
                extent,
                self.sampler,
            )
        else:
            sampled = sample_box_keypoints_fpn(fpn, projection.uv, sampler=self.sampler)
            evidence = self.evidence(query, sampled, projection.geometry, extent)
        query = self.evidence_norm(evidence.query)
        query = self.ffn_norm(query + self.ffn(query))
        task_feature = (
            query + position
            if self.config.state_conditioned_task_heads
            else query
        )
        state = self.refinement(
            query,
            state,
            position,
            evidence.signed_face_features,
            evidence.symmetric_face_features,
        )
        return query, state, evidence, task_feature, position


@dataclass(frozen=True)
class FisheyeBoxDecoderV4Output:
    classification: tuple[Tensor, ...]
    prediction: tuple[Tensor, ...]
    quality: tuple[Tensor, ...]
    states: tuple[HybridBoxState, ...]
    camera_level_attention: tuple[Tensor, ...]
    point_attention: tuple[Tensor, ...]
    level_compatibility: tuple[Tensor, ...]
    raw_log_range_std: tuple[Tensor, ...] | None
    query: Tensor


class FisheyeSparseBoxDecoderV4(nn.Module):
    """Six-stage decoder with no pre-D0 observation or seventh refinement."""

    def __init__(
        self,
        config: FisheyeBoxDecoderV4Config | None = None,
        *,
        projector: NativeFisheyeProjector | None = None,
        sampler: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or FisheyeBoxDecoderV4Config()
        self.projector = projector or NativeFisheyeProjector()
        self.sampler = sampler or FusedBilinearSampler()
        if self.config.pre_d0_query_evidence:
            self.initializer = FisheyeCenterQueryInitializer(
                self.config,
                self.projector,
                self.sampler,
            )
        self.layers = nn.ModuleList(
            FisheyeSparseBoxDecoderV4Layer(
                self.config,
                self.projector,
                self.sampler,
                center_prefusion=(
                    self.config.d0_center_prefusion and layer_index == 0
                ),
                structured_face_refinement=(
                    self.config.structured_face_refinement
                    and (
                        self.config.structured_face_refinement_layers is None
                        or layer_index
                        in self.config.structured_face_refinement_layers
                    )
                ),
                sparse4d_size_head=(
                    self.config.sparse4d_size_head_layers is not None
                    and layer_index in self.config.sparse4d_size_head_layers
                ),
                dedicated_size_gate=(
                    self.config.dedicated_size_gate
                    and (
                        self.config.dedicated_size_gate_layers is None
                        or layer_index in self.config.dedicated_size_gate_layers
                    )
                ),
                c0_additive_size_update=(
                    self.config.c0_additive_size_update
                    or (
                        self.config.c0_additive_size_update_layers is not None
                        and layer_index
                        in self.config.c0_additive_size_update_layers
                    )
                ),
                camera_ray_low_rank_rfl=(
                    self.config.camera_ray_low_rank_rfl
                    and (
                        self.config.camera_ray_low_rank_rfl_layers is None
                        or layer_index
                        in self.config.camera_ray_low_rank_rfl_layers
                    )
                ),
            )
            for layer_index in range(self.config.num_layers)
        )

        def make_task_head(output_dims: int) -> nn.Module:
            if not self.config.sparse4d_task_heads:
                return nn.Linear(self.config.embed_dims, output_dims)
            layers: list[nn.Module] = []
            for _ in range(2):
                layers.extend(
                    (
                        nn.Linear(self.config.embed_dims, self.config.embed_dims),
                        nn.ReLU(inplace=True),
                        nn.LayerNorm(self.config.embed_dims),
                    )
                )
            layers.append(nn.Linear(self.config.embed_dims, output_dims))
            return nn.Sequential(*layers)

        self.classifiers = nn.ModuleList(
            make_task_head(self.config.num_classes)
            for _ in range(self.config.num_layers)
        )
        self.quality_heads = nn.ModuleList(
            make_task_head(2) for _ in range(self.config.num_layers)
        )
        self.range_uncertainty_heads = (
            nn.ModuleList(
                nn.Linear(self.config.embed_dims, 1)
                for _ in range(self.config.num_layers)
            )
            if self.config.predict_range_uncertainty
            else None
        )
        prior = 0.01
        bias = math.log(prior / (1.0 - prior))
        for classifier in self.classifiers:
            output_layer = (
                classifier[-1]
                if isinstance(classifier, nn.Sequential)
                else classifier
            )
            if not self.config.sparse4d_task_heads:
                nn.init.xavier_uniform_(output_layer.weight)
            nn.init.constant_(output_layer.bias, bias)
        if not self.config.sparse4d_task_heads:
            for quality in self.quality_heads:
                nn.init.xavier_uniform_(quality.weight)
                nn.init.zeros_(quality.bias)
        if self.range_uncertainty_heads is not None:
            for uncertainty in self.range_uncertainty_heads:
                nn.init.zeros_(uncertainty.weight)
                nn.init.zeros_(uncertainty.bias)

    def forward(
        self,
        fpn: Sequence[Tensor],
        metadata: dict,
        instance_feature: Tensor,
        anchor: Tensor,
        *,
        self_attention_mask: Tensor | None = None,
    ) -> FisheyeBoxDecoderV4Output:
        if not fpn:
            raise ValueError("decoder requires FPN features")
        if (
            instance_feature.ndim != 3
            or instance_feature.shape[-1] != self.config.embed_dims
        ):
            raise ValueError("instance_feature must be [B,Q,embed_dims]")
        if anchor.shape != instance_feature.shape[:2] + (11,):
            raise ValueError("anchor must be [B,Q,11]")
        state = HybridBoxState.from_sparse_anchor(anchor)
        query = instance_feature
        if self.config.pre_d0_query_evidence:
            query = self.initializer(query, state, fpn, metadata)
        formatted_fpn = None
        if self.config.use_fused_daf:
            try:
                from projects.mmdet3d_plugin.ops import feature_maps_format
            except ImportError as exc:  # pragma: no cover - host-only path
                raise RuntimeError(
                    "fused V4 sampling requires the Sparse4D CUDA host"
                ) from exc
            formatted_fpn = feature_maps_format(fpn)

        classifications = []
        predictions = []
        qualities = []
        states = []
        camera_level_attention = []
        point_attention = []
        level_compatibility = []
        raw_log_range_std = []
        for layer_index, (layer, classifier, quality_head) in enumerate(
            zip(self.layers, self.classifiers, self.quality_heads)
        ):
            query, state, evidence, task_feature, position = layer(
                query,
                state,
                fpn,
                metadata,
                self_attention_mask,
                formatted_fpn,
            )
            classification_feature = (
                query if self.config.sparse4d_task_routing else task_feature
            )
            classifications.append(classifier(classification_feature))
            predictions.append(state.to_sparse_anchor()[..., :10])
            qualities.append(quality_head(task_feature))
            if self.range_uncertainty_heads is not None:
                uncertainty_feature = (
                    query + position
                    if self.config.state_conditioned_uncertainty
                    else query
                )
                uncertainty_input = (
                    uncertainty_feature.detach()
                    if self.config.detach_range_uncertainty_input
                    else uncertainty_feature
                )
                raw_log_range_std.append(
                    self.range_uncertainty_heads[layer_index](uncertainty_input)
                )
            states.append(state)
            camera_level_attention.append(evidence.camera_level_attention)
            point_attention.append(evidence.point_attention)
            level_compatibility.append(evidence.level_compatibility)
        return FisheyeBoxDecoderV4Output(
            classification=tuple(classifications),
            prediction=tuple(predictions),
            quality=tuple(qualities),
            states=tuple(states),
            camera_level_attention=tuple(camera_level_attention),
            point_attention=tuple(point_attention),
            level_compatibility=tuple(level_compatibility),
            raw_log_range_std=(
                tuple(raw_log_range_std)
                if self.range_uncertainty_heads is not None
                else None
            ),
            query=query,
        )


__all__ = [
    "FisheyeBoxDecoderV4Config",
    "FisheyeBoxDecoderV4Output",
    "FisheyeCenterQueryInitializer",
    "FisheyeSparseBoxDecoderV4",
    "HybridBoxRefinement",
    "HybridBoxState",
]
