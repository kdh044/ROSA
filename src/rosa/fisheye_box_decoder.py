"""Ray-initialized sparse box decoder for native fisheye cameras.

The decoder keeps a conventional 11-value Sparse4D anchor and query feature,
uses one calibrated centre-ray observation to produce a fisheye-corrected
initial box, then re-observes 13 box support points in every D0--D5 stage.
Camera/FPN fusion and point fusion are deliberately factorized: ROSA first
selects camera x level evidence independently for every semantic point, then a
query-conditioned attention aggregates the points.  The production path uses
the Sparse4D fused CUDA sampler only as a weighted native-fisheye feature read;
the weights and subsequent per-point RFL remain V3-owned operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .fisheye_decoder import RayStatePositionEncoder, state_log_angular_extent
from .fisheye_evidence import (
    FisheyeEvidenceConfig,
    FisheyeObservationGeometry,
    UnifiedFisheyeEvidenceAttention,
    angular_fpn_geometry,
)
from .fisheye_projection import (
    NativeFisheyeProjector,
    NativeFisheyeProjection,
    sample_native_fisheye_fpn,
)
from .ray_state import (
    FisheyeRayStateRefinement,
    RayRefinementConfig,
    RayState,
    normalize_direction,
)
from .rfl import SphericalRayFeatureLifting
from .sampling import FusedBilinearSampler
from .types import RayFeatureRelation


@dataclass(frozen=True)
class FisheyeBoxDecoderConfig:
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
    checkpoint_point_chunks: bool = True
    use_fused_daf: bool = False
    camera_ray_rfl: bool = False
    camera_ray_low_rank_rfl: bool = False
    camera_rfl_before_rosa: bool = False
    structured_face_contrast: bool = False
    structured_face_refinement: bool = False
    dropout: float = 0.0
    min_range_m: float = 0.5
    max_range_m: float = 68.0

    def __post_init__(self) -> None:
        if self.embed_dims <= 0 or self.num_queries <= 0 or self.num_layers <= 0:
            raise ValueError("decoder dimensions must be positive")
        if self.num_layers != 6:
            raise ValueError("V3 fixes the Sparse4D D0--D5 six-stage schedule")
        if self.embed_dims % self.num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")
        if self.attention_dims <= 0 or self.attention_dims % self.num_heads != 0:
            raise ValueError("attention_dims must be divisible by num_heads")
        if self.num_classes <= 0 or self.num_cameras <= 0:
            raise ValueError("class and camera counts must be positive")
        if self.num_learnable_points < 0:
            raise ValueError("num_learnable_points must be non-negative")
        if self.point_chunk_size <= 0:
            raise ValueError("point_chunk_size must be positive")
        if not 0.0 < self.fixed_keypoint_scale <= 0.5:
            raise ValueError("fixed_keypoint_scale must lie in (0, 0.5]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid decoder dropout")
        if self.camera_ray_low_rank_rfl and not self.use_fused_daf:
            raise ValueError("low-rank camera RFL requires fused DAF")
        if self.camera_rfl_before_rosa and not self.use_fused_daf:
            raise ValueError("camera-wise RFL-before-ROSA requires fused DAF")
        if self.camera_rfl_before_rosa and (
            self.camera_ray_rfl or self.camera_ray_low_rank_rfl
        ):
            raise ValueError(
                "camera-wise RFL-before-ROSA is a separate camera RFL ablation"
            )

    @property
    def num_fixed_points(self) -> int:
        return 7

    @property
    def num_points(self) -> int:
        return self.num_fixed_points + self.num_learnable_points


@dataclass(frozen=True)
class BoxPointProjection:
    """Native-fisheye projection with an explicit box-point dimension."""

    uv: Tensor
    geometry: FisheyeObservationGeometry


@dataclass(frozen=True)
class FisheyeBoxEvidenceOutput:
    query: Tensor
    camera_level_attention: Tensor
    point_attention: Tensor
    level_compatibility: Tensor
    supported: Tensor
    signed_face_features: Tensor | None = None
    symmetric_face_features: Tensor | None = None


@dataclass(frozen=True)
class RayGuidedInitializationOutput:
    query: Tensor
    state: RayState
    gate: Tensor
    projection: NativeFisheyeProjection
    camera_level_attention: Tensor


class CameraRaySHValueModulation(nn.Module):
    """Inject camera-ray identity into DAF values without another image read.

    ROSA owns the positive camera-by-level selection distribution.  This
    module supplies a zero-initialized, degree-2-SH residual independently for
    every DAF channel group.  Algebraically, multiplying DAF's grouped weights
    is the same as multiplying the sampled values, but it lets the existing
    CUDA reader perform sampling and reduction in one call.

    The learned ray response is centred under the ROSA distribution.  The
    effective weights therefore retain the exact per-head mass of the input
    distribution without a hand-selected scale, clamp, or normalization
    constant.
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError("camera-ray RFL requires a positive head count")
        self.num_heads = int(num_heads)
        self.ray_gate = nn.Linear(
            SphericalRayFeatureLifting.num_sh_basis,
            self.num_heads,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.ray_gate.weight)

    def forward(
        self,
        rosa_attention: Tensor,
        camera_rays: Tensor,
        valid: Tensor,
    ) -> Tensor:
        if rosa_attention.ndim != 6:
            raise ValueError(
                "ROSA attention must be [B,Q,P,C,L,H] for camera-ray RFL"
            )
        batch, queries, points, cameras, _levels, heads = rosa_attention.shape
        if heads != self.num_heads:
            raise ValueError("ROSA attention head count does not match ray RFL")
        expected_rays = (batch, queries, points, cameras, 3)
        if tuple(camera_rays.shape) != expected_rays:
            raise ValueError(
                f"camera rays must be {expected_rays}, got {tuple(camera_rays.shape)}"
            )
        if tuple(valid.shape) != expected_rays[:-1] or valid.dtype is not torch.bool:
            raise ValueError("camera-ray validity must be boolean [B,Q,P,C]")

        basis = SphericalRayFeatureLifting.real_sh_degree_2(camera_rays)
        gate = self.ray_gate(basis.to(dtype=self.ray_gate.weight.dtype)).to(
            dtype=rosa_attention.dtype
        )
        gate = torch.where(valid[..., None], gate, torch.zeros_like(gate))
        gate = gate.unsqueeze(4)

        mass = rosa_attention.sum(dim=(3, 4), keepdim=True)
        mean_gate = (rosa_attention * gate).sum(
            dim=(3, 4), keepdim=True
        ) / mass.clamp_min(torch.finfo(rosa_attention.dtype).tiny)
        centred_gate = gate - mean_gate
        effective = rosa_attention * (1.0 + centred_gate)
        return torch.where(
            valid[..., None, None],
            effective,
            torch.zeros_like(effective),
        )


class CameraRayLowRankResidual(nn.Module):
    """Direct camera-appearance by viewing-ray interaction for fused DAF.

    The ordinary ROSA/DAF result remains an untouched base path.  This branch
    uses signed, ray-conditioned DAF weights to aggregate a second appearance
    carrier, then maps it through a compact rank-16 linear bottleneck.  The
    final projection is zero initialized, making the complete module an exact
    identity at reset without a hand-selected residual scale or gate.

    The constant degree-zero SH term is omitted deliberately: every residual
    weight must depend on viewing direction instead of duplicating the base
    appearance aggregation.
    """

    rank = SphericalRayFeatureLifting.compact_factor_dims
    directional_sh_dims = SphericalRayFeatureLifting.num_sh_basis - 1

    def __init__(self, embed_dims: int, num_heads: int) -> None:
        super().__init__()
        if int(embed_dims) <= 0 or int(num_heads) <= 0:
            raise ValueError("low-rank camera RFL dimensions must be positive")
        if int(embed_dims) % int(num_heads) != 0:
            raise ValueError("camera RFL width must be divisible by its heads")
        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.ray_proj = nn.Linear(
            self.directional_sh_dims,
            self.num_heads,
            bias=False,
        )
        self.feature_norm = nn.LayerNorm(
            self.embed_dims,
            elementwise_affine=False,
        )
        self.appearance_proj = nn.Linear(
            self.embed_dims,
            self.rank,
            bias=False,
        )
        self.output_proj = nn.Linear(
            self.rank,
            self.embed_dims,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.ray_proj.weight)
        nn.init.xavier_uniform_(self.appearance_proj.weight)
        nn.init.zeros_(self.output_proj.weight)

    def signed_weights(
        self,
        rosa_attention: Tensor,
        camera_rays: Tensor,
        valid: Tensor,
    ) -> Tensor:
        if rosa_attention.ndim != 6:
            raise ValueError("ROSA attention must be [B,Q,P,C,L,H]")
        batch, queries, points, cameras, _levels, heads = rosa_attention.shape
        if heads != self.num_heads:
            raise ValueError("ROSA attention head count does not match camera RFL")
        expected_rays = (batch, queries, points, cameras, 3)
        if tuple(camera_rays.shape) != expected_rays:
            raise ValueError(
                f"camera rays must be {expected_rays}, got {tuple(camera_rays.shape)}"
            )
        if tuple(valid.shape) != expected_rays[:-1] or valid.dtype is not torch.bool:
            raise ValueError("camera-ray validity must be boolean [B,Q,P,C]")

        basis = SphericalRayFeatureLifting.real_sh_degree_2(camera_rays)[..., 1:]
        ray_response = self.ray_proj(
            basis.to(dtype=self.ray_proj.weight.dtype)
        ).to(dtype=rosa_attention.dtype)
        ray_response = torch.where(
            valid[..., None],
            ray_response,
            torch.zeros_like(ray_response),
        )
        return rosa_attention * ray_response.unsqueeze(4)

    def forward(self, ray_aggregated_features: Tensor) -> Tensor:
        if (
            ray_aggregated_features.ndim != 4
            or ray_aggregated_features.shape[-1] != self.embed_dims
        ):
            raise ValueError("camera RFL features must be [B,Q,P,embed_dims]")
        normalized = self.feature_norm(ray_aggregated_features.float())
        interaction = self.appearance_proj(normalized)
        residual = self.output_proj(interaction)
        return residual.to(dtype=ray_aggregated_features.dtype)


class CameraWiseRFLBeforeROSA(nn.Module):
    """Make each camera observation ray-aware before camera fusion.

    DAF has already reduced the FPN levels *within* each camera when this
    module is called, but no camera has yet been fused with another.  RFL can
    therefore form a genuine appearance-by-viewing-ray residual for every
    point-camera pair.  That residual changes both the value passed onward and
    the ROSA camera logit.  RFL's final projection is zero initialized, so the
    complete branch is an exact identity at initialization without a manual
    gate or residual scale.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attention_dims: int,
    ) -> None:
        super().__init__()
        if attention_dims % num_heads != 0:
            raise ValueError("RFL-before-ROSA attention width must divide heads")
        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.key_head_dims = int(attention_dims) // int(num_heads)
        self.rfl = SphericalRayFeatureLifting.for_compact_profile(embed_dims)
        self.key_proj = nn.Linear(embed_dims, attention_dims, bias=False)
        self.reliability = nn.Linear(embed_dims, num_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.rfl.reset_parameters()
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.reliability.weight)

    def forward(
        self,
        camera_features: Tensor,
        camera_rays: Tensor,
        valid: Tensor,
        query_heads: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if camera_features.ndim != 5:
            raise ValueError("camera features must be [B,Q,P,C,D]")
        batch, queries, points, cameras, dims = camera_features.shape
        expected_rays = (batch, queries, points, cameras, 3)
        if dims != self.embed_dims or tuple(camera_rays.shape) != expected_rays:
            raise ValueError("camera RFL feature/ray dimensions do not match")
        if tuple(valid.shape) != expected_rays[:-1] or valid.dtype is not torch.bool:
            raise ValueError("camera validity must be boolean [B,Q,P,C]")
        expected_query = (batch, queries, self.num_heads, self.key_head_dims)
        if tuple(query_heads.shape) != expected_query:
            raise ValueError("query heads do not match camera RFL attention")

        # RFL's public layout is [B,N,C,P,D].  Cameras remain separate here;
        # the permutation is a view and does not read the FPN a second time.
        features_for_rfl = camera_features.permute(0, 1, 3, 2, 4)
        rays_for_rfl = camera_rays.permute(0, 1, 3, 2, 4)
        valid_for_rfl = valid.permute(0, 1, 3, 2)
        residual = self.rfl(
            features_for_rfl,
            RayFeatureRelation(ray=rays_for_rfl, valid=valid_for_rfl),
        ).permute(0, 1, 3, 2, 4)
        residual = residual.to(dtype=camera_features.dtype)
        ray_aware_features = camera_features + residual

        keys = self.key_proj(residual).reshape(
            batch,
            queries,
            points,
            cameras,
            self.num_heads,
            self.key_head_dims,
        )
        delta_logits = torch.einsum(
            "bqhd,bqpchd->bqpch",
            query_heads,
            keys,
        ) / math.sqrt(self.key_head_dims)
        delta_logits = delta_logits + self.reliability(residual)
        delta_logits = torch.where(
            valid[..., None],
            delta_logits,
            torch.zeros_like(delta_logits),
        )
        return ray_aware_features, delta_logits


@dataclass(frozen=True)
class FisheyeBoxDecoderOutput:
    classification: tuple[Tensor, ...]
    prediction: tuple[Tensor, ...]
    quality: tuple[Tensor, ...]
    states: tuple[RayState, ...]
    initializer_state: RayState
    initializer_gate: Tensor
    camera_level_attention: tuple[Tensor, ...]
    point_attention: tuple[Tensor, ...]
    level_compatibility: tuple[Tensor, ...]
    query: Tensor


def _reshape_point_geometry(
    geometry: FisheyeObservationGeometry,
    batch: int,
    queries: int,
    points: int,
    cameras: int,
) -> FisheyeObservationGeometry:
    prefix = (batch, queries, points, cameras)
    return FisheyeObservationGeometry(
        rays=geometry.rays.reshape(prefix + (3,)),
        tangent=geometry.tangent.reshape(prefix + (2,)),
        jacobian=geometry.jacobian.reshape(prefix + (2, 2)),
        lens_margin=geometry.lens_margin.reshape(prefix + (1,)),
        overlap=geometry.overlap.reshape(prefix + (1,)),
        valid=geometry.valid.reshape(prefix),
    )


def project_box_keypoints(
    projector: NativeFisheyeProjector,
    keypoints: Tensor,
    metadata: dict,
) -> BoxPointProjection:
    """Project metric ``[B,Q,P,3]`` support points without pinhole warping."""

    if keypoints.ndim != 4 or keypoints.shape[-1] != 3:
        raise ValueError("keypoints must be [B,Q,P,3]")
    batch, queries, points = keypoints.shape[:3]
    flat_points = keypoints.reshape(batch, queries * points, 3)
    flat = projector(
        normalize_direction(flat_points),
        metadata,
        points=flat_points,
    )
    cameras = flat.uv.shape[2]
    return BoxPointProjection(
        uv=flat.uv.reshape(batch, queries, points, cameras, 2),
        geometry=_reshape_point_geometry(
            flat.geometry,
            batch,
            queries,
            points,
            cameras,
        ),
    )


def sample_box_keypoints_fpn(
    fpn: Sequence[Tensor],
    uv: Tensor,
    *,
    sampler: nn.Module,
) -> Tensor:
    """Sample all box points and return ``[B,Q,P,C,L,D]``."""

    if uv.ndim != 5 or uv.shape[-1] != 2:
        raise ValueError("uv must be [B,Q,P,C,2]")
    batch, queries, points, cameras = uv.shape[:4]
    flat = sample_native_fisheye_fpn(
        fpn,
        uv.reshape(batch, queries * points, cameras, 2),
        sampler=sampler,
    )
    return flat.reshape(
        batch,
        queries,
        points,
        cameras,
        flat.shape[3],
        flat.shape[4],
    )


class FisheyeBoxKeypointGenerator(nn.Module):
    """Generate seven fixed and optional learned points from a ray box state."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_learnable_points: int = 6,
        fixed_scale: float = 0.45,
    ) -> None:
        super().__init__()
        if embed_dims <= 0 or num_learnable_points < 0:
            raise ValueError("invalid keypoint generator dimensions")
        if not 0.0 < fixed_scale <= 0.5:
            raise ValueError("fixed_scale must lie in (0, 0.5]")
        self.embed_dims = int(embed_dims)
        self.num_learnable_points = int(num_learnable_points)
        fixed = torch.tensor(
            (
                (0.0, 0.0, 0.0),
                (fixed_scale, 0.0, 0.0),
                (-fixed_scale, 0.0, 0.0),
                (0.0, fixed_scale, 0.0),
                (0.0, -fixed_scale, 0.0),
                (0.0, 0.0, fixed_scale),
                (0.0, 0.0, -fixed_scale),
            ),
            dtype=torch.float32,
        )
        self.register_buffer("fixed_scales", fixed, persistent=True)
        self.learnable_fc = (
            nn.Linear(embed_dims, num_learnable_points * 3)
            if num_learnable_points > 0
            else None
        )
        self.reset_parameters()

    @property
    def num_points(self) -> int:
        return int(self.fixed_scales.shape[0]) + self.num_learnable_points

    def reset_parameters(self) -> None:
        if self.learnable_fc is not None:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            nn.init.zeros_(self.learnable_fc.bias)

    def forward(self, query: Tensor, state: RayState) -> Tensor:
        state.validate()
        if query.ndim != 3 or query.shape[-1] != self.embed_dims:
            raise ValueError("query must be [B,Q,embed_dims]")
        if query.shape[:2] != state.direction.shape[:2]:
            raise ValueError("query and state must share [B,Q]")
        batch, queries = query.shape[:2]
        size = state.log_size.exp()
        scales = self.fixed_scales.to(device=query.device, dtype=query.dtype)
        scales = scales.view(1, 1, -1, 3).expand(batch, queries, -1, -1)
        if self.learnable_fc is not None:
            learned = (
                self.learnable_fc(query.float())
                .to(query.dtype)
                .reshape(batch, queries, self.num_learnable_points, 3)
                .sigmoid()
                - 0.5
            )
            scales = torch.cat((scales, learned), dim=2)
        local = scales * size[:, :, None, :]

        anchor = state.to_sparse_anchor()
        sin_yaw, cos_yaw = F.normalize(
            anchor[..., 6:8], dim=-1, eps=1e-6
        ).unbind(dim=-1)
        x, y, z = local.unbind(dim=-1)
        rotated = torch.stack(
            (
                cos_yaw[:, :, None] * x - sin_yaw[:, :, None] * y,
                sin_yaw[:, :, None] * x + cos_yaw[:, :, None] * y,
                z,
            ),
            dim=-1,
        )
        return state.center[:, :, None, :] + rotated


def _blend_ray_states(base: RayState, candidate: RayState, gate: Tensor) -> RayState:
    base.validate()
    candidate.validate()
    if gate.shape != base.log_range.shape:
        raise ValueError("gate must be [B,Q,1]")
    gate = gate.clamp(0.0, 1.0)
    direction = normalize_direction(
        base.direction + gate * (candidate.direction - base.direction)
    )
    yaw = F.normalize(base.yaw + gate * (candidate.yaw - base.yaw), dim=-1, eps=1e-6)
    return RayState(
        direction=direction,
        log_range=base.log_range + gate * (candidate.log_range - base.log_range),
        log_range_std=(
            base.log_range_std
            + gate * (candidate.log_range_std - base.log_range_std)
        ),
        log_size=base.log_size + gate * (candidate.log_size - base.log_size),
        yaw=yaw,
        velocity=base.velocity + gate * (candidate.velocity - base.velocity),
    ).validate()


class RayGuidedBoxInitializer(nn.Module):
    """Correct a learned 3D anchor using one calibrated centre observation."""

    def __init__(
        self,
        config: FisheyeBoxDecoderConfig,
        projector: NativeFisheyeProjector,
        sampler: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        self.projector = projector
        self.sampler = sampler
        evidence_cfg = FisheyeEvidenceConfig(
            embed_dims=config.embed_dims,
            num_heads=config.num_heads,
            attention_dims=config.attention_dims,
            num_cameras=config.num_cameras,
            level_strides=config.level_strides,
            attention_dropout=config.dropout,
            projection_dropout=config.dropout,
            predict_angular_extent=False,
        )
        self.evidence = UnifiedFisheyeEvidenceAttention(evidence_cfg)
        self.norm = nn.LayerNorm(config.embed_dims)
        self.refinement = FisheyeRayStateRefinement(
            RayRefinementConfig(
                embed_dims=config.embed_dims,
                hidden_dims=config.embed_dims,
                min_range_m=config.min_range_m,
                max_range_m=config.max_range_m,
            ),
            initialize=False,
        )

    @staticmethod
    def _observation_gate(
        projection: NativeFisheyeProjection,
        attention: Tensor,
    ) -> Tensor:
        valid = projection.geometry.valid
        typed_valid = valid.to(attention.dtype)
        valid_fraction = typed_valid.mean(dim=2, keepdim=True)
        valid_count = typed_valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        margin = (
            projection.geometry.lens_margin.squeeze(-1) * typed_valid
        ).sum(dim=2, keepdim=True) / valid_count
        cameras, levels = attention.shape[2:4]
        uniform = 1.0 / float(cameras * levels)
        concentration = attention.flatten(2, 3).amax(dim=2).mean(dim=-1, keepdim=True)
        concentration = ((concentration - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
        gate = valid_fraction * margin.clamp(0.0, 1.0) * (0.5 + 0.5 * concentration)
        return torch.where(valid.any(dim=2, keepdim=True), gate, torch.zeros_like(gate))

    def forward(
        self,
        query: Tensor,
        state: RayState,
        fpn: Sequence[Tensor],
        metadata: dict,
    ) -> RayGuidedInitializationOutput:
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
            log_angular_extent=state_log_angular_extent(state),
        )
        query = self.norm(evidence.query)
        candidate = self.refinement(query, state.direction, state)
        gate = self._observation_gate(projection, evidence.attention)
        initialized = _blend_ray_states(state, candidate, gate)
        return RayGuidedInitializationOutput(
            query=query,
            state=initialized,
            gate=gate,
            projection=projection,
            camera_level_attention=evidence.attention,
        )


class FisheyeBoxEvidenceAttention(nn.Module):
    """Factorized ROSA camera-level and semantic-point evidence attention."""

    geometry_dims = 9

    def __init__(
        self,
        config: FisheyeBoxDecoderConfig,
        *,
        center_prefusion: bool = False,
        return_signed_faces: bool | None = None,
        return_symmetric_faces: bool = False,
        camera_ray_low_rank_rfl: bool | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.center_prefusion = bool(center_prefusion)
        self.key_head_dims = config.attention_dims // config.num_heads
        self.value_head_dims = config.embed_dims // config.num_heads
        self.register_buffer(
            "level_strides",
            torch.tensor(config.level_strides, dtype=torch.float32),
            persistent=True,
        )
        self.query_norm = nn.LayerNorm(config.embed_dims)
        self.feature_norm = nn.LayerNorm(config.embed_dims)
        self.point_norm = nn.LayerNorm(config.embed_dims)
        self.query_proj = nn.Linear(config.embed_dims, config.attention_dims, bias=False)
        self.key_proj = (
            None
            if config.use_fused_daf
            else nn.Linear(config.embed_dims, config.attention_dims, bias=False)
        )
        self.geometry_key = nn.Sequential(
            nn.Linear(self.geometry_dims, 64),
            nn.SiLU(),
            nn.Linear(64, config.attention_dims, bias=False),
        )
        self.geometry_reliability = nn.Sequential(
            nn.Linear(self.geometry_dims, 64),
            nn.SiLU(),
            nn.Linear(64, config.num_heads),
        )
        self.feature_reliability = (
            None
            if config.use_fused_daf
            else nn.Linear(config.embed_dims, config.num_heads, bias=False)
        )
        self.rfl = SphericalRayFeatureLifting.for_compact_profile(
            embed_dims=config.embed_dims
        )
        self.camera_ray_rfl = (
            CameraRaySHValueModulation(config.num_heads)
            if config.camera_ray_rfl
            else None
        )
        use_low_rank_camera_rfl = (
            config.camera_ray_low_rank_rfl
            if camera_ray_low_rank_rfl is None
            else bool(camera_ray_low_rank_rfl)
        )
        self.camera_ray_low_rank_rfl = (
            CameraRayLowRankResidual(config.embed_dims, config.num_heads)
            if use_low_rank_camera_rfl
            else None
        )
        # Keep the V42 parameter initialization stream unchanged.  This makes
        # the new experiment differ only by the explicitly enabled branch.
        if getattr(config, "camera_rfl_before_rosa", False):
            with torch.random.fork_rng(devices=[]):
                self.camera_rfl_before_rosa = CameraWiseRFLBeforeROSA(
                    config.embed_dims,
                    config.num_heads,
                    config.attention_dims,
                )
        else:
            self.camera_rfl_before_rosa = None
        self.point_embedding = nn.Parameter(
            torch.empty(config.num_points, config.embed_dims)
        )
        self.point_query_proj = nn.Linear(
            config.embed_dims, config.attention_dims, bias=False
        )
        self.point_key_proj = nn.Linear(
            config.embed_dims, config.attention_dims, bias=False
        )
        self.point_reliability = nn.Linear(
            config.embed_dims, config.num_heads, bias=False
        )
        self.output_proj = nn.Linear(config.embed_dims, config.embed_dims)
        # Zero initialization makes the optional signed path an exact identity
        # at reset; the existing output projection supplies channel mixing.
        self.face_contrast_weight = (
            nn.Parameter(torch.zeros(3, config.embed_dims))
            if config.structured_face_contrast
            else None
        )
        self.return_signed_faces = (
            bool(getattr(config, "structured_face_refinement", False))
            if return_signed_faces is None
            else bool(return_signed_faces)
        )
        self.return_symmetric_faces = bool(return_symmetric_faces)
        if self.center_prefusion:
            self.center_residual_proj = nn.Linear(
                config.embed_dims, config.embed_dims
            )
            self.center_query_norm = nn.LayerNorm(config.embed_dims)
        self.dropout = nn.Dropout(config.dropout)
        self.scale_match_raw = nn.Parameter(torch.zeros(config.num_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.query_proj,
            self.key_proj,
            self.feature_reliability,
            self.point_query_proj,
            self.point_key_proj,
            self.point_reliability,
            self.output_proj,
            getattr(self, "center_residual_proj", None),
        ):
            if module is not None:
                nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.output_proj.bias)
        if self.face_contrast_weight is not None:
            nn.init.zeros_(self.face_contrast_weight)
        if self.center_prefusion:
            nn.init.zeros_(self.center_residual_proj.bias)
        nn.init.normal_(self.point_embedding, std=0.02)
        for sequence in (self.geometry_key, self.geometry_reliability):
            for module in sequence.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.rfl.reset_parameters()
        if self.camera_ray_rfl is not None:
            self.camera_ray_rfl.reset_parameters()
        if self.camera_ray_low_rank_rfl is not None:
            self.camera_ray_low_rank_rfl.reset_parameters()
        if self.camera_rfl_before_rosa is not None:
            with torch.random.fork_rng(devices=[]):
                self.camera_rfl_before_rosa.reset_parameters()

    def _prefuse_center(
        self,
        query: Tensor,
        point_features: Tensor,
        point_valid: Tensor,
    ) -> Tensor:
        """Condition D0 on its existing centre token without another FPN read."""

        if not self.center_prefusion:
            return query
        center_feature = self.point_norm(point_features[:, :, 0])
        candidate = self.center_query_norm(
            query + self.dropout(self.center_residual_proj(center_feature))
        )
        center_valid = point_valid[:, :, 0, None]
        return torch.where(center_valid, candidate, query)

    @staticmethod
    def _masked_softmax(logits: Tensor, valid: Tensor, dim: int) -> Tensor:
        masked = torch.where(
            valid,
            logits,
            torch.full_like(logits, -torch.finfo(logits.dtype).max),
        )
        weights = torch.softmax(masked.float(), dim=dim).to(logits.dtype)
        weights = weights * valid.to(weights.dtype)
        denominator = weights.sum(dim=dim, keepdim=True)
        return torch.where(
            denominator > 0.0,
            weights / denominator.clamp_min(1e-8),
            torch.zeros_like(weights),
        )

    def _camera_level_evidence(
        self,
        query: Tensor,
        sampled: Tensor,
        geometry: FisheyeObservationGeometry,
        log_angular_extent: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        if query.ndim != 3 or query.shape[-1] != cfg.embed_dims:
            raise ValueError("query must be [B,Q,embed_dims]")
        batch, queries = query.shape[:2]
        if sampled.ndim != 6:
            raise ValueError("sampled must be [B,Q,P,C,L,D]")
        points = sampled.shape[2]
        levels = len(cfg.level_strides)
        expected = (
            batch,
            queries,
            points,
            cfg.num_cameras,
            levels,
            cfg.embed_dims,
        )
        if tuple(sampled.shape) != expected:
            raise ValueError(f"sampled shape {tuple(sampled.shape)} != {expected}")
        geometry_prefix = (batch, queries, points, cfg.num_cameras)
        geometry_shapes = {
            "rays": geometry_prefix + (3,),
            "tangent": geometry_prefix + (2,),
            "jacobian": geometry_prefix + (2, 2),
            "lens_margin": geometry_prefix + (1,),
            "overlap": geometry_prefix + (1,),
            "valid": geometry_prefix,
        }
        for name, shape in geometry_shapes.items():
            value = getattr(geometry, name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                observed = getattr(value, "shape", None)
                raise ValueError(
                    f"point geometry {name} shape {observed} does not match {shape}"
                )
        if geometry.valid.dtype is not torch.bool:
            raise TypeError("point geometry validity must be boolean")
        if log_angular_extent.shape != (batch, queries, 2):
            raise ValueError("log_angular_extent must be [B,Q,2]")

        flat_prefix = (batch, queries * points, cfg.num_cameras)
        descriptor, compatibility = angular_fpn_geometry(
            geometry.tangent.reshape(flat_prefix + (2,)),
            geometry.jacobian.reshape(flat_prefix + (2, 2)),
            geometry.lens_margin.reshape(flat_prefix + (1,)),
            geometry.overlap.reshape(flat_prefix + (1,)),
            geometry.valid.reshape(flat_prefix),
            self.level_strides,
            log_angular_extent[:, :, None, :]
            .expand(-1, -1, points, -1)
            .reshape(batch, queries * points, 2),
        )
        descriptor = descriptor.reshape(
            batch, queries, points, cfg.num_cameras, levels, self.geometry_dims
        )
        compatibility = compatibility.reshape(
            batch, queries, points, cfg.num_cameras, levels
        )
        valid = geometry.valid[..., None].expand(
            batch, queries, points, cfg.num_cameras, levels
        )
        finite = torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)

        rfl_features = finite.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            queries,
            cfg.num_cameras,
            points * levels,
            cfg.embed_dims,
        )
        rfl_rays = (
            geometry.rays[..., None, :]
            .expand(batch, queries, points, cfg.num_cameras, levels, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, queries, cfg.num_cameras, points * levels, 3)
        )
        rfl_valid = valid.permute(0, 1, 3, 2, 4).reshape(
            batch, queries, cfg.num_cameras, points * levels
        )
        lifted = rfl_features + self.rfl(
            rfl_features,
            RayFeatureRelation(ray=rfl_rays, valid=rfl_valid),
        )
        lifted = lifted.reshape(
            batch,
            queries,
            cfg.num_cameras,
            points,
            levels,
            cfg.embed_dims,
        ).permute(0, 1, 3, 2, 4, 5)

        normalized_query = self.query_norm(query)
        normalized_features = self.feature_norm(finite)
        heads = cfg.num_heads
        key_dims = self.key_head_dims
        q = self.query_proj(normalized_query).reshape(
            batch, queries, heads, key_dims
        )
        k = (
            self.key_proj(normalized_features)
            + self.geometry_key(descriptor).to(normalized_features.dtype)
        ).reshape(
            batch,
            queries,
            points,
            cfg.num_cameras,
            levels,
            heads,
            key_dims,
        )
        logits = torch.einsum("bqhd,bqpclhd->bqpclh", q, k) / math.sqrt(key_dims)
        logits = logits + self.geometry_reliability(descriptor)
        logits = logits + self.feature_reliability(normalized_features)
        scale = F.softplus(self.scale_match_raw).view(1, 1, 1, 1, 1, heads)
        logits = logits + compatibility[..., None] * scale
        flat_logits = logits.flatten(3, 4)
        flat_valid = valid.flatten(3, 4)[..., None]
        camera_level_attention = self._masked_softmax(
            flat_logits,
            flat_valid,
            dim=3,
        )
        value_attention = camera_level_attention.reshape(
            batch,
            queries,
            points,
            cfg.num_cameras,
            levels,
            heads,
        )
        if self.camera_ray_rfl is not None:
            value_attention = self.camera_ray_rfl(
                value_attention,
                geometry.rays,
                geometry.valid,
            )
        value_attention = value_attention.flatten(3, 4)

        values = self.feature_norm(lifted).reshape(
            batch,
            queries,
            points,
            cfg.num_cameras * levels,
            heads,
            self.value_head_dims,
        )
        point_features = torch.einsum(
            "bqpsh,bqpshd->bqphd",
            value_attention,
            values,
        ).reshape(batch, queries, points, cfg.embed_dims)
        return (
            point_features,
            camera_level_attention.reshape(
                batch,
                queries,
                points,
                cfg.num_cameras,
                levels,
                heads,
            ),
            compatibility,
            geometry.valid.any(dim=3),
        )

    def _fuse_points(
        self,
        query: Tensor,
        point_features: Tensor,
        point_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        cfg = self.config
        batch, queries = query.shape[:2]
        points = point_features.shape[2]
        if point_features.shape != (batch, queries, cfg.num_points, cfg.embed_dims):
            raise ValueError("point fusion requires the complete configured point set")
        if point_valid.shape != (batch, queries, points):
            raise ValueError("point_valid must be [B,Q,P]")
        heads = cfg.num_heads
        key_dims = self.key_head_dims
        normalized_query = self.query_norm(query)
        point_tokens = self.point_norm(
            point_features
            + self.point_embedding[None, None].to(point_features.dtype)
        )
        point_q = self.point_query_proj(normalized_query).reshape(
            batch, queries, heads, key_dims
        )
        point_k = self.point_key_proj(point_tokens).reshape(
            batch, queries, points, heads, key_dims
        )
        point_logits = torch.einsum("bqhd,bqphd->bqph", point_q, point_k)
        point_logits = point_logits / math.sqrt(key_dims)
        point_logits = point_logits + self.point_reliability(point_tokens)
        point_attention = self._masked_softmax(
            point_logits,
            point_valid[..., None],
            dim=2,
        )
        point_values = self.point_norm(point_features).reshape(
            batch, queries, points, heads, self.value_head_dims
        )
        evidence = torch.einsum(
            "bqph,bqphd->bqhd",
            point_attention,
            point_values,
        ).reshape(batch, queries, cfg.embed_dims)
        signed_faces = None
        if self.face_contrast_weight is not None or self.return_signed_faces:
            # Fixed points 1:7 are (+x,-x,+y,-y,+z,-z).  Unlike another
            # sampler or attention block, these three signed differences
            # retain directional box-face evidence already produced by the
            # calibrated ROSA/RFL path.  A pair contributes only when both
            # projected endpoints are supported.
            normalized_points = point_values.reshape(
                batch, queries, points, cfg.embed_dims
            )
            positive = normalized_points[:, :, (1, 3, 5)]
            negative = normalized_points[:, :, (2, 4, 6)]
            pair_valid = (
                point_valid[:, :, (1, 3, 5)]
                & point_valid[:, :, (2, 4, 6)]
            )
            signed_faces = (positive - negative) * pair_valid[..., None]
        if self.face_contrast_weight is not None:
            assert signed_faces is not None
            evidence = evidence + (
                signed_faces
                * self.face_contrast_weight[None, None].to(signed_faces.dtype)
            ).sum(dim=2)
        updated = query + self.dropout(self.output_proj(evidence))
        return updated, point_attention, signed_faces

    def _symmetric_face_features(
        self,
        point_features: Tensor,
        point_valid: Tensor,
    ) -> Tensor | None:
        """Return face-pair evidence that is invariant to endpoint order."""

        if not self.return_symmetric_faces:
            return None
        batch, queries, points = point_features.shape[:3]
        if points < 7 or point_valid.shape != (batch, queries, points):
            raise ValueError("symmetric faces require the seven fixed box points")
        normalized = self.point_norm(point_features)
        center = normalized[:, :, 0:1]
        positive = normalized[:, :, (1, 3, 5)]
        negative = normalized[:, :, (2, 4, 6)]
        pair_valid = (
            point_valid[:, :, (1, 3, 5)]
            & point_valid[:, :, (2, 4, 6)]
            & point_valid[:, :, 0:1]
        )
        symmetric = 0.5 * (positive + negative) - center
        return symmetric * pair_valid[..., None]

    def forward(
        self,
        query: Tensor,
        sampled: Tensor,
        geometry: FisheyeObservationGeometry,
        log_angular_extent: Tensor,
    ) -> FisheyeBoxEvidenceOutput:
        (
            point_features,
            camera_level_attention,
            compatibility,
            point_valid,
        ) = self._camera_level_evidence(
            query,
            sampled,
            geometry,
            log_angular_extent,
        )
        query = self._prefuse_center(query, point_features, point_valid)
        updated, point_attention, signed_faces = self._fuse_points(
            query,
            point_features,
            point_valid,
        )
        symmetric_faces = self._symmetric_face_features(
            point_features,
            point_valid,
        )
        return FisheyeBoxEvidenceOutput(
            query=updated,
            camera_level_attention=camera_level_attention,
            point_attention=point_attention,
            level_compatibility=compatibility,
            supported=point_valid.any(dim=2),
            signed_face_features=signed_faces,
            symmetric_face_features=symmetric_faces,
        )

    def forward_fused_daf(
        self,
        query: Tensor,
        projection: BoxPointProjection,
        log_angular_extent: Tensor,
        point_rays: Tensor,
        formatted_fpn: tuple[Tensor, Tensor, Tensor],
    ) -> FisheyeBoxEvidenceOutput:
        """Fuse camera/level evidence for each point in one Sparse4D CUDA read.

        Every box point is presented to DAF as an independent one-keypoint
        sparse anchor.  DAF therefore performs only bilinear feature reading
        and grouped weighted reduction; V3 still owns the native-fisheye
        projection, calibrated ROSA logits, per-point RFL, and semantic point
        attention.
        """

        try:
            from projects.mmdet3d_plugin.ops import (
                deformable_aggregation_function,
            )
            if self.camera_rfl_before_rosa is not None:
                from projects.mmdet3d_plugin.ops import (
                    deformable_aggregation_per_camera_function,
                )
        except ImportError as exc:  # pragma: no cover - host-only path
            raise RuntimeError("fused V3 sampling requires the Sparse4D CUDA host") from exc

        cfg = self.config
        batch, queries, points, cameras = projection.uv.shape[:4]
        levels = len(cfg.level_strides)
        if query.shape != (batch, queries, cfg.embed_dims):
            raise ValueError("fused query shape does not match projection")
        if points != cfg.num_points or cameras != cfg.num_cameras:
            raise ValueError("fused projection does not match decoder config")
        if point_rays.shape != (batch, queries, points, 3):
            raise ValueError("point_rays must be [B,Q,P,3]")
        if log_angular_extent.shape != (batch, queries, 2):
            raise ValueError("log_angular_extent must be [B,Q,2]")

        geometry = projection.geometry
        flat_prefix = (batch, queries * points, cameras)
        descriptor, compatibility = angular_fpn_geometry(
            geometry.tangent.reshape(flat_prefix + (2,)),
            geometry.jacobian.reshape(flat_prefix + (2, 2)),
            geometry.lens_margin.reshape(flat_prefix + (1,)),
            geometry.overlap.reshape(flat_prefix + (1,)),
            geometry.valid.reshape(flat_prefix),
            self.level_strides,
            log_angular_extent[:, :, None, :]
            .expand(-1, -1, points, -1)
            .reshape(batch, queries * points, 2),
        )
        descriptor = descriptor.reshape(
            batch, queries, points, cameras, levels, self.geometry_dims
        )
        compatibility = compatibility.reshape(
            batch, queries, points, cameras, levels
        )
        valid = geometry.valid[..., None].expand(
            batch, queries, points, cameras, levels
        )

        heads = cfg.num_heads
        key_dims = self.key_head_dims
        normalized_query = self.query_norm(query)
        q = self.query_proj(normalized_query).reshape(
            batch, queries, heads, key_dims
        )
        geometry_key = self.geometry_key(descriptor).reshape(
            batch, queries, points, cameras, levels, heads, key_dims
        )
        logits = torch.einsum(
            "bqhd,bqpclhd->bqpclh",
            q,
            geometry_key,
        ) / math.sqrt(key_dims)
        logits = logits + self.geometry_reliability(descriptor)
        scale = F.softplus(self.scale_match_raw).view(1, 1, 1, 1, 1, heads)
        logits = logits + compatibility[..., None] * scale
        col_feats, spatial_shape, scale_start_index = formatted_fpn
        sampling_location = (
            projection.uv.reshape(batch, queries * points, cameras, 2)
            .unsqueeze(2)
            .contiguous()
        )
        camera_ray_residual = None
        if self.camera_rfl_before_rosa is not None:
            # Factor the original joint camera x level softmax exactly as
            # p(camera, level) = p(level | camera) p(camera).  DAF performs
            # the conditional level reduction once and returns all cameras;
            # RFL then conditions the camera marginal before camera fusion.
            conditional_level_attention = self._masked_softmax(
                logits,
                valid[..., None],
                dim=4,
            )
            masked_logits = torch.where(
                valid[..., None],
                logits,
                torch.full_like(logits, -torch.inf),
            )
            camera_logits = torch.logsumexp(masked_logits.float(), dim=4).to(
                dtype=logits.dtype
            )
            grouped_weights = conditional_level_attention.reshape(
                batch, queries * points, 1, cameras, levels, heads
            ).contiguous()

            _unused_fused, camera_features = (
                deformable_aggregation_per_camera_function(
                    col_feats,
                    spatial_shape,
                    scale_start_index,
                    sampling_location,
                    grouped_weights,
                )
            )
            camera_features = camera_features.reshape(
                batch, queries, points, cameras, cfg.embed_dims
            )
            camera_features, camera_delta = self.camera_rfl_before_rosa(
                camera_features,
                geometry.rays,
                geometry.valid,
                q,
            )
            camera_attention = self._masked_softmax(
                camera_logits + camera_delta,
                geometry.valid[..., None],
                dim=3,
            )
            camera_values = camera_features.reshape(
                batch,
                queries,
                points,
                cameras,
                heads,
                self.value_head_dims,
            )
            point_features = torch.einsum(
                "bqpch,bqpchd->bqphd",
                camera_attention,
                camera_values,
            ).reshape(batch, queries, points, cfg.embed_dims)
            attention = (
                camera_attention.unsqueeze(4) * conditional_level_attention
            )
        else:
            attention = self._masked_softmax(
                logits.flatten(3, 4),
                valid.flatten(3, 4)[..., None],
                dim=3,
            ).reshape(batch, queries, points, cameras, levels, heads)
            value_attention = attention
            if self.camera_ray_rfl is not None:
                value_attention = self.camera_ray_rfl(
                    attention,
                    geometry.rays,
                    geometry.valid,
                )
            grouped_weights = value_attention.reshape(
                batch, queries * points, 1, cameras, levels, heads
            ).contiguous()
            point_features = deformable_aggregation_function(
                col_feats,
                spatial_shape,
                scale_start_index,
                sampling_location,
                grouped_weights,
            ).reshape(batch, queries, points, cfg.embed_dims)

            if self.camera_ray_low_rank_rfl is not None:
                ray_weights = self.camera_ray_low_rank_rfl.signed_weights(
                    attention,
                    geometry.rays,
                    geometry.valid,
                ).reshape(
                    batch, queries * points, 1, cameras, levels, heads
                ).contiguous()
                ray_aggregated = deformable_aggregation_function(
                    col_feats,
                    spatial_shape,
                    scale_start_index,
                    sampling_location,
                    ray_weights,
                ).reshape(batch, queries, points, cfg.embed_dims)
                camera_ray_residual = self.camera_ray_low_rank_rfl(ray_aggregated)

        point_valid = geometry.valid.any(dim=3)
        point_features = self.feature_norm(point_features)
        rfl_features = point_features[:, :, None, :, :]
        rfl_rays = normalize_direction(point_rays)[:, :, None, :, :]
        rfl_valid = point_valid[:, :, None, :]
        point_features = point_features + self.rfl(
            rfl_features,
            RayFeatureRelation(ray=rfl_rays, valid=rfl_valid),
        ).squeeze(2)
        if camera_ray_residual is not None:
            point_features = point_features + camera_ray_residual
        query = self._prefuse_center(query, point_features, point_valid)
        updated, point_attention, signed_faces = self._fuse_points(
            query,
            point_features,
            point_valid,
        )
        symmetric_faces = self._symmetric_face_features(
            point_features,
            point_valid,
        )
        return FisheyeBoxEvidenceOutput(
            query=updated,
            camera_level_attention=attention,
            point_attention=point_attention,
            level_compatibility=compatibility,
            supported=point_valid.any(dim=2),
            signed_face_features=signed_faces,
            symmetric_face_features=symmetric_faces,
        )

    def forward_streaming_training(
        self,
        query: Tensor,
        fpn: Sequence[Tensor],
        projection: BoxPointProjection,
        log_angular_extent: Tensor,
        sampler: nn.Module,
    ) -> FisheyeBoxEvidenceOutput:
        """Checkpoint point chunks so 13-point activations are never resident.

        Sampling is inside each checkpointed closure.  Backward therefore
        recomputes the deterministic gather and camera-level attention from
        the shared FPN inputs instead of retaining a full
        ``[B,Q,13,C,L,D]`` carrier for every decoder stage.
        """

        cfg = self.config
        batch, queries, points, cameras = projection.uv.shape[:4]
        if points != cfg.num_points or cameras != cfg.num_cameras:
            raise ValueError("streaming projection does not match decoder config")
        point_features = []
        point_valid = []
        chunk_size = min(cfg.point_chunk_size, points)
        for start in range(0, points, chunk_size):
            end = min(points, start + chunk_size)
            chunk_geometry = FisheyeObservationGeometry(
                rays=projection.geometry.rays[:, :, start:end],
                tangent=projection.geometry.tangent[:, :, start:end],
                jacobian=projection.geometry.jacobian[:, :, start:end],
                lens_margin=projection.geometry.lens_margin[:, :, start:end],
                overlap=projection.geometry.overlap[:, :, start:end],
                valid=projection.geometry.valid[:, :, start:end],
            )
            uv = projection.uv[:, :, start:end]

            def run_chunk(
                query_value,
                uv_value,
                rays,
                tangent,
                jacobian,
                lens_margin,
                overlap,
                valid,
                extent,
                *feature_levels,
            ):
                geometry_value = FisheyeObservationGeometry(
                    rays=rays,
                    tangent=tangent,
                    jacobian=jacobian,
                    lens_margin=lens_margin,
                    overlap=overlap,
                    valid=valid,
                )
                sampled_value = sample_box_keypoints_fpn(
                    feature_levels,
                    uv_value,
                    sampler=sampler,
                )
                fused, _, _, _ = self._camera_level_evidence(
                    query_value,
                    sampled_value,
                    geometry_value,
                    extent,
                )
                return fused

            tensor_inputs = (
                query,
                uv,
                chunk_geometry.rays,
                chunk_geometry.tangent,
                chunk_geometry.jacobian,
                chunk_geometry.lens_margin,
                chunk_geometry.overlap,
                chunk_geometry.valid,
                log_angular_extent,
                *fpn,
            )
            if cfg.checkpoint_point_chunks and torch.is_grad_enabled():
                fused = checkpoint(
                    run_chunk,
                    *tensor_inputs,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                fused = run_chunk(*tensor_inputs)
            point_features.append(fused)
            point_valid.append(chunk_geometry.valid.any(dim=3))

        point_features_value = torch.cat(point_features, dim=2)
        point_valid_value = torch.cat(point_valid, dim=2)
        query = self._prefuse_center(
            query,
            point_features_value,
            point_valid_value,
        )
        updated, point_attention, signed_faces = self._fuse_points(
            query,
            point_features_value,
            point_valid_value,
        )
        symmetric_faces = self._symmetric_face_features(
            point_features_value,
            point_valid_value,
        )
        # Training diagnostics deliberately omit the camera-level carrier.  It
        # is available on the ordinary eval path, while retaining it here
        # would defeat activation streaming.
        empty_attention = query.new_empty(
            batch, queries, points, 0, 0, cfg.num_heads
        )
        empty_compatibility = query.new_empty(batch, queries, points, 0, 0)
        return FisheyeBoxEvidenceOutput(
            query=updated,
            camera_level_attention=empty_attention,
            point_attention=point_attention,
            level_compatibility=empty_compatibility,
            supported=point_valid_value.any(dim=2),
            signed_face_features=signed_faces,
            symmetric_face_features=symmetric_faces,
        )


class FisheyeSparseBoxDecoderLayer(nn.Module):
    """One D0--D5 box observation and spherical refinement stage."""

    def __init__(
        self,
        config: FisheyeBoxDecoderConfig,
        projector: NativeFisheyeProjector,
        sampler: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        self.projector = projector
        self.sampler = sampler
        self.position_encoder = RayStatePositionEncoder(config.embed_dims)
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
        self.evidence = FisheyeBoxEvidenceAttention(config)
        self.evidence_norm = nn.LayerNorm(config.embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dims, config.ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dims, config.embed_dims),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.embed_dims)
        self.refinement = FisheyeRayStateRefinement(
            RayRefinementConfig(
                embed_dims=config.embed_dims,
                hidden_dims=config.embed_dims,
                min_range_m=config.min_range_m,
                max_range_m=config.max_range_m,
            ),
            initialize=False,
        )

    def forward(
        self,
        query: Tensor,
        state: RayState,
        fpn: Sequence[Tensor],
        metadata: dict,
        self_attention_mask: Tensor | None = None,
        formatted_fpn: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, RayState, FisheyeBoxEvidenceOutput, Tensor]:
        state.validate()
        position = self.position_encoder(state.direction, state)
        attended, _ = self.self_attention(
            query + position,
            query + position,
            query,
            attn_mask=self_attention_mask,
            need_weights=False,
        )
        query = self.self_norm(query + attended)
        keypoints = self.keypoints(query, state)
        projection = project_box_keypoints(self.projector, keypoints, metadata)
        extent = state_log_angular_extent(state)
        if self.config.use_fused_daf:
            if formatted_fpn is None:
                raise RuntimeError("fused V3 layer requires formatted FPN features")
            evidence = self.evidence.forward_fused_daf(
                query,
                projection,
                extent,
                normalize_direction(keypoints),
                formatted_fpn,
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
            sampled = sample_box_keypoints_fpn(
                fpn,
                projection.uv,
                sampler=self.sampler,
            )
            evidence = self.evidence(
                query,
                sampled,
                projection.geometry,
                extent,
            )
        query = self.evidence_norm(evidence.query)
        query = self.ffn_norm(query + self.ffn(query))
        state = self.refinement(query, state.direction, state)
        return query, state, evidence, keypoints


class FisheyeSparseBoxDecoder(nn.Module):
    """Centre-ray initializer followed by six 13-point box decoder stages."""

    def __init__(
        self,
        config: FisheyeBoxDecoderConfig | None = None,
        *,
        projector: NativeFisheyeProjector | None = None,
        sampler: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or FisheyeBoxDecoderConfig()
        self.projector = projector or NativeFisheyeProjector()
        self.sampler = sampler or FusedBilinearSampler()
        self.initializer = RayGuidedBoxInitializer(
            self.config,
            self.projector,
            self.sampler,
        )
        self.layers = nn.ModuleList(
            FisheyeSparseBoxDecoderLayer(
                self.config,
                self.projector,
                self.sampler,
            )
            for _ in range(self.config.num_layers)
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(self.config.embed_dims, self.config.num_classes)
            for _ in range(self.config.num_layers)
        )
        self.quality_heads = nn.ModuleList(
            nn.Linear(self.config.embed_dims, 2)
            for _ in range(self.config.num_layers)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        prior = 0.01
        bias = math.log(prior / (1.0 - prior))
        for classifier in self.classifiers:
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.constant_(classifier.bias, bias)
        for quality in self.quality_heads:
            nn.init.xavier_uniform_(quality.weight)
            nn.init.zeros_(quality.bias)

    def forward(
        self,
        fpn: Sequence[Tensor],
        metadata: dict,
        instance_feature: Tensor,
        anchor: Tensor,
        *,
        self_attention_mask: Tensor | None = None,
    ) -> FisheyeBoxDecoderOutput:
        if not fpn:
            raise ValueError("decoder requires FPN features")
        if instance_feature.ndim != 3 or instance_feature.shape[-1] != self.config.embed_dims:
            raise ValueError("instance_feature must be [B,Q,embed_dims]")
        if anchor.shape != instance_feature.shape[:2] + (11,):
            raise ValueError("anchor must be [B,Q,11]")
        if instance_feature.shape[1] < self.config.num_queries:
            raise ValueError("decoder received fewer than the free-query budget")
        total_queries = instance_feature.shape[1]
        if self_attention_mask is not None and self_attention_mask.shape != (
            total_queries,
            total_queries,
        ):
            raise ValueError("self_attention_mask must match total query count")

        state = RayState.from_sparse_anchor(anchor)
        initialized = self.initializer(instance_feature, state, fpn, metadata)
        query, state = initialized.query, initialized.state
        formatted_fpn = None
        if self.config.use_fused_daf:
            try:
                from projects.mmdet3d_plugin.ops import feature_maps_format
            except ImportError as exc:  # pragma: no cover - host-only path
                raise RuntimeError(
                    "fused V3 sampling requires the Sparse4D CUDA host"
                ) from exc
            formatted_fpn = feature_maps_format(fpn)
        classifications = []
        predictions = []
        qualities = []
        states = []
        camera_level_attention = []
        point_attention = []
        level_compatibility = []
        for layer, classifier, quality_head in zip(
            self.layers,
            self.classifiers,
            self.quality_heads,
        ):
            query, state, evidence, _ = layer(
                query,
                state,
                fpn,
                metadata,
                self_attention_mask,
                formatted_fpn,
            )
            classifications.append(classifier(query))
            predictions.append(state.to_sparse_anchor()[..., :10])
            qualities.append(quality_head(query))
            states.append(state)
            camera_level_attention.append(evidence.camera_level_attention)
            point_attention.append(evidence.point_attention)
            level_compatibility.append(evidence.level_compatibility)
        return FisheyeBoxDecoderOutput(
            classification=tuple(classifications),
            prediction=tuple(predictions),
            quality=tuple(qualities),
            states=tuple(states),
            initializer_state=initialized.state,
            initializer_gate=initialized.gate,
            camera_level_attention=tuple(camera_level_attention),
            point_attention=tuple(point_attention),
            level_compatibility=tuple(level_compatibility),
            query=query,
        )


__all__ = [
    "CameraRaySHValueModulation",
    "CameraRayLowRankResidual",
    "BoxPointProjection",
    "FisheyeBoxDecoderConfig",
    "FisheyeBoxDecoderOutput",
    "FisheyeBoxEvidenceAttention",
    "FisheyeBoxEvidenceOutput",
    "FisheyeBoxKeypointGenerator",
    "FisheyeSparseBoxDecoder",
    "FisheyeSparseBoxDecoderLayer",
    "RayGuidedBoxInitializer",
    "RayGuidedInitializationOutput",
    "project_box_keypoints",
    "sample_box_keypoints_fpn",
]
