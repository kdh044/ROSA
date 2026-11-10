"""Box keypoints and ROSA evidence attention.

Each decoder stage observes 13 support points of every box (7 fixed, 6
learned) in all fisheye views.  For each point ROSA weights camera x level
samples by their projection cues, lifts the viewing ray into the feature
(RFL), and then attends over the points.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cues import angular_fpn_geometry
from .projection import FisheyeObservationGeometry, NativeFisheyeProjector
from .ray import normalize_direction
from .rfl import SphericalRayFeatureLifting
from .types import RayFeatureRelation


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
    signed_face_features: Tensor


class CameraWiseRFLBeforeROSA(nn.Module):
    """Make each camera observation ray-aware before camera fusion.

    Applied after DAF has reduced the FPN levels within each camera.  The RFL
    residual updates both the per-camera value and the ROSA camera logit; its
    output projection is zero-initialized.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attention_dims: int,
    ) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.key_head_dims = int(attention_dims) // int(num_heads)
        self.rfl = SphericalRayFeatureLifting.for_compact_profile(embed_dims)
        self.key_proj = nn.Linear(embed_dims, attention_dims, bias=False)
        self.reliability = nn.Linear(embed_dims, num_heads, bias=False)
        self._streaming_compact_cache = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self._streaming_compact_cache = None
        self.rfl.reset_parameters()
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.reliability.weight)

    def streaming_compact_weights(self) -> tuple[Tensor, Tensor]:
        """Collapse the two linear RFL logit maps once for eval deployment."""

        parameters = (
            self.rfl.output_proj.weight,
            self.key_proj.weight,
            self.reliability.weight,
        )
        signature = tuple(
            (value.data_ptr(), value._version, value.device, value.dtype)
            for value in parameters
        )
        cached = self._streaming_compact_cache
        if cached is None or cached[0] != signature:
            output_weight, key_weight, reliability_weight = parameters
            key_rfl_weight = torch.mm(key_weight, output_weight).contiguous()
            reliability_rfl_weight = torch.mm(
                reliability_weight, output_weight
            ).contiguous()
            cached = (signature, key_rfl_weight, reliability_rfl_weight)
            self._streaming_compact_cache = cached
        return cached[1], cached[2]

    def forward(
        self,
        camera_features: Tensor,
        camera_rays: Tensor,
        valid: Tensor,
        query_heads: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, queries, points, cameras, dims = camera_features.shape
        expected_rays = (batch, queries, points, cameras, 3)
        if dims != self.embed_dims or tuple(camera_rays.shape) != expected_rays:
            raise ValueError("camera RFL feature/ray dimensions do not match")
        if tuple(valid.shape) != expected_rays[:-1] or valid.dtype is not torch.bool:
            raise ValueError("camera validity must be boolean [B,Q,P,C]")
        expected_query = (batch, queries, self.num_heads, self.key_head_dims)
        if tuple(query_heads.shape) != expected_query:
            raise ValueError("query heads do not match camera RFL attention")

        # RFL expects [B,N,C,P,D].
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


class FisheyeBoxKeypointGenerator(nn.Module):
    """Generate seven fixed and optional learned points from a ray box state."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_learnable_points: int = 6,
        fixed_scale: float = 0.45,
    ) -> None:
        super().__init__()
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

    def forward(self, query: Tensor, state) -> Tensor:
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


class FisheyeBoxEvidenceAttention(nn.Module):
    """Factorized ROSA camera-level and semantic-point evidence attention."""

    geometry_dims = 9

    def __init__(self, config, *, center_prefusion: bool = False) -> None:
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
        self.rfl = SphericalRayFeatureLifting.for_compact_profile(
            embed_dims=config.embed_dims
        )
        # The forked RNG keeps the initialization of the other modules unchanged.
        with torch.random.fork_rng(devices=[]):
            self.camera_rfl_before_rosa = CameraWiseRFLBeforeROSA(
                config.embed_dims,
                config.num_heads,
                config.attention_dims,
            )
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
            self.point_query_proj,
            self.point_key_proj,
            self.point_reliability,
            self.output_proj,
            getattr(self, "center_residual_proj", None),
        ):
            if module is not None:
                nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.output_proj.bias)
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
        with torch.random.fork_rng(devices=[]):
            self.camera_rfl_before_rosa.reset_parameters()

    def _prefuse_center(
        self,
        query: Tensor,
        point_features: Tensor,
        point_valid: Tensor,
    ) -> Tensor:
        """Condition the query on its centre-point feature."""

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

    def _fuse_points(
        self,
        query: Tensor,
        point_features: Tensor,
        point_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        point_reliability = self.point_reliability(point_tokens)
        point_values = self.point_norm(point_features).reshape(
            batch, queries, points, heads, self.value_head_dims
        )
        use_fused_point_attention = (
            not self.training
            and not torch.is_grad_enabled()
            and point_q.is_cuda
            and points == 13
            and heads == 8
            and key_dims == 8
            and self.value_head_dims == 32
        )
        if use_fused_point_attention:
            from projects.mmdet3d_plugin.ops import (
                rosa_point_attention_inference,
            )

            point_attention, evidence = rosa_point_attention_inference(
                point_q,
                point_k,
                point_reliability,
                point_values,
                point_valid,
            )
            evidence = evidence.reshape(batch, queries, cfg.embed_dims)
        else:
            point_logits = torch.einsum("bqhd,bqphd->bqph", point_q, point_k)
            point_logits = point_logits / math.sqrt(key_dims)
            point_logits = point_logits + point_reliability
            point_attention = self._masked_softmax(
                point_logits,
                point_valid[..., None],
                dim=2,
            )
            evidence = torch.einsum(
                "bqph,bqphd->bqhd",
                point_attention,
                point_values,
            ).reshape(batch, queries, cfg.embed_dims)
        # Fixed points 1:7 are (+x,-x,+y,-y,+z,-z); a face pair is used only
        # when both endpoints are visible.
        normalized_points = point_values.reshape(batch, queries, points, cfg.embed_dims)
        positive = normalized_points[:, :, (1, 3, 5)]
        negative = normalized_points[:, :, (2, 4, 6)]
        pair_valid = point_valid[:, :, (1, 3, 5)] & point_valid[:, :, (2, 4, 6)]
        signed_faces = (positive - negative) * pair_valid[..., None]
        updated = query + self.dropout(self.output_proj(evidence))
        return updated, point_attention, signed_faces

    def forward_fused_daf(
        self,
        query: Tensor,
        projection: BoxPointProjection,
        log_angular_extent: Tensor,
        point_rays: Tensor,
        formatted_fpn: tuple[Tensor, Tensor, Tensor],
        precomputed_geometry: tuple[Tensor, Tensor] | None = None,
    ) -> FisheyeBoxEvidenceOutput:
        """Fuse camera/level evidence for each point in one Sparse4D CUDA read.

        Every box point is passed to DAF as a one-keypoint anchor; DAF only
        performs bilinear sampling and the weighted reduction.
        """

        from projects.mmdet3d_plugin.ops import deformable_aggregation_per_camera_function

        cfg = self.config
        batch, queries, points, cameras = projection.uv.shape[:4]
        levels = len(cfg.level_strides)
        if query.shape != (batch, queries, cfg.embed_dims):
            raise ValueError("fused query shape does not match projection")
        if point_rays.shape != (batch, queries, points, 3):
            raise ValueError("point_rays must be [B,Q,P,3]")
        if log_angular_extent.shape != (batch, queries, 2):
            raise ValueError("log_angular_extent must be [B,Q,2]")

        geometry = projection.geometry
        flat_prefix = (batch, queries * points, cameras)
        if precomputed_geometry is None:
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
        else:
            descriptor, compatibility = precomputed_geometry
            if descriptor.shape != flat_prefix + (levels, self.geometry_dims):
                raise ValueError("precomputed descriptor shape mismatch")
            if compatibility.shape != flat_prefix + (levels,):
                raise ValueError("precomputed compatibility shape mismatch")
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
        # p(camera, level) = p(level | camera) p(camera): DAF reduces
        # levels per camera, then RFL updates the camera marginal.
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

        use_streaming_camera_cuda = (
            cfg.streaming_camera_rfl_cuda
            and not self.training
            and not torch.is_grad_enabled()
        )
        if use_streaming_camera_cuda:
            from projects.mmdet3d_plugin.ops import (
                deformable_aggregation_rfl_rosa_inference,
            )

            rfl = self.camera_rfl_before_rosa.rfl
            key_rfl_weight, reliability_rfl_weight = (
                self.camera_rfl_before_rosa.streaming_compact_weights()
            )
            point_features_flat, camera_attention_flat = (
                deformable_aggregation_rfl_rosa_inference(
                    col_feats,
                    spatial_shape,
                    scale_start_index,
                    sampling_location,
                    grouped_weights,
                    camera_logits.reshape(
                        batch, queries * points, cameras, heads
                    ),
                    geometry.rays.reshape(
                        batch, queries * points, cameras, 3
                    ),
                    geometry.valid.reshape(
                        batch, queries * points, cameras
                    ),
                    q,
                    rfl.appearance_proj.weight,
                    rfl.ray_proj.weight,
                    rfl.output_proj.weight,
                    key_rfl_weight,
                    reliability_rfl_weight,
                    points,
                )
            )
            point_features = point_features_flat.reshape(
                batch, queries, points, cfg.embed_dims
            ).to(dtype=query.dtype)
            camera_attention = camera_attention_flat.reshape(
                batch, queries, points, cameras, heads
            ).to(dtype=logits.dtype)
        else:
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

        point_valid = geometry.valid.any(dim=3)
        return self._post_daf_evidence_fusion(
            query,
            point_features,
            point_valid,
            point_rays,
            attention,
            compatibility,
        )

    def _post_daf_evidence_fusion(
        self,
        query: Tensor,
        point_features: Tensor,
        point_valid: Tensor,
        point_rays: Tensor,
        attention: Tensor,
        compatibility: Tensor,
    ) -> FisheyeBoxEvidenceOutput:
        """Post-DAF evidence fusion (pure PyTorch, ``torch.compile``-friendly)."""
        point_features = self.feature_norm(point_features)
        rfl_features = point_features[:, :, None, :, :]
        rfl_rays = normalize_direction(point_rays)[:, :, None, :, :]
        rfl_valid = point_valid[:, :, None, :]
        point_features = point_features + self.rfl(
            rfl_features,
            RayFeatureRelation(ray=rfl_rays, valid=rfl_valid),
        ).squeeze(2)
        query = self._prefuse_center(query, point_features, point_valid)
        updated, point_attention, signed_faces = self._fuse_points(
            query,
            point_features,
            point_valid,
        )
        return FisheyeBoxEvidenceOutput(
            query=updated,
            camera_level_attention=attention,
            point_attention=point_attention,
            level_compatibility=compatibility,
            signed_face_features=signed_faces,
        )


__all__ = [
    "BoxPointProjection",
    "FisheyeBoxEvidenceAttention",
    "FisheyeBoxEvidenceOutput",
    "FisheyeBoxKeypointGenerator",
    "project_box_keypoints",
]
