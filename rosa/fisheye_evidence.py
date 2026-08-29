"""One-pass ROSA--RFL evidence attention for four native fisheye views.

The module consumes image features that have already been sampled once at a
query ray.  It deliberately owns the only camera/level softmax: RFL changes
the values, calibrated tangent/Jacobian geometry changes the keys and
reliability logits, and the resulting evidence updates the decoder query.
There is no second sidecar observation or box-keypoint aggregation here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rfl import SphericalRayFeatureLifting
from .types import RayFeatureRelation


@dataclass(frozen=True)
class FisheyeObservationGeometry:
    """Calibrated per-query geometry before FPN-level expansion.

    All values use the query-leading convention ``[B,Q,C,...]``.  ``rays``
    are camera-frame unit rays.  ``tangent`` expresses the camera observation
    in the ego query's local two-axis tangent frame.  ``jacobian`` maps a
    local angular displacement in radians to native-image pixels.
    """

    rays: Tensor
    tangent: Tensor
    jacobian: Tensor
    lens_margin: Tensor
    overlap: Tensor
    valid: Tensor

    def validate(self, batch: int, queries: int, cameras: int) -> None:
        prefix = (batch, queries, cameras)
        expected = {
            "rays": prefix + (3,),
            "tangent": prefix + (2,),
            "jacobian": prefix + (2, 2),
            "lens_margin": prefix + (1,),
            "overlap": prefix + (1,),
            "valid": prefix,
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                observed = getattr(value, "shape", None)
                raise ValueError(f"{name} shape {observed} does not match {shape}")
        if self.valid.dtype is not torch.bool:
            raise TypeError("fisheye observation validity must be boolean")


@dataclass(frozen=True)
class FisheyeEvidenceOutput:
    """Updated query and bounded diagnostics for losses/analysis."""

    query: Tensor
    attention: Tensor
    level_compatibility: Tensor
    log_angular_extent: Tensor
    supported: Tensor


@dataclass(frozen=True)
class FisheyeEvidenceConfig:
    embed_dims: int = 256
    num_heads: int = 8
    attention_dims: int = 64
    num_cameras: int = 4
    level_strides: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0)
    geometry_hidden_dims: int = 64
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0
    predict_angular_extent: bool = True
    initial_angular_extent_rad: tuple[float, float] = (0.12, 0.08)
    min_angular_extent_rad: float = math.radians(0.25)
    max_angular_extent_rad: float = math.radians(60.0)
    scale_match_init: float = 1.0

    def __post_init__(self) -> None:
        if self.embed_dims <= 0 or self.num_heads <= 0:
            raise ValueError("attention dimensions must be positive")
        if self.embed_dims % self.num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")
        if self.attention_dims <= 0 or self.attention_dims % self.num_heads != 0:
            raise ValueError(
                "attention_dims must be positive and divisible by num_heads"
            )
        if self.num_cameras <= 0 or not self.level_strides:
            raise ValueError("camera and FPN level counts must be positive")
        if any(stride <= 0.0 for stride in self.level_strides):
            raise ValueError("FPN strides must be positive")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("invalid attention dropout")
        if not 0.0 <= self.projection_dropout < 1.0:
            raise ValueError("invalid projection dropout")
        if not (0.0 < self.min_angular_extent_rad < self.max_angular_extent_rad):
            raise ValueError("invalid angular extent interval")


def angular_fpn_geometry(
    tangent: Tensor,
    jacobian: Tensor,
    lens_margin: Tensor,
    overlap: Tensor,
    valid: Tensor,
    level_strides: Tensor,
    log_angular_extent: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build the distortion-aware descriptor and FPN compatibility score.

    The singular values of the fisheye Jacobian are pixels per radian along
    its local principal axes.  Dividing an FPN stride by those values gives
    the angular footprint of one feature cell.  Compatibility compares that
    footprint to the query's expected angular object extent.
    """

    if tangent.ndim != 4 or tangent.shape[-1] != 2:
        raise ValueError("tangent must be [B,Q,C,2]")
    prefix = tangent.shape[:-1]
    if jacobian.shape != prefix + (2, 2):
        raise ValueError("jacobian must align with tangent and end in [2,2]")
    if lens_margin.shape != prefix + (1,) or overlap.shape != prefix + (1,):
        raise ValueError("lens_margin and overlap must be [B,Q,C,1]")
    if valid.shape != prefix or valid.dtype is not torch.bool:
        raise ValueError("valid must be boolean [B,Q,C]")
    if level_strides.ndim != 1 or level_strides.numel() == 0:
        raise ValueError("level_strides must be a non-empty vector")
    if log_angular_extent.shape != prefix[:2] + (2,):
        raise ValueError("log_angular_extent must be [B,Q,2]")

    # The lens Jacobian is calibrated observation metadata, not a learnable
    # state.  Its two singular values coincide on locally isotropic image
    # regions; differentiating the closed-form eigensystem there has an
    # undefined direction (sqrt'(0) = inf) and can contaminate otherwise
    # finite query/backbone gradients.  Keep the exact forward descriptor but
    # stop gradients through this non-identifiable decomposition.  Query-ray
    # learning still flows through native-image UV sampling.
    safe_jacobian = torch.nan_to_num(
        jacobian.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).detach()
    # Exact closed-form singular values for a 2x2 matrix.  Calling the generic
    # batched SVD launches a Jacobi solver for every decoder stage, which is
    # disproportionate for 2x2 fisheye Jacobians.
    a, b = safe_jacobian[..., 0, :].unbind(dim=-1)
    c, d = safe_jacobian[..., 1, :].unbind(dim=-1)
    trace = a.square() + b.square() + c.square() + d.square()
    determinant_squared = (a * d - b * c).square()
    discriminant = (trace.square() - 4.0 * determinant_squared).clamp_min(0.0)
    root = discriminant.sqrt()
    singular = torch.stack(
        (
            ((trace + root) * 0.5).clamp_min(1e-8).sqrt(),
            ((trace - root) * 0.5).clamp_min(1e-8).sqrt(),
        ),
        dim=-1,
    ).clamp_min(1e-4)
    log_singular = singular.log()
    log_stride = level_strides.float().log()
    log_cell_extent = log_stride.view(1, 1, 1, -1, 1) - log_singular[..., None, :]
    expected = log_angular_extent.float()[..., None, None, :]
    compatibility = -(log_cell_extent - expected).square().mean(dim=-1)

    levels = level_strides.numel()
    tangent_l = tangent.float()[..., None, :].expand(*prefix, levels, 2)
    singular_l = log_singular[..., None, :].expand(*prefix, levels, 2)
    anisotropy = (
        (log_singular[..., 0:1] - log_singular[..., 1:2])
        .abs()[..., None, :]
        .expand(*prefix, levels, 1)
    )
    margin_l = (
        torch.nan_to_num(lens_margin.float())
        .clamp(0.0, 1.0)[..., None, :]
        .expand(*prefix, levels, 1)
    )
    overlap_l = (
        torch.nan_to_num(overlap.float())
        .clamp(0.0, 1.0)[..., None, :]
        .expand(*prefix, levels, 1)
    )
    stride_l = log_stride.view(1, 1, 1, levels, 1).expand(*prefix, levels, 1)
    descriptor = torch.cat(
        (
            tangent_l,
            singular_l,
            anisotropy,
            margin_l,
            overlap_l,
            stride_l,
            compatibility[..., None],
        ),
        dim=-1,
    )
    descriptor = torch.where(
        valid[..., None, None], descriptor, torch.zeros_like(descriptor)
    )
    compatibility = torch.where(
        valid[..., None], compatibility, torch.zeros_like(compatibility)
    )
    return descriptor, compatibility


class UnifiedFisheyeEvidenceAttention(nn.Module):
    """Fuse camera and level evidence through exactly one joint softmax."""

    geometry_dims = 9

    def __init__(self, config: FisheyeEvidenceConfig | None = None) -> None:
        super().__init__()
        self.config = config or FisheyeEvidenceConfig()
        cfg = self.config
        self.key_head_dims = cfg.attention_dims // cfg.num_heads
        self.value_head_dims = cfg.embed_dims // cfg.num_heads
        self.register_buffer(
            "level_strides",
            torch.tensor(cfg.level_strides, dtype=torch.float32),
            persistent=True,
        )
        self.query_norm = nn.LayerNorm(cfg.embed_dims)
        self.feature_norm = nn.LayerNorm(cfg.embed_dims)
        # Selection keys are compact, while values keep the full detector
        # width.  A bias-free value projection commutes with the weighted
        # reduction, so the output map is applied once per query below rather
        # than once for every camera-level observation.
        self.query_proj = nn.Linear(cfg.embed_dims, cfg.attention_dims, bias=False)
        self.key_proj = nn.Linear(cfg.embed_dims, cfg.attention_dims, bias=False)
        self.geometry_key = nn.Sequential(
            nn.Linear(self.geometry_dims, cfg.geometry_hidden_dims),
            nn.SiLU(),
            nn.Linear(cfg.geometry_hidden_dims, cfg.attention_dims, bias=False),
        )
        self.geometry_reliability = nn.Sequential(
            nn.Linear(self.geometry_dims, cfg.geometry_hidden_dims),
            nn.SiLU(),
            nn.Linear(cfg.geometry_hidden_dims, cfg.num_heads),
        )
        self.feature_reliability = nn.Linear(cfg.embed_dims, cfg.num_heads, bias=False)
        self.extent_head = (
            nn.Linear(cfg.embed_dims, 2) if cfg.predict_angular_extent else None
        )
        self.rfl = SphericalRayFeatureLifting.for_compact_profile(
            embed_dims=cfg.embed_dims
        )
        self.output_proj = nn.Linear(cfg.embed_dims, cfg.embed_dims)
        self.attention_dropout = nn.Dropout(cfg.attention_dropout)
        self.projection_dropout = nn.Dropout(cfg.projection_dropout)
        self.scale_match_raw = nn.Parameter(
            torch.full((cfg.num_heads,), self._inverse_softplus(cfg.scale_match_init))
        )
        self.reset_parameters()

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value <= 0.0:
            raise ValueError("scale_match_init must be positive")
        return math.log(math.expm1(value))

    def reset_parameters(self) -> None:
        for module in (
            self.query_proj,
            self.key_proj,
            self.output_proj,
            self.feature_reliability,
        ):
            nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.output_proj.bias)
        for sequence in (self.geometry_key, self.geometry_reliability):
            for module in sequence.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        if self.extent_head is not None:
            nn.init.zeros_(self.extent_head.weight)
            initial = torch.tensor(self.config.initial_angular_extent_rad).log()
            self.extent_head.bias.data.copy_(initial)
        # RFL owns a deliberately zero output map; reapply its contract after
        # the surrounding attention projections are initialized.
        self.rfl.reset_parameters()

    def _angular_extent(self, query: Tensor, supplied: Tensor | None) -> Tensor:
        cfg = self.config
        if supplied is None:
            if self.extent_head is None:
                raise ValueError(
                    "log_angular_extent is required when extent prediction is disabled"
                )
            value = self.extent_head(self.query_norm(query).float())
        else:
            if supplied.shape != query.shape[:2] + (2,):
                raise ValueError("log_angular_extent must be [B,Q,2]")
            value = supplied.float()
        return value.clamp(
            min=math.log(cfg.min_angular_extent_rad),
            max=math.log(cfg.max_angular_extent_rad),
        )

    def forward(
        self,
        query: Tensor,
        sampled_features: Tensor,
        geometry: FisheyeObservationGeometry,
        *,
        log_angular_extent: Tensor | None = None,
    ) -> FisheyeEvidenceOutput:
        cfg = self.config
        if query.ndim != 3 or query.shape[-1] != cfg.embed_dims:
            raise ValueError("query must be [B,Q,embed_dims]")
        batch, queries = query.shape[:2]
        expected_features = (
            batch,
            queries,
            cfg.num_cameras,
            len(cfg.level_strides),
            cfg.embed_dims,
        )
        if tuple(sampled_features.shape) != expected_features:
            raise ValueError(
                f"sampled_features shape {tuple(sampled_features.shape)} "
                f"does not match {expected_features}"
            )
        geometry.validate(batch, queries, cfg.num_cameras)

        extent = self._angular_extent(query, log_angular_extent)
        descriptor, compatibility = angular_fpn_geometry(
            geometry.tangent,
            geometry.jacobian,
            geometry.lens_margin,
            geometry.overlap,
            geometry.valid,
            self.level_strides,
            extent,
        )
        valid = geometry.valid[..., None].expand(
            batch, queries, cfg.num_cameras, len(cfg.level_strides)
        )
        rays = geometry.rays[..., None, :].expand(
            batch, queries, cfg.num_cameras, len(cfg.level_strides), 3
        )
        finite_features = torch.nan_to_num(
            sampled_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        lifted_values = finite_features + self.rfl(
            finite_features,
            RayFeatureRelation(ray=rays, valid=valid),
        )

        normalized_query = self.query_norm(query)
        normalized_features = self.feature_norm(finite_features)
        heads = cfg.num_heads
        key_dims = self.key_head_dims
        q = self.query_proj(normalized_query).reshape(batch, queries, heads, key_dims)
        k = (
            self.key_proj(normalized_features)
            + self.geometry_key(descriptor).to(normalized_features.dtype)
        ).reshape(
            batch,
            queries,
            cfg.num_cameras,
            len(cfg.level_strides),
            heads,
            key_dims,
        )
        v = self.feature_norm(lifted_values).reshape(
            batch,
            queries,
            cfg.num_cameras,
            len(cfg.level_strides),
            heads,
            self.value_head_dims,
        )
        logits = torch.einsum("bqhd,bqclhd->bqclh", q, k) / math.sqrt(key_dims)
        logits = logits + self.geometry_reliability(descriptor)
        logits = logits + self.feature_reliability(normalized_features)
        scale = F.softplus(self.scale_match_raw).view(1, 1, 1, 1, heads)
        logits = logits + compatibility[..., None] * scale

        flat_logits = logits.flatten(start_dim=2, end_dim=3)
        flat_valid = valid.flatten(start_dim=2, end_dim=3)[..., None]
        masked_logits = torch.where(
            flat_valid,
            flat_logits,
            torch.full_like(flat_logits, -torch.finfo(flat_logits.dtype).max),
        )
        attention = torch.softmax(masked_logits.float(), dim=2).to(logits.dtype)
        attention = attention * flat_valid.to(attention.dtype)
        denominator = attention.sum(dim=2, keepdim=True)
        attention = torch.where(
            denominator > 0.0,
            attention / denominator.clamp_min(1e-8),
            torch.zeros_like(attention),
        )
        attention = self.attention_dropout(attention)

        flat_values = v.flatten(start_dim=2, end_dim=3)
        evidence = torch.einsum("bqsh,bqshd->bqhd", attention, flat_values)
        evidence = evidence.reshape(batch, queries, cfg.embed_dims)
        updated = query + self.projection_dropout(self.output_proj(evidence))
        return FisheyeEvidenceOutput(
            query=updated,
            attention=attention.reshape(
                batch,
                queries,
                cfg.num_cameras,
                len(cfg.level_strides),
                heads,
            ),
            level_compatibility=compatibility,
            log_angular_extent=extent,
            supported=geometry.valid.any(dim=2),
        )


__all__ = [
    "FisheyeEvidenceConfig",
    "FisheyeEvidenceOutput",
    "FisheyeObservationGeometry",
    "UnifiedFisheyeEvidenceAttention",
    "angular_fpn_geometry",
]
