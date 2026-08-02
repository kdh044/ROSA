"""Camera- and keypoint-level observation fusion."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ROSAConfig
from .rfl import SphericalRayFeatureLifting
from .types import ObservationEncoding, RayFeatureRelation


def masked_softmax(logits: Tensor, valid: Tensor, dim: int) -> Tensor:
    """Apply softmax without producing NaNs for fully masked rows."""

    if logits.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("masked softmax tensors are misaligned")
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked.float(), dim=dim).to(logits.dtype)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    denominator = weights.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > 0,
        weights / denominator.clamp_min(1e-8),
        torch.zeros_like(weights),
    )


def mean_pairwise_camera_cosine(values: Tensor, valid: Tensor) -> Tensor:
    """Average valid camera-pair cosine without materializing ``C x C``."""

    if values.ndim != 5 or valid.shape != values.shape[:-1]:
        raise ValueError("camera consistency tensors are misaligned")
    masked = values.permute(0, 1, 3, 2, 4) * valid.permute(0, 1, 3, 2)[..., None].to(
        values.dtype
    )
    camera_count = valid.sum(dim=2).to(values.dtype)
    pair_count = camera_count * (camera_count - 1.0) * 0.5
    summed = masked.sum(dim=3)
    pair_sum = (summed.square().sum(dim=-1) - masked.square().sum(dim=(-1, -2))) * 0.5
    return torch.where(
        pair_count > 0,
        pair_sum / pair_count.clamp_min(1.0),
        torch.zeros_like(pair_sum),
    )


class PointwiseROSA(nn.Module):
    """Fuse camera evidence independently at each object keypoint."""

    geometry_dims = 14

    def __init__(self, config: ROSAConfig) -> None:
        super().__init__()
        self.cfg = config
        dims = config.embed_dims
        hidden = config.hidden_dims
        self.rfl = (
            SphericalRayFeatureLifting.for_compact_profile(embed_dims=dims)
            if config.architecture_profile == "compact"
            else SphericalRayFeatureLifting(
                embed_dims=dims,
                hidden_dims=max(16, hidden // 4),
            )
        )
        self.feature_norm = nn.LayerNorm(dims)
        self.query_proj = nn.Linear(dims, hidden, bias=False)
        self.key_proj = nn.Linear(
            dims + self.geometry_dims,
            hidden,
            bias=False,
        )
        self.value_proj = (
            nn.Identity()
            if config.architecture_profile == "compact"
            else nn.Linear(dims, dims, bias=False)
        )
        self.geometry_bias = nn.Sequential(
            nn.Linear(self.geometry_dims, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.point_embedding = nn.Embedding(config.num_keypoints, dims)
        if config.architecture_profile == "compact":
            self.point_query = None
            self.point_key = None
            self.point_output = None
        else:
            self.point_query = nn.Linear(dims, dims, bias=False)
            self.point_key = nn.Linear(dims, dims, bias=False)
            self.point_output = nn.Sequential(
                nn.LayerNorm(dims),
                nn.Linear(dims, dims),
                nn.SiLU(),
            )

    def reset_parameters(self) -> None:
        self.rfl.reset_parameters()
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.rfl.output_proj:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.point_embedding.weight, std=0.02)

    def forward(
        self,
        features: Tensor,
        geometry: Tensor,
        relation: Tensor | RayFeatureRelation,
        valid: Tensor,
        query: Tensor,
        *,
        observation_usage: str = "full",
    ) -> ObservationEncoding:
        if observation_usage not in ("full", "base_support", "candidate"):
            raise ValueError(f"invalid observation usage: {observation_usage!r}")
        if (
            features.ndim != 5
            or geometry.shape != features.shape[:-1] + (self.geometry_dims,)
            or valid.shape != features.shape[:-1]
            or query.shape != features.shape[:2] + (self.cfg.embed_dims,)
        ):
            raise ValueError("ROSA observation tensors are misaligned")

        retain_auxiliary = observation_usage == "full"
        need_consistency = observation_usage in ("full", "candidate")
        rfl_scale = 1.0 if self.cfg.enable_rfl else 0.0
        lifted = features + rfl_scale * self.rfl(features, relation)
        normalized = self.feature_norm(lifted)
        query_key = self.query_proj(query)[:, :, None, None, :]
        scoring_geometry = (
            geometry if self.cfg.enable_rosa_geometry else torch.zeros_like(geometry)
        )
        camera_key = self.key_proj(torch.cat((normalized, scoring_geometry), dim=-1))
        logits = (query_key * camera_key).sum(dim=-1)
        logits = logits / math.sqrt(self.cfg.hidden_dims)
        logits = logits + self.geometry_bias(scoring_geometry).squeeze(-1)
        if self.cfg.uniform_camera_fusion:
            logits = logits * 0.0

        camera_weights = masked_softmax(logits, valid, dim=2)
        values = self.value_proj(lifted)
        point_features = (camera_weights[..., None] * values).sum(dim=2)
        point_valid = valid.any(dim=2)
        point_ids = torch.arange(
            self.cfg.num_keypoints,
            device=query.device,
        )
        point_features = point_features + self.point_embedding(point_ids)[None, None]
        if self.cfg.architecture_profile == "compact":
            point_logits = (
                F.normalize(query, dim=-1, eps=1e-6)[:, :, None, :]
                * F.normalize(point_features, dim=-1, eps=1e-6)
            ).sum(dim=-1)
        else:
            point_logits = (
                self.point_query(query)[:, :, None, :] * self.point_key(point_features)
            ).sum(dim=-1)
            point_logits = point_logits / math.sqrt(self.cfg.embed_dims)
        if self.cfg.uniform_point_fusion:
            point_logits = point_logits * 0.0
        point_weights = masked_softmax(point_logits, point_valid, dim=2)
        object_feature = (point_weights[..., None] * point_features).sum(dim=2)
        if self.point_output is None:
            object_feature = F.layer_norm(
                object_feature,
                (self.cfg.embed_dims,),
            )
        else:
            object_feature = self.point_output(object_feature)

        if not need_consistency:
            consistency = None
        else:
            normalized_values = F.normalize(
                torch.nan_to_num(values.float()),
                dim=-1,
                eps=1e-6,
            )
            if self.cfg.architecture_profile == "compact":
                point_consistency = mean_pairwise_camera_cosine(
                    normalized_values,
                    valid,
                )
            else:
                similarity = torch.einsum(
                    "bncpd,bnfpd->bnpcf",
                    normalized_values,
                    normalized_values,
                )
                camera_valid = valid.permute(0, 1, 3, 2)
                pair_valid = camera_valid[..., :, None] & camera_valid[..., None, :]
                upper_triangle = torch.triu(
                    torch.ones(
                        self.cfg.num_cameras,
                        self.cfg.num_cameras,
                        device=valid.device,
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )
                pair_valid = pair_valid & upper_triangle
                pair_count = pair_valid.sum(dim=(-1, -2))
                point_consistency = torch.where(
                    pair_count > 0,
                    (similarity * pair_valid.to(similarity.dtype)).sum(dim=(-1, -2))
                    / pair_count.clamp_min(1).to(similarity.dtype),
                    torch.zeros_like(pair_count, dtype=similarity.dtype),
                )
            consistency = torch.where(
                point_valid.any(dim=-1),
                (point_consistency * point_valid.to(point_consistency.dtype)).sum(
                    dim=-1
                )
                / point_valid.sum(dim=-1).clamp_min(1).to(point_consistency.dtype),
                torch.zeros_like(point_consistency[..., 0]),
            )

        if retain_auxiliary:
            valid_float = valid.to(geometry.dtype)
            valid_count = valid_float.sum(dim=3).clamp_min(1.0)

            def camera_mean(value: Tensor) -> Tensor:
                return (value * valid_float).sum(dim=3) / valid_count

            optical_angle = torch.acos((-geometry[..., 2]).clamp(-1.0, 1.0))
            camera_geometry_summary = torch.stack(
                (
                    camera_mean(optical_angle),
                    camera_mean(geometry[..., 5]),
                    camera_mean(geometry[..., 6]),
                    camera_mean(geometry[..., 7]),
                    camera_mean(geometry[..., 8]),
                    camera_mean(geometry[..., 9]),
                    valid_float.mean(dim=3),
                ),
                dim=-1,
            )
            camera_weight_mean = camera_weights.sum(dim=3) / valid_count
        else:
            camera_geometry_summary = None
            camera_weight_mean = None
        return ObservationEncoding(
            object_feature=object_feature,
            point_features=point_features if retain_auxiliary else None,
            camera_weights=camera_weights if retain_auxiliary else None,
            valid=valid,
            supported=point_valid.any(dim=-1),
            cross_camera_consistency=(
                consistency.to(query.dtype) if consistency is not None else None
            ),
            camera_weight_mean=(
                camera_weight_mean.to(query.dtype)
                if camera_weight_mean is not None
                else None
            ),
            camera_geometry_summary=(
                camera_geometry_summary.to(query.dtype)
                if camera_geometry_summary is not None
                else None
            ),
        )


__all__ = [
    "PointwiseROSA",
    "masked_softmax",
    "mean_pairwise_camera_cosine",
]
