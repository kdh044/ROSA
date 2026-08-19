"""Six-stage box-free ray decoder for four native fisheye cameras."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .fisheye_evidence import (
    FisheyeEvidenceConfig,
    FisheyeEvidenceOutput,
    UnifiedFisheyeEvidenceAttention,
)
from .fisheye_projection import (
    NativeFisheyeProjection,
    NativeFisheyeProjector,
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


@dataclass(frozen=True)
class FisheyeRayDecoderConfig:
    embed_dims: int = 256
    num_queries: int = 300
    num_layers: int = 6
    num_classes: int = 6
    num_heads: int = 8
    ffn_dims: int = 512
    num_cameras: int = 4
    level_strides: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0)
    dropout: float = 0.0
    min_range_m: float = 0.5
    max_range_m: float = 68.0

    def __post_init__(self) -> None:
        if self.embed_dims <= 0 or self.num_queries <= 0 or self.num_layers <= 0:
            raise ValueError("decoder dimensions must be positive")
        if self.embed_dims % self.num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")
        if self.num_classes <= 0 or self.num_cameras <= 0:
            raise ValueError("class and camera counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid decoder dropout")


@dataclass(frozen=True)
class FisheyeRayDecoderOutput:
    """Per-stage detector predictions and ray diagnostics."""

    classification: tuple[Tensor, ...]
    prediction: tuple[Tensor, ...]
    quality: tuple[Tensor, ...]
    states: tuple[RayState, ...]
    attention: tuple[Tensor, ...]
    level_compatibility: tuple[Tensor, ...]
    query: Tensor


class SphericalRayQueryBank(nn.Module):
    """Learnable query features and directions with no initial box/range."""

    def __init__(self, initial_directions: Tensor, embed_dims: int = 256) -> None:
        super().__init__()
        if initial_directions.ndim != 2 or initial_directions.shape[-1] != 3:
            raise ValueError("initial_directions must be [Q,3]")
        if embed_dims <= 0:
            raise ValueError("embed_dims must be positive")
        self.num_queries = initial_directions.shape[0]
        self.embed_dims = int(embed_dims)
        self.direction = nn.Parameter(normalize_direction(initial_directions.float()))
        self.feature = nn.Parameter(torch.empty(self.num_queries, self.embed_dims))
        self.reset_parameters()

    @classmethod
    def from_sparse_anchors(
        cls, anchors: Tensor, embed_dims: int = 256
    ) -> "SphericalRayQueryBank":
        """Use only normalized center directions from a fair Sparse4D prior."""

        if anchors.ndim != 2 or anchors.shape[-1] < 3:
            raise ValueError("anchors must be [Q,D] with XYZ first")
        return cls(anchors[..., :3], embed_dims=embed_dims)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.feature)

    def forward(self, batch_size: int) -> tuple[Tensor, Tensor]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        query = self.feature[None].expand(batch_size, -1, -1)
        direction = normalize_direction(self.direction)[None].expand(batch_size, -1, -1)
        return query, direction


class RayStatePositionEncoder(nn.Module):
    """Encode the propagated spherical state for decoder self-attention."""

    input_dims = 17  # SH9 + range + std + size3 + yaw2 + initialized1

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )

    def forward(self, direction: Tensor, state: RayState | None) -> Tensor:
        basis = SphericalRayFeatureLifting.real_sh_degree_2(direction)
        shape = direction.shape[:-1]
        if state is None:
            scalar = direction.new_zeros(shape + (1,))
            size = direction.new_zeros(shape + (3,))
            yaw = direction.new_zeros(shape + (2,))
            initialized = scalar
            inputs = torch.cat((basis, scalar, scalar, size, yaw, initialized), dim=-1)
        else:
            state.validate()
            initialized = direction.new_ones(shape + (1,))
            inputs = torch.cat(
                (
                    basis,
                    state.log_range,
                    state.log_range_std,
                    state.log_size,
                    state.yaw,
                    initialized,
                ),
                dim=-1,
            )
        return self.encoder(inputs)


def state_log_angular_extent(state: RayState) -> Tensor:
    """Convert metric object size/range to horizontal and vertical angles."""

    state.validate()
    size = state.log_size.exp()
    distance = state.range.clamp_min(1e-4)
    horizontal_size = size[..., :2].amax(dim=-1, keepdim=True)
    horizontal = 2.0 * torch.atan(0.5 * horizontal_size / distance)
    vertical = 2.0 * torch.atan(0.5 * size[..., 2:3] / distance)
    return torch.cat((horizontal, vertical), dim=-1).clamp_min(1e-5).log()


class UnifiedFisheyeDecoderLayer(nn.Module):
    """Self-attention, one fisheye observation, FFN, and ray refinement."""

    def __init__(
        self,
        config: FisheyeRayDecoderConfig,
        *,
        initialize: bool,
        projector: NativeFisheyeProjector,
        sampler: nn.Module,
    ) -> None:
        super().__init__()
        self.config = config
        self.initialize = bool(initialize)
        self.projector = projector
        self.sampler = sampler
        self.position_encoder = RayStatePositionEncoder(config.embed_dims)
        self.self_attention = nn.MultiheadAttention(
            config.embed_dims,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_dropout = nn.Dropout(config.dropout)
        self.self_norm = nn.LayerNorm(config.embed_dims)
        evidence_config = FisheyeEvidenceConfig(
            embed_dims=config.embed_dims,
            num_heads=config.num_heads,
            num_cameras=config.num_cameras,
            level_strides=config.level_strides,
            attention_dropout=config.dropout,
            projection_dropout=config.dropout,
            predict_angular_extent=initialize,
        )
        self.evidence = UnifiedFisheyeEvidenceAttention(evidence_config)
        self.evidence_norm = nn.LayerNorm(config.embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dims, config.ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dims, config.embed_dims),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.embed_dims)
        refinement_config = RayRefinementConfig(
            embed_dims=config.embed_dims,
            hidden_dims=config.embed_dims,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
        )
        self.refinement = FisheyeRayStateRefinement(
            refinement_config, initialize=initialize
        )

    def forward(
        self,
        query: Tensor,
        direction: Tensor,
        state: RayState | None,
        fpn: tuple[Tensor, ...] | list[Tensor],
        metadata: dict,
        self_attention_mask: Tensor | None = None,
    ) -> tuple[
        Tensor,
        RayState,
        NativeFisheyeProjection,
        FisheyeEvidenceOutput,
    ]:
        if self.initialize != (state is None):
            raise ValueError("D0 requires no state and D1-D5 require a state")
        current_direction = direction if state is None else state.direction
        position = self.position_encoder(current_direction, state)
        attended, _ = self.self_attention(
            query + position,
            query + position,
            query,
            attn_mask=self_attention_mask,
            need_weights=False,
        )
        query = self.self_norm(query + self.self_dropout(attended))

        projection = self.projector(
            current_direction,
            metadata,
            points=None if state is None else state.center,
        )
        sampled = sample_native_fisheye_fpn(
            fpn,
            projection.uv,
            sampler=self.sampler,
        )
        extent = None if state is None else state_log_angular_extent(state)
        evidence = self.evidence(
            query,
            sampled,
            projection.geometry,
            log_angular_extent=extent,
        )
        query = self.evidence_norm(evidence.query)
        query = self.ffn_norm(query + self.ffn(query))
        next_state = self.refinement(
            query,
            current_direction,
            state,
        )
        return query, next_state, projection, evidence


class FisheyeRayDecoder(nn.Module):
    """End-to-end D0--D5 query propagation without box-keypoint sampling."""

    def __init__(
        self,
        initial_directions: Tensor,
        config: FisheyeRayDecoderConfig | None = None,
        *,
        projector: NativeFisheyeProjector | None = None,
        sampler: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or FisheyeRayDecoderConfig(
            num_queries=initial_directions.shape[0]
        )
        if initial_directions.shape[0] != self.config.num_queries:
            raise ValueError("initial direction count must match num_queries")
        self.query_bank = SphericalRayQueryBank(
            initial_directions, self.config.embed_dims
        )
        self.projector = projector or NativeFisheyeProjector()
        self.sampler = sampler or FusedBilinearSampler()
        self.layers = nn.ModuleList(
            UnifiedFisheyeDecoderLayer(
                self.config,
                initialize=index == 0,
                projector=self.projector,
                sampler=self.sampler,
            )
            for index in range(self.config.num_layers)
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(self.config.embed_dims, self.config.num_classes)
            for _ in range(self.config.num_layers)
        )
        self.quality_heads = nn.ModuleList(
            nn.Linear(self.config.embed_dims, 2) for _ in range(self.config.num_layers)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        prior = 0.01
        bias = math.log(prior / (1.0 - prior))
        for classifier in self.classifiers:
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.constant_(classifier.bias, bias)
        for quality_head in self.quality_heads:
            nn.init.xavier_uniform_(quality_head.weight)
            nn.init.zeros_(quality_head.bias)

    def forward(
        self,
        fpn: tuple[Tensor, ...] | list[Tensor],
        metadata: dict,
        *,
        extra_directions: Tensor | None = None,
        self_attention_mask: Tensor | None = None,
    ) -> FisheyeRayDecoderOutput:
        if not fpn:
            raise ValueError("decoder requires FPN features")
        batch = fpn[0].shape[0]
        query, direction = self.query_bank(batch)
        if extra_directions is not None:
            if (
                extra_directions.ndim != 3
                or extra_directions.shape[0] != batch
                or extra_directions.shape[-1] != 3
            ):
                raise ValueError("extra_directions must be [B,N,3]")
            extra_directions = normalize_direction(
                extra_directions.to(device=direction.device, dtype=direction.dtype)
            )
            extra_query = query.new_zeros(
                batch, extra_directions.shape[1], query.shape[-1]
            )
            query = torch.cat((query, extra_query), dim=1)
            direction = torch.cat((direction, extra_directions), dim=1)
        expected_queries = query.shape[1]
        if self_attention_mask is not None:
            if self_attention_mask.shape != (expected_queries, expected_queries):
                raise ValueError(
                    "self_attention_mask must match the combined query count"
                )
            self_attention_mask = self_attention_mask.to(
                device=query.device, dtype=torch.bool
            )
        state = None
        classifications = []
        predictions = []
        qualities = []
        states = []
        attentions = []
        compatibilities = []
        for index, (layer, classifier, quality_head) in enumerate(
            zip(self.layers, self.classifiers, self.quality_heads)
        ):
            query, state, _, evidence = layer(
                query,
                direction if index == 0 else state.direction,
                state,
                fpn,
                metadata,
                self_attention_mask,
            )
            classifications.append(classifier(query))
            # Fisheye3DOD supervises planar velocity and its native Sparse4D
            # matcher/decoder therefore consumes a 10-value prediction.  Keep
            # vertical velocity in the propagated ray state but do not change
            # that established detector ABI.
            predictions.append(state.to_sparse_anchor()[..., :10])
            qualities.append(quality_head(query))
            states.append(state)
            attentions.append(evidence.attention)
            compatibilities.append(evidence.level_compatibility)
        return FisheyeRayDecoderOutput(
            classification=tuple(classifications),
            prediction=tuple(predictions),
            quality=tuple(qualities),
            states=tuple(states),
            attention=tuple(attentions),
            level_compatibility=tuple(compatibilities),
            query=query,
        )


__all__ = [
    "FisheyeRayDecoder",
    "FisheyeRayDecoderConfig",
    "FisheyeRayDecoderOutput",
    "RayStatePositionEncoder",
    "SphericalRayQueryBank",
    "UnifiedFisheyeDecoderLayer",
    "state_log_angular_extent",
]
