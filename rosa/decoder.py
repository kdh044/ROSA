"""ROSA decoder with a hybrid box state.

The box centre is a viewing-ray direction plus log-range; size, yaw and
velocity stay in the global frame.  Each of the six stages observes the box
keypoints in all fisheye views (ROSA observation aggregation) and applies gated
residual updates in these coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch import Tensor

from .aggregation import (
    BoxPointProjection,
    ObservationAggregation,
    AggregationOutput,
    KeypointGenerator,
    project_box_keypoints,
)
from .projection import ObservationGeometry, Projector
from .rfl import SphericalRayFeatureLifting
from .ray import normalize_direction, rotate_sin_cos, spherical_exp_map


def _global_yaw(value: Tensor) -> Tensor:
    """Normalize a global [sin, cos] yaw with an identity fallback."""

    if value.shape[-1] != 2:
        raise ValueError("yaw must end in [sin, cos]")
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    identity = value.new_tensor((0.0, 1.0))
    return torch.where(norm > 1e-6, value / norm.clamp_min(1e-6), identity)


@dataclass(frozen=True)
class BoxState:
    """Ray-aligned center with globally framed box attributes."""

    direction: Tensor
    log_range: Tensor
    log_size: Tensor
    yaw: Tensor
    velocity: Tensor

    def validate(self) -> "BoxState":
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
    def from_sparse_anchor(cls, anchor: Tensor) -> "BoxState":
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


def log_angular_extent_of(state: BoxState) -> Tensor:
    """Metric box footprint expressed as horizontal/vertical visual angle."""

    state.validate()
    size = state.log_size.exp()
    distance = state.range.clamp_min(1e-6)
    horizontal_size = size[..., :2].amax(dim=-1, keepdim=True)
    horizontal = 2.0 * torch.atan(0.5 * horizontal_size / distance)
    vertical = 2.0 * torch.atan(0.5 * size[..., 2:3] / distance)
    return torch.cat((horizontal, vertical), dim=-1).clamp_min(1e-6).log()


@dataclass(frozen=True)
class ROSADecoderConfig:
    """Configuration of the hybrid-state box decoder."""

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
    # "ocam" (Fisheye3DOD) or "pinhole_mei" (pinhole + MEI).
    projection_type: str = "ocam"
    # Fused DAF + camera RFL inference kernel (OCam only).
    fused_camera_rfl: bool = False
    dropout: float = 0.1

    @property
    def num_fixed_points(self) -> int:
        return 7

    @property
    def num_points(self) -> int:
        return self.num_fixed_points + self.num_learnable_points


class BoxPositionEncoder(nn.Module):
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

    def forward(self, state: BoxState) -> Tensor:
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




class BoxRefinement(nn.Module):
    """Gated, zero-initialized residual updates of the hybrid box state.

    Outputs ``[d_dir(2), d_log_range, d_log_size(3), d_yaw, d_velocity(3)]``.
    Size has its own Sparse4D-style regression branch, and the yaw update gets
    an extra residual from the signed face-pair features.
    """

    output_dims = 10

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, self.output_dims),
        )
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)
        # Forked RNG: these branches do not change the other initializations.
        with torch.random.fork_rng(devices=[]):
            size_hidden: list[nn.Module] = []
            for _ in range(2):
                for _ in range(2):
                    size_hidden.extend(
                        (nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True))
                    )
                size_hidden.append(nn.LayerNorm(embed_dims))
            self.size_trunk = nn.Sequential(*size_hidden, nn.Linear(embed_dims, 3))
        nn.init.zeros_(self.size_trunk[-1].weight)
        nn.init.zeros_(self.size_trunk[-1].bias)
        self.update_gate = nn.Linear(embed_dims, 4)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)
        with torch.random.fork_rng(devices=[]):
            # Only the yaw output (index 3) is used; the size outputs are unused.
            self.face_refinement = nn.Linear(3 * embed_dims, 4, bias=False)
        nn.init.zeros_(self.face_refinement.weight)
        with torch.random.fork_rng(devices=[]):
            self.face_yaw_gate = nn.Linear(embed_dims, 1)
        nn.init.zeros_(self.face_yaw_gate.weight)
        nn.init.zeros_(self.face_yaw_gate.bias)

    def forward(
        self,
        feature: Tensor,
        state: BoxState,
        state_feature: Tensor,
        signed_face_features: Tensor,
    ) -> BoxState:
        state.validate()
        refinement_feature = feature + state_feature
        raw = self.trunk(refinement_feature.float()).to(feature.dtype)
        size = self.size_trunk(refinement_feature.float()).to(feature.dtype)
        raw = torch.cat((raw[..., :3], size, raw[..., 6:]), dim=-1)
        face_yaw = self.face_refinement(
            signed_face_features.flatten(-2).float()
        ).to(feature.dtype)[..., 3:4]

        if not torch.is_grad_enabled() and feature.is_cuda:
            from projects.mmdet3d_plugin.ops import rosa_hybrid_state_update_inference

            updated = rosa_hybrid_state_update_inference(
                state.direction,
                state.log_range,
                state.log_size,
                state.yaw,
                state.velocity,
                raw,
                self.update_gate(refinement_feature.float()),
                face_yaw,
                self.face_yaw_gate(refinement_feature.float()),
            )
            return BoxState(*updated).validate()

        gates = self.update_gate(refinement_feature.float()).sigmoid().to(feature.dtype)
        center_gate, size_gate, yaw_gate, velocity_gate = gates.split(1, dim=-1)
        face_yaw_gate = (
            self.face_yaw_gate(refinement_feature.float()).sigmoid().to(feature.dtype)
        )
        yaw_residual = raw[..., 6:7] * yaw_gate + face_yaw * face_yaw_gate
        return BoxState(
            direction=spherical_exp_map(state.direction, raw[..., 0:2] * center_gate),
            log_range=state.log_range + raw[..., 2:3] * center_gate,
            log_size=state.log_size + raw[..., 3:6] * size_gate,
            yaw=rotate_sin_cos(_global_yaw(state.yaw), yaw_residual),
            velocity=state.velocity + raw[..., 7:10] * velocity_gate,
        ).validate()


class ROSADecoderLayer(nn.Module):
    """One decoder stage: self-attention, ROSA observation, FFN, box update."""

    def __init__(
        self,
        config: ROSADecoderConfig,
        projector: Projector,
        *,
        center_prefusion: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.projector = projector
        self.position_encoder = BoxPositionEncoder(config.embed_dims)
        self.self_attention = nn.MultiheadAttention(
            config.embed_dims,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(config.embed_dims)
        self.keypoints = KeypointGenerator(
            config.embed_dims,
            config.num_learnable_points,
            config.fixed_keypoint_scale,
        )
        self.aggregation = ObservationAggregation(
            config, center_prefusion=center_prefusion
        )
        self.aggregation_norm = nn.LayerNorm(config.embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dims, config.ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dims, config.embed_dims),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.embed_dims)
        self.refinement = BoxRefinement(config.embed_dims)

    def _fused_keypoint_projection(self, query, state, metadata, extent):
        """Build, project and describe the keypoints in one kernel (OCam only)."""

        from projects.mmdet3d_plugin.ops import (
            rosa_build_keypoints_project_geometry_inference,
        )

        batch, queries = query.shape[:2]
        points = self.keypoints.num_points
        cameras = self.config.num_cameras
        raw_fov = metadata.get(
            "fisheye_lens_fov_deg", metadata.get("ocam_lens_fov_deg", 220.0)
        )
        fov = torch.as_tensor(raw_fov, device=query.device, dtype=query.dtype)
        if fov.numel() == 1:
            fov = fov.reshape(1, 1).expand(batch, cameras)
        elif fov.ndim == 1 and fov.shape[0] == batch:
            fov = fov[:, None].expand(batch, cameras)
        (
            _keypoints,
            point_rays,
            uv,
            rays,
            valid,
            descriptor,
            compatibility,
        ) = rosa_build_keypoints_project_geometry_inference(
            state.direction, state.log_range, state.log_size, state.yaw,
            self.keypoints.learnable_fc(query.float()), metadata["lidar2cam"],
            metadata["image_wh"], metadata["ocam_intrinsic"],
            metadata["ocam_pol"], torch.deg2rad(fov * 0.5), extent,
            self.aggregation.level_strides, points,
            self.config.fixed_keypoint_scale,
        )
        empty = rays.new_empty((0,))
        projection = BoxPointProjection(
            uv=uv.reshape(batch, queries, points, cameras, 2),
            geometry=ObservationGeometry(
                rays=rays.reshape(batch, queries, points, cameras, 3),
                tangent=empty, jacobian=empty, lens_margin=empty, overlap=empty,
                valid=valid.reshape(batch, queries, points, cameras),
            ),
        )
        point_rays = point_rays.reshape(batch, queries, points, 3)
        return projection, point_rays, (descriptor, compatibility)

    def forward(
        self,
        query: Tensor,
        state: BoxState,
        metadata: dict,
        formatted_fpn: tuple[Tensor, Tensor, Tensor],
        self_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, BoxState, AggregationOutput, Tensor]:
        use_fused = (
            self.config.projection_type == "ocam"
            and not self.training
            and not torch.is_grad_enabled()
            and state.direction.is_cuda
        )
        if use_fused:
            from projects.mmdet3d_plugin.ops import rosa_state_geometry_inputs_inference

            position_inputs, extent = rosa_state_geometry_inputs_inference(
                state.direction,
                state.log_range,
                state.log_size,
                state.yaw,
                state.velocity,
            )
            position = self.position_encoder.encoder(position_inputs)
        else:
            position = self.position_encoder(state)
            extent = log_angular_extent_of(state)
        attended, _ = self.self_attention(
            query + position,
            query + position,
            query,
            attn_mask=self_attention_mask,
            need_weights=False,
        )
        query = self.self_norm(query + attended)

        if use_fused:
            projection, point_rays, precomputed_geometry = (
                self._fused_keypoint_projection(query, state, metadata, extent)
            )
        else:
            keypoints = self.keypoints(query, state)
            point_rays = normalize_direction(keypoints)
            projection = project_box_keypoints(self.projector, keypoints, metadata)
            precomputed_geometry = None
        evidence = self.aggregation(
            query,
            projection,
            extent,
            point_rays,
            formatted_fpn,
            precomputed_geometry,
        )
        query = self.aggregation_norm(evidence.query)
        query = self.ffn_norm(query + self.ffn(query))
        state = self.refinement(query, state, position, evidence.signed_face_features)
        return query, state, evidence, position


@dataclass(frozen=True)
class ROSADecoderOutput:
    classification: tuple[Tensor, ...]
    prediction: tuple[Tensor, ...]
    quality: tuple[Tensor, ...]
    states: tuple[BoxState, ...]
    camera_level_attention: tuple[Tensor, ...]
    point_attention: tuple[Tensor, ...]
    level_compatibility: tuple[Tensor, ...]
    raw_log_range_std: tuple[Tensor, ...]
    query: Tensor


class ROSADecoder(nn.Module):
    """Six-stage hybrid-state box decoder."""

    def __init__(
        self,
        config: ROSADecoderConfig | None = None,
        *,
        projector: Projector | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ROSADecoderConfig()
        self.projector = projector or Projector()
        self.layers = nn.ModuleList(
            ROSADecoderLayer(
                self.config,
                self.projector,
                # The first stage also conditions the query on its centre point.
                center_prefusion=layer_index == 0,
            )
            for layer_index in range(self.config.num_layers)
        )

        def make_task_head(output_dims: int) -> nn.Module:
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
        # Log-std of the range, trained by the ray auxiliary loss.
        self.range_uncertainty_heads = nn.ModuleList(
            nn.Linear(self.config.embed_dims, 1)
            for _ in range(self.config.num_layers)
        )
        prior = 0.01
        bias = math.log(prior / (1.0 - prior))
        for classifier in self.classifiers:
            nn.init.constant_(classifier[-1].bias, bias)
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
    ) -> ROSADecoderOutput:
        from projects.mmdet3d_plugin.ops import feature_maps_format

        state = BoxState.from_sparse_anchor(anchor)
        query = instance_feature
        formatted_fpn = feature_maps_format(fpn)

        classifications = []
        predictions = []
        qualities = []
        states = []
        camera_level_attention = []
        point_attention = []
        level_compatibility = []
        raw_log_range_std = []
        for layer, classifier, quality_head, uncertainty_head in zip(
            self.layers,
            self.classifiers,
            self.quality_heads,
            self.range_uncertainty_heads,
        ):
            query, state, evidence, position = layer(
                query,
                state,
                metadata,
                formatted_fpn,
                self_attention_mask,
            )
            task_feature = query + position
            classifications.append(classifier(task_feature))
            predictions.append(state.to_sparse_anchor()[..., :10])
            qualities.append(quality_head(task_feature))
            raw_log_range_std.append(uncertainty_head(query.detach()))
            states.append(state)
            camera_level_attention.append(evidence.camera_level_attention)
            point_attention.append(evidence.point_attention)
            level_compatibility.append(evidence.level_compatibility)
        return ROSADecoderOutput(
            classification=tuple(classifications),
            prediction=tuple(predictions),
            quality=tuple(qualities),
            states=tuple(states),
            camera_level_attention=tuple(camera_level_attention),
            point_attention=tuple(point_attention),
            level_compatibility=tuple(level_compatibility),
            raw_log_range_std=tuple(raw_log_range_std),
            query=query,
        )


__all__ = [
    "ROSADecoderConfig",
    "ROSADecoderOutput",
    "ROSADecoder",
    "BoxRefinement",
    "BoxState",
]
