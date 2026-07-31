"""Detector-agnostic adaptive ROSA box refinement."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .attention import PointwiseROSA, masked_softmax
from .config import ROSAConfig
from .observation import (
    project_and_describe,
    sample_multiscale_features,
)
from .ops import (
    exact_zero_detached_base_add,
    exact_zero_detached_base_yaw_rotation,
)
from .sampling import DeterministicBilinearSampler
from .types import ObservationEncoding, ROSAOutput

_SIZE = slice(3, 6)
_YAW = slice(6, 8)
_SIGMA_POINTS = (-2.0, -1.0, 0.0, 1.0, 2.0)
KeypointGenerator = Callable[[Tensor, Tensor], Tensor]


@dataclass
class _CandidateEvidence:
    features: Tensor
    supported: Tensor
    logits: Tensor
    probabilities: Tensor
    offsets: Tensor
    radial_delta: Tensor
    entropy: Tensor
    margin: Tensor
    weighted_feature: Tensor


@dataclass
class _RefinementStep:
    base_anchor: Tensor
    anchor: Tensor
    shape_anchor: Tensor
    shape_size_delta: Tensor
    shape_yaw_delta: Tensor
    distribution_mean: Tensor
    distribution_std: Tensor
    candidates: _CandidateEvidence
    radial_gate: Tensor
    radial_safe: Tensor
    tangent_delta: Tensor
    feedback: Tensor
    observation: ObservationEncoding


def _make_head(
    input_dims: int,
    hidden_dims: int,
    output_dims: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dims),
        nn.Linear(input_dims, hidden_dims),
        nn.SiLU(),
        nn.Linear(hidden_dims, output_dims),
    )


def _zero_last(module: nn.Sequential) -> None:
    final = module[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("the final head layer must be Linear")
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)


def _compose_shape(
    anchor: Tensor,
    size_delta: Tensor,
    yaw_delta: Tensor,
) -> Tensor:
    return torch.cat(
        (
            anchor[..., :3],
            exact_zero_detached_base_add(
                anchor[..., _SIZE],
                size_delta,
            ),
            exact_zero_detached_base_yaw_rotation(
                anchor[..., _YAW],
                yaw_delta,
            ),
            anchor[..., 8:],
        ),
        dim=-1,
    )


def _compose_center(anchor: Tensor, delta_xy: Tensor) -> Tensor:
    if delta_xy.shape != anchor.shape[:2] + (2,):
        raise ValueError("center delta must be [B,N,2]")
    return torch.cat(
        (
            exact_zero_detached_base_add(
                anchor[..., :2],
                delta_xy,
            ),
            anchor[..., 2:],
        ),
        dim=-1,
    )


def _radial_basis(center_xy: Tensor) -> tuple[Tensor, Tensor]:
    radial = F.normalize(center_xy, dim=-1, eps=1e-6)
    tangent = torch.stack(
        (-radial[..., 1], radial[..., 0]),
        dim=-1,
    )
    return radial, tangent


class ROSARefiner(nn.Module):
    """Refine encoded 3D boxes from calibrated multiview evidence."""

    def __init__(
        self,
        config: ROSAConfig,
        projection_type: str = "ocam_fisheye",
    ) -> None:
        super().__init__()
        if projection_type != "ocam_fisheye":
            raise ValueError("ROSA supports native OCam fisheye projection only")
        self.cfg = config
        self.projection_type = projection_type

        dims = config.embed_dims
        hidden = config.hidden_dims
        self.sampler = DeterministicBilinearSampler()
        self.level_head = nn.Linear(dims, config.num_levels)
        self.rosa = PointwiseROSA(config)
        self.refinement_trunk = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(dims, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dims),
            nn.SiLU(),
        )
        self.shared_shape_head = _make_head(dims, hidden, 4)
        self.shared_tangent_head = _make_head(dims, hidden, 1)
        self.range_distribution_head = _make_head(dims, hidden, 2)

        candidate_dims = dims * 2 + 4
        self.candidate_head = _make_head(
            candidate_dims,
            hidden,
            1,
        )
        self.center_gate_head = _make_head(
            candidate_dims,
            hidden,
            1,
        )
        self.quality_head = _make_head(
            candidate_dims,
            hidden,
            1,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.level_head.weight)
        nn.init.zeros_(self.level_head.bias)
        self.rosa.reset_parameters()

        heads = (
            self.refinement_trunk,
            self.shared_shape_head,
            self.shared_tangent_head,
            self.range_distribution_head,
            self.candidate_head,
            self.center_gate_head,
            self.quality_head,
        )
        for head in heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for head in heads[1:]:
            _zero_last(head)

        target = self.cfg.initial_log_range_std - self.cfg.min_log_range_std
        inverse_softplus = math.log(math.expm1(target))
        with torch.no_grad():
            self.range_distribution_head[-1].bias[1] = inverse_softplus

    def _encode(
        self,
        anchor: Tensor,
        keypoints: Tensor,
        query: Tensor,
        fpn: Sequence[Tensor],
        metadata: dict,
    ) -> ObservationEncoding:
        uv, geometry, relation, valid = project_and_describe(
            keypoints,
            anchor,
            metadata,
            self.cfg.projection_contract,
        )
        features = sample_multiscale_features(
            fpn,
            uv,
            query,
            self.sampler,
            self.level_head,
        )
        return self.rosa(
            features,
            geometry,
            relation,
            valid,
            query,
        )

    def _observe(
        self,
        anchor: Tensor,
        query: Tensor,
        fpn: Sequence[Tensor],
        metadata: dict,
        keypoint_generator: KeypointGenerator,
    ) -> ObservationEncoding:
        with torch.no_grad():
            keypoints = keypoint_generator(anchor, query)
        return self._encode(
            anchor,
            keypoints.detach(),
            query,
            fpn,
            metadata,
        )

    def _candidate_offsets(
        self,
        current_range: Tensor,
        distribution_mean: Tensor,
        distribution_std: Tensor,
        sigma_points: Tensor,
        fixed_metric: bool,
    ) -> Tensor:
        config = self.cfg
        if fixed_metric:
            candidate_range = (
                current_range[..., None] + config.radial_step_m * sigma_points
            )
        else:
            log_ratio = (
                distribution_mean[..., None]
                + distribution_std[..., None] * sigma_points
            )
            candidate_range = current_range[..., None] * log_ratio.exp()
        candidate_range = candidate_range.clamp(
            min=config.min_candidate_range_m,
            max=config.max_candidate_range_m,
        )
        return candidate_range - current_range[..., None]

    def _search_candidates(
        self,
        feature: Tensor,
        tangent_anchor: Tensor,
        radial: Tensor,
        offsets: Tensor,
        current_range: Tensor,
        sigma_points: Tensor,
        query: Tensor,
        fpn: Sequence[Tensor],
        metadata: dict,
        keypoint_generator: KeypointGenerator,
    ) -> _CandidateEvidence:
        candidate_features = []
        candidate_consistency = []
        candidate_supported = []
        for index in range(len(_SIGMA_POINTS)):
            candidate_anchor = _compose_center(
                tangent_anchor,
                radial * offsets[..., index, None],
            )
            geometry_anchor = (
                candidate_anchor.detach()
                if self.cfg.detach_candidate_geometry
                else candidate_anchor
            )
            observation = self._observe(
                geometry_anchor,
                query,
                fpn,
                metadata,
                keypoint_generator,
            )
            candidate_features.append(observation.object_feature)
            candidate_consistency.append(observation.cross_camera_consistency)
            candidate_supported.append(observation.supported)

        candidates = torch.stack(candidate_features, dim=2)
        consistency = torch.stack(candidate_consistency, dim=2)
        support = torch.stack(candidate_supported, dim=2)
        standardized = sigma_points[None, None, :, None].expand(
            feature.shape[0],
            feature.shape[1],
            -1,
            -1,
        )
        relative_offset = (offsets / current_range[..., None].clamp_min(1e-3))[
            ..., None
        ]
        candidate_input = torch.cat(
            (
                feature[:, :, None].expand_as(candidates),
                candidates,
                consistency[..., None],
                support[..., None].to(candidates.dtype),
                standardized,
                relative_offset,
            ),
            dim=-1,
        )
        logits = self.candidate_head(candidate_input).squeeze(-1)
        probabilities = masked_softmax(logits, support, dim=2)
        any_candidate = support.any(dim=2)
        probabilities = torch.where(
            any_candidate[..., None],
            probabilities,
            torch.full_like(
                probabilities,
                1.0 / len(_SIGMA_POINTS),
            ),
        )
        radial_delta = (probabilities * offsets).sum(dim=-1)
        safe_probability = probabilities.clamp_min(1e-8)
        entropy = -(safe_probability * safe_probability.log()).sum(dim=-1) / math.log(
            len(_SIGMA_POINTS)
        )
        center_probability = probabilities[
            ...,
            len(_SIGMA_POINTS) // 2,
        ]
        margin = probabilities.max(dim=-1).values - center_probability
        weighted_feature = (probabilities[..., None] * candidates).sum(dim=2)
        return _CandidateEvidence(
            features=candidates,
            supported=support,
            logits=logits,
            probabilities=probabilities,
            offsets=offsets,
            radial_delta=radial_delta,
            entropy=entropy,
            margin=margin,
            weighted_feature=weighted_feature,
        )

    def _direct_second_pass(
        self,
        feature: Tensor,
        supported: Tensor,
        offsets: Tensor,
        current_range: Tensor,
        distribution_mean: Tensor,
    ) -> _CandidateEvidence:
        count = len(_SIGMA_POINTS)
        candidates = feature[:, :, None].expand(-1, -1, count, -1)
        support = supported[..., None].expand(-1, -1, count)
        direct_range = (current_range * distribution_mean.exp()).clamp(
            min=self.cfg.min_candidate_range_m,
            max=self.cfg.max_candidate_range_m,
        )
        radial_delta = direct_range - current_range
        return _CandidateEvidence(
            features=candidates,
            supported=support,
            logits=torch.zeros_like(offsets),
            probabilities=torch.full_like(offsets, 1.0 / count),
            offsets=offsets,
            radial_delta=radial_delta,
            entropy=torch.zeros_like(radial_delta),
            margin=torch.zeros_like(radial_delta),
            weighted_feature=feature,
        )

    def _refine_once(
        self,
        current: Tensor,
        query: Tensor,
        fpn: Sequence[Tensor],
        metadata: dict,
        keypoint_generator: KeypointGenerator,
        step: int,
        sigma_points: Tensor,
    ) -> _RefinementStep:
        config = self.cfg
        geometry_base = current.detach()
        observation = self._observe(
            geometry_base,
            query,
            fpn,
            metadata,
            keypoint_generator,
        )
        feature = observation.object_feature
        shared = feature + self.refinement_trunk(feature)

        raw_shape = self.shared_shape_head(shared)
        size_limit = (
            config.max_log_size_delta if step == 0 else config.max_second_log_size_delta
        )
        yaw_limit = (
            config.max_yaw_delta_rad if step == 0 else config.max_second_yaw_delta_rad
        )
        if config.enable_shape_refinement:
            size_delta = raw_shape[..., :3].tanh() * size_limit
            yaw_delta = raw_shape[..., 3:4].tanh() * yaw_limit
        else:
            size_delta = torch.zeros_like(raw_shape[..., :3])
            yaw_delta = torch.zeros_like(raw_shape[..., 3:4])
        shape_anchor = _compose_shape(
            current,
            size_delta,
            yaw_delta,
        )

        _, tangent = _radial_basis(current[..., :2])
        tangent_raw = self.shared_tangent_head(shared).squeeze(-1)
        tangent_delta = (
            tangent_raw.tanh() * config.max_tangent_delta_m
            if config.enable_tangent
            else torch.zeros_like(tangent_raw)
        )
        tangent_anchor = _compose_center(
            shape_anchor,
            tangent * tangent_delta[..., None],
        )
        radial, _ = _radial_basis(tangent_anchor[..., :2])

        distribution = self.range_distribution_head(shared)
        distribution_mean = distribution[..., 0]
        distribution_std = (
            config.min_log_range_std + F.softplus(distribution[..., 1])
        ).clamp(max=config.max_log_range_std)
        current_range = torch.linalg.vector_norm(
            tangent_anchor[..., :2],
            dim=-1,
        ).clamp_min(config.min_candidate_range_m)
        offsets = self._candidate_offsets(
            current_range,
            distribution_mean,
            distribution_std,
            sigma_points,
            fixed_metric=(step == 0 and config.fixed_metric_radial_hypotheses),
        )
        if step == 0:
            candidates = self._search_candidates(
                feature,
                tangent_anchor,
                radial,
                offsets,
                current_range,
                sigma_points,
                query,
                fpn,
                metadata,
                keypoint_generator,
            )
        else:
            candidates = self._direct_second_pass(
                feature,
                observation.supported,
                offsets,
                current_range,
                distribution_mean,
            )

        feedback = torch.cat(
            (
                feature,
                candidates.weighted_feature,
                torch.stack(
                    (
                        candidates.radial_delta,
                        candidates.entropy,
                        candidates.margin,
                        distribution_std,
                    ),
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        radial_gate = (
            2.0 * torch.sigmoid(self.center_gate_head(feedback).squeeze(-1)) - 1.0
        ).clamp(0.0, 1.0)
        if config.force_radial_gate_one:
            radial_gate = torch.ones_like(radial_gate) + 0.0 * radial_gate
        radial_safe = (
            radial_gate * candidates.radial_delta
            if config.enable_radial
            else torch.zeros_like(candidates.radial_delta)
        )
        updated = _compose_center(
            tangent_anchor,
            radial * radial_safe[..., None],
        )
        return _RefinementStep(
            base_anchor=geometry_base,
            anchor=updated,
            shape_anchor=shape_anchor,
            shape_size_delta=size_delta,
            shape_yaw_delta=yaw_delta,
            distribution_mean=distribution_mean,
            distribution_std=distribution_std,
            candidates=candidates,
            radial_gate=radial_gate,
            radial_safe=radial_safe,
            tangent_delta=tangent_delta,
            feedback=feedback,
            observation=observation,
        )

    def forward(
        self,
        baseline_anchor: Tensor,
        query: Tensor,
        fpn: Sequence[Tensor],
        metas: dict,
        keypoint_generator: KeypointGenerator,
    ) -> ROSAOutput:
        base = baseline_anchor.detach()
        query_sidecar = query.detach()
        sigma_points = base.new_tensor(_SIGMA_POINTS)

        steps: list[_RefinementStep] = []
        current = base
        for step_index in range(self.cfg.num_refinement_steps):
            step = self._refine_once(
                current,
                query_sidecar,
                fpn,
                metas,
                keypoint_generator,
                step_index,
                sigma_points,
            )
            steps.append(step)
            current = step.anchor

        first = steps[0]
        final = steps[-1]
        base_radial, base_tangent = _radial_basis(base[..., :2])
        center_delta = current[..., :2] - base[..., :2]
        total_radial = (center_delta * base_radial).sum(dim=-1)
        total_tangent = (center_delta * base_tangent).sum(dim=-1)

        quality_delta = (
            self.cfg.max_quality_delta_logit
            * self.quality_head(final.feedback).tanh()
            * final.radial_gate.detach()[..., None]
            if self.cfg.enable_quality_score
            else torch.zeros_like(final.feedback[..., :1])
        )
        quality_delta = torch.where(
            final.candidates.supported.any(dim=2)[..., None],
            quality_delta,
            torch.zeros_like(quality_delta),
        )
        if len(steps) == 2:
            second_size = final.shape_size_delta
            second_yaw = final.shape_yaw_delta
        else:
            second_size = torch.zeros_like(first.shape_size_delta)
            second_yaw = torch.zeros_like(first.shape_yaw_delta)

        return ROSAOutput(
            final_anchor=current,
            shape1_anchor=first.shape_anchor,
            candidate_probabilities=final.candidates.probabilities,
            candidate_logits=final.candidates.logits,
            candidate_features=final.candidates.features,
            candidate_supported=final.candidates.supported,
            radial_raw=total_radial,
            radial_gate=final.radial_gate,
            radial_safe=total_radial,
            tangent_delta=total_tangent,
            shape1_size_delta=first.shape_size_delta,
            shape1_yaw_delta=first.shape_yaw_delta,
            shape2_size_delta=second_size,
            shape2_yaw_delta=second_yaw,
            quality_delta=quality_delta,
            initial_supported=first.observation.supported,
            posterior_entropy=first.candidates.entropy,
            posterior_margin=first.candidates.margin,
            distribution_mean=final.distribution_mean,
            distribution_std=final.distribution_std,
            candidate_offsets=final.candidates.offsets,
            iteration_base_anchors=torch.stack(
                [step.base_anchor for step in steps],
                dim=2,
            ),
            iteration_anchors=torch.stack(
                [step.anchor for step in steps],
                dim=2,
            ),
            iteration_distribution_mean=torch.stack(
                [step.distribution_mean for step in steps],
                dim=2,
            ),
            iteration_distribution_std=torch.stack(
                [step.distribution_std for step in steps],
                dim=2,
            ),
            iteration_candidate_offsets=torch.stack(
                [step.candidates.offsets for step in steps],
                dim=2,
            ),
            iteration_candidate_probabilities=torch.stack(
                [step.candidates.probabilities for step in steps],
                dim=2,
            ),
            iteration_candidate_supported=torch.stack(
                [step.candidates.supported for step in steps],
                dim=2,
            ),
            iteration_radial_safe=torch.stack(
                [step.radial_safe for step in steps],
                dim=2,
            ),
            iteration_tangent_delta=torch.stack(
                [step.tangent_delta for step in steps],
                dim=2,
            ),
            iteration_camera_weight_mean=torch.stack(
                [step.observation.camera_weight_mean.detach() for step in steps],
                dim=2,
            ),
            iteration_camera_geometry_summary=torch.stack(
                [step.observation.camera_geometry_summary.detach() for step in steps],
                dim=2,
            ),
        )


ClosedLoopROSAConfig = ROSAConfig
ClosedLoopROSAStage = ROSARefiner
ClosedLoopROSAOutput = ROSAOutput

__all__ = [
    "ObservationEncoding",
    "PointwiseROSA",
    "ROSAConfig",
    "ROSAOutput",
    "ROSARefiner",
]
