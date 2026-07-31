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
from .sampling import (
    DeterministicBilinearSampler,
    FusedBilinearSampler,
)
from .types import ObservationEncoding, ROSAOutput

_SIZE = slice(3, 6)
_YAW = slice(6, 8)
_LEGACY_SIGMA_POINTS = (-2.0, -1.0, 0.0, 1.0, 2.0)
# Keep the same outer search interval while avoiding the two redundant
# intermediate re-observations in the vectorized profile.
_COMPACT_SIGMA_POINTS = (-2.0, 0.0, 2.0)
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
    radial_proposal: Tensor
    radial_gate: Tensor
    radial_gate_logit: Tensor
    radial_safe: Tensor
    tangent_delta: Tensor
    yaw_raw_delta: Tensor
    yaw_gate: Tensor
    yaw_gate_logit: Tensor
    feedback: Tensor
    reliability_context: Tensor
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


def _make_compact_head(input_dims: int, output_dims: int) -> nn.Sequential:
    """A normalized linear readout for already-processed shared features."""

    return nn.Sequential(
        nn.LayerNorm(input_dims),
        nn.Linear(input_dims, output_dims),
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
    return _compose_yaw(_compose_size(anchor, size_delta), yaw_delta)


def _compose_size(anchor: Tensor, size_delta: Tensor) -> Tensor:
    return torch.cat(
        (
            anchor[..., :3],
            exact_zero_detached_base_add(
                anchor[..., _SIZE],
                size_delta,
            ),
            anchor[..., 6:],
        ),
        dim=-1,
    )


def _compose_yaw(anchor: Tensor, yaw_delta: Tensor) -> Tensor:
    return torch.cat(
        (
            anchor[..., :6],
            exact_zero_detached_base_yaw_rotation(anchor[..., _YAW], yaw_delta),
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


def _normalized_geometry_yaw(anchor: Tensor) -> Tensor:
    """Normalize yaw for geometry without changing the detector-owned box."""

    yaw = anchor[..., _YAW]
    norm = torch.linalg.vector_norm(yaw, dim=-1, keepdim=True)
    fallback = torch.zeros_like(yaw)
    fallback[..., 1] = 1.0
    normalized = torch.where(
        norm > 1e-6,
        yaw / norm.clamp_min(1e-6),
        fallback,
    )
    return torch.cat(
        (anchor[..., :6], normalized, anchor[..., 8:]),
        dim=-1,
    )


def _radial_basis(center_xy: Tensor) -> tuple[Tensor, Tensor]:
    radial = F.normalize(center_xy, dim=-1, eps=1e-6)
    tangent = torch.stack(
        (-radial[..., 1], radial[..., 0]),
        dim=-1,
    )
    return radial, tangent


def _observation_reliability_context(
    observation: ObservationEncoding,
) -> Tensor:
    """Summarize evidence that should control rather than define a correction."""

    camera_valid = observation.valid.any(dim=-1)
    valid_float = camera_valid.to(observation.camera_geometry_summary.dtype)
    valid_count = valid_float.sum(dim=-1).clamp_min(1.0)
    geometry = observation.camera_geometry_summary

    def valid_mean(value: Tensor) -> Tensor:
        return (value * valid_float).sum(dim=-1) / valid_count

    anisotropy = geometry[..., 4]
    positive_inf = torch.full_like(anisotropy, float("inf"))
    negative_inf = torch.full_like(anisotropy, float("-inf"))
    anisotropy_min = torch.where(
        camera_valid, anisotropy, positive_inf
    ).amin(dim=-1)
    anisotropy_max = torch.where(
        camera_valid, anisotropy, negative_inf
    ).amax(dim=-1)
    anisotropy_gap = torch.where(
        valid_count > 1,
        anisotropy_max - anisotropy_min,
        torch.zeros_like(anisotropy_min),
    )
    context = torch.stack(
        (
            observation.cross_camera_consistency,
            valid_float.mean(dim=-1),
            valid_mean(anisotropy.abs()),
            anisotropy_gap,
            valid_mean(geometry[..., 5]),
            valid_mean(geometry[..., 6]),
        ),
        dim=-1,
    )
    return torch.nan_to_num(context)


def _yaw_reliability_gate(
    reliability_context: Tensor,
    supported: Tensor,
    config: ROSAConfig,
) -> Tensor:
    """Return a deterministic yaw gate without adding learned parameters."""

    if config.yaw_gate_mode == "none":
        return torch.ones_like(reliability_context[..., :1])
    support = supported[..., None].to(reliability_context.dtype)
    if config.yaw_gate_mode == "support":
        return support
    anisotropy_gap = reliability_context[..., 3:4]
    camera_count = reliability_context[..., 1:2] * config.num_cameras
    overlap_gate = torch.sigmoid(
        (config.yaw_anisotropy_threshold - anisotropy_gap)
        / config.yaw_anisotropy_temperature
    )
    return (
        torch.where(
            camera_count >= 2,
            overlap_gate,
            torch.ones_like(overlap_gate),
        )
        * support
    )


def _candidate_confidence(
    probabilities: Tensor,
    supported: Tensor,
    *,
    legacy: bool,
) -> tuple[Tensor, Tensor]:
    """Summarize only supported hypotheses as entropy and top-two margin."""

    count = int(probabilities.shape[-1])
    if legacy:
        safe_probability = probabilities.clamp_min(1e-8)
        entropy = -(
            safe_probability * safe_probability.log()
        ).sum(dim=-1) / math.log(count)
        center_probability = probabilities[..., count // 2]
        margin = probabilities.max(dim=-1).values - center_probability
        return entropy, margin

    support_count = supported.sum(dim=-1)
    safe_probability = probabilities.clamp_min(1e-8)
    raw_entropy = -(
        torch.where(
            supported,
            safe_probability * safe_probability.log(),
            torch.zeros_like(safe_probability),
        )
    ).sum(dim=-1)
    entropy = torch.where(
        support_count > 1,
        raw_entropy
        / support_count.clamp_min(2).to(raw_entropy.dtype).log(),
        torch.zeros_like(raw_entropy),
    )
    supported_probability = torch.where(
        supported,
        probabilities,
        torch.full_like(probabilities, -1.0),
    )
    top_two = supported_probability.topk(2, dim=-1).values
    margin = torch.where(
        support_count > 1,
        top_two[..., 0] - top_two[..., 1],
        torch.where(
            support_count == 1,
            torch.ones_like(top_two[..., 0]),
            torch.zeros_like(top_two[..., 0]),
        ),
    )
    return entropy, margin


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
        self.sampler = (
            DeterministicBilinearSampler()
            if config.architecture_profile == "legacy"
            else FusedBilinearSampler()
        )
        self.level_head = nn.Linear(dims, config.num_levels)
        self.rosa = PointwiseROSA(config)
        self.refinement_trunk = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(dims, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dims),
            nn.SiLU(),
        )
        make_head = (
            lambda input_dims, output_dims: _make_compact_head(
                input_dims, output_dims
            )
            if config.architecture_profile == "compact"
            else _make_head(input_dims, hidden, output_dims)
        )
        self.shared_shape_head = make_head(dims, 4)
        self.shared_tangent_head = make_head(dims, 1)
        self.range_distribution_head = make_head(dims, 2)

        candidate_dims = dims * 2 + 4
        feedback_dims = (
            candidate_dims
            if config.architecture_profile == "legacy"
            else candidate_dims + 3
        )
        self.candidate_head = make_head(candidate_dims, 1)
        self.center_gate_head = make_head(feedback_dims, 1)
        self.quality_head = (
            make_head(feedback_dims, 1)
            if (
                config.architecture_profile == "legacy"
                or config.enable_quality_score
            )
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.level_head.weight)
        nn.init.zeros_(self.level_head.bias)
        self.rosa.reset_parameters()

        released_heads = tuple(
            head
            for head in (
            self.refinement_trunk,
            self.shared_shape_head,
            self.shared_tangent_head,
            self.range_distribution_head,
            self.candidate_head,
            self.center_gate_head,
            self.quality_head,
            )
            if head is not None
        )
        for head in released_heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for head in released_heads[1:]:
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
            anchor_z_is_bottom=self.cfg.anchor_z_is_bottom,
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
        observation_anchor = (
            anchor
            if self.cfg.architecture_profile == "legacy"
            else _normalized_geometry_yaw(anchor)
        )
        with torch.no_grad():
            keypoints = keypoint_generator(observation_anchor, query)
        return self._encode(
            observation_anchor,
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

    def _posterior_radial_proposal(
        self,
        candidates: _CandidateEvidence,
        current_range: Tensor,
        distribution_mean: Tensor,
        distribution_std: Tensor,
        sigma_points: Tensor,
        fixed_metric: bool,
        step: int,
    ) -> Tensor:
        """Decode a posterior correction with an exact zero uniform prior."""

        if step > 0:
            proposal_range = (
                current_range * distribution_mean.exp()
            ).clamp(
                min=self.cfg.min_candidate_range_m,
                max=self.cfg.max_candidate_range_m,
            )
            return proposal_range - current_range
        support = candidates.supported.to(candidates.probabilities.dtype)
        support_prior = support / support.sum(dim=-1, keepdim=True).clamp_min(1.0)
        support_prior = torch.where(
            candidates.supported.any(dim=-1, keepdim=True),
            support_prior,
            torch.full_like(
                support_prior,
                1.0 / candidates.probabilities.shape[-1],
            ),
        )
        posterior_residual = candidates.probabilities - support_prior
        if fixed_metric:
            return (posterior_residual * candidates.offsets).sum(dim=-1)

        expected_sigma = (
            posterior_residual
            * sigma_points[None, None, :]
        ).sum(dim=-1)
        posterior_log_ratio = (
            distribution_mean + distribution_std * expected_sigma
        )
        proposal_range = (
            current_range * posterior_log_ratio.exp()
        ).clamp(
            min=self.cfg.min_candidate_range_m,
            max=self.cfg.max_candidate_range_m,
        )
        return proposal_range - current_range

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
        candidate_count = int(offsets.shape[-1])
        if self.cfg.architecture_profile == "legacy":
            candidate_features = []
            candidate_consistency = []
            candidate_supported = []
            for index in range(candidate_count):
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
                candidate_consistency.append(
                    observation.cross_camera_consistency
                )
                candidate_supported.append(observation.supported)

            candidates = torch.stack(candidate_features, dim=2)
            consistency = torch.stack(candidate_consistency, dim=2)
            support = torch.stack(candidate_supported, dim=2)
        else:
            batch, queries, box_dims = tangent_anchor.shape
            feature_dims = query.shape[-1]
            expanded_anchor = tangent_anchor[:, :, None, :].expand(
                -1, -1, candidate_count, -1
            )
            candidate_anchor = _compose_center(
                expanded_anchor.reshape(
                    batch,
                    queries * candidate_count,
                    box_dims,
                ),
                (
                    radial[:, :, None, :]
                    * offsets[..., None]
                ).reshape(batch, queries * candidate_count, 2),
            )
            expanded_query = query[:, :, None, :].expand(
                -1, -1, candidate_count, -1
            )
            geometry_anchor = (
                candidate_anchor.detach()
                if self.cfg.detach_candidate_geometry
                else candidate_anchor
            )
            observation = self._observe(
                geometry_anchor,
                expanded_query.reshape(
                    batch,
                    queries * candidate_count,
                    feature_dims,
                ),
                fpn,
                metadata,
                keypoint_generator,
            )
            candidates = observation.object_feature.reshape(
                batch,
                queries,
                candidate_count,
                -1,
            )
            consistency = observation.cross_camera_consistency.reshape(
                batch,
                queries,
                candidate_count,
            )
            support = observation.supported.reshape(
                batch,
                queries,
                candidate_count,
            )
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
                1.0 / candidate_count,
            ),
        )
        radial_delta = (probabilities * offsets).sum(dim=-1)
        entropy, margin = _candidate_confidence(
            probabilities,
            support,
            legacy=self.cfg.architecture_profile == "legacy",
        )
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
        count = int(offsets.shape[-1])
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
            entropy=(
                torch.zeros_like(radial_delta)
                if self.cfg.architecture_profile == "legacy"
                else supported.to(radial_delta.dtype)
            ),
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
        composition_base = (
            current
            if config.architecture_profile == "legacy"
            else geometry_base
        )
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
        else:
            # Preserve a zero-valued graph edge so controlled ablations remain
            # compatible with DDP ``find_unused_parameters=False``.
            size_delta = raw_shape[..., :3] * 0.0
        if config.mask_unsupported_updates:
            size_delta = torch.where(
                observation.supported[..., None],
                size_delta,
                torch.zeros_like(size_delta),
            )

        legacy_yaw = config.yaw_update_mode == "legacy_shared"
        yaw_enabled = (
            config.enable_shape_refinement
            and config.yaw_update_steps != "off"
            and (step == 0 or config.yaw_update_steps == "both")
        )
        if legacy_yaw and yaw_enabled:
            yaw_raw_delta = raw_shape[..., 3:4].tanh() * yaw_limit
            yaw_gate = torch.ones_like(yaw_raw_delta)
            yaw_gate_logit = torch.zeros_like(yaw_raw_delta)
            if config.mask_unsupported_updates:
                yaw_raw_delta = torch.where(
                    observation.supported[..., None],
                    yaw_raw_delta,
                    torch.zeros_like(yaw_raw_delta),
                )
                yaw_gate = torch.where(
                    observation.supported[..., None],
                    yaw_gate,
                    torch.zeros_like(yaw_gate),
                )
            yaw_delta = yaw_raw_delta
            shape_anchor = _compose_shape(
                composition_base,
                size_delta,
                yaw_delta,
            )
        else:
            yaw_raw_delta = raw_shape[..., 3:4] * 0.0
            yaw_gate = torch.zeros_like(yaw_raw_delta)
            yaw_gate_logit = torch.zeros_like(yaw_raw_delta)
            yaw_delta = torch.zeros_like(yaw_raw_delta)
            shape_anchor = _compose_size(composition_base, size_delta)

        # Corrected profiles optimize each pass from a detached box. Published
        # training intentionally retains its historical step-2 gradient path.
        tangent_base = (
            current
            if config.architecture_profile == "legacy"
            else geometry_base
        )
        _, tangent = _radial_basis(tangent_base[..., :2])
        tangent_raw = self.shared_tangent_head(shared).squeeze(-1)
        tangent_delta = (
            tangent_raw.tanh() * config.max_tangent_delta_m
            if config.enable_tangent
            else tangent_raw * 0.0
        )
        if config.mask_unsupported_updates:
            tangent_delta = torch.where(
                observation.supported,
                tangent_delta,
                torch.zeros_like(tangent_delta),
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

        scalar_feedback = [
            candidates.radial_delta,
            candidates.entropy,
            candidates.margin,
            distribution_std,
        ]
        if config.architecture_profile != "legacy":
            scalar_feedback.extend(
                (
                    candidates.supported.to(feature.dtype).mean(dim=-1),
                    candidates.supported[..., candidates.supported.shape[-1] // 2].to(
                        feature.dtype
                    ),
                    torch.full_like(
                        candidates.radial_delta,
                        1.0 if step == 0 else 0.0,
                    ),
                )
            )
        feedback = torch.cat(
            (
                feature,
                candidates.weighted_feature,
                torch.stack(tuple(scalar_feedback), dim=-1),
            ),
            dim=-1,
        )
        reliability_context = _observation_reliability_context(observation)
        radial_supported = observation.supported & candidates.supported.any(dim=2)
        if config.center_update_mode == "legacy_candidate_gate":
            radial_proposal = candidates.radial_delta
            radial_gate_logit = self.center_gate_head(feedback).squeeze(-1)
            radial_gate = (
                2.0 * torch.sigmoid(radial_gate_logit) - 1.0
            ).clamp(0.0, 1.0)
        else:
            radial_proposal = self._posterior_radial_proposal(
                candidates,
                current_range,
                distribution_mean,
                distribution_std,
                sigma_points,
                fixed_metric=(
                    step == 0 and config.fixed_metric_radial_hypotheses
                ),
                step=step,
            )
            radial_gate_logit = self.center_gate_head(feedback).squeeze(-1)
            radial_gate = torch.sigmoid(radial_gate_logit)

        if config.mask_unsupported_updates:
            radial_proposal = torch.where(
                radial_supported,
                radial_proposal,
                torch.zeros_like(radial_proposal),
            )
            radial_gate = torch.where(
                radial_supported,
                radial_gate,
                torch.zeros_like(radial_gate),
            )
        if config.force_radial_gate_one:
            radial_gate = (
                radial_supported.to(radial_gate.dtype)
                if config.mask_unsupported_updates
                else torch.ones_like(radial_gate)
            ) + 0.0 * radial_gate
        radial_safe = (
            radial_gate * radial_proposal
            if config.enable_radial
            else radial_gate * radial_proposal * 0.0
        )
        center_anchor = _compose_center(
            tangent_anchor,
            radial * radial_safe[..., None],
        )

        if not legacy_yaw and yaw_enabled:
            yaw_raw_delta = raw_shape[..., 3:4].tanh() * yaw_limit
            yaw_gate_logit = torch.zeros_like(yaw_raw_delta)
            if config.yaw_gate_mode == "none":
                yaw_gate = torch.ones_like(yaw_raw_delta)
            elif config.yaw_gate_mode == "support":
                yaw_gate = observation.supported[..., None].to(
                    yaw_raw_delta.dtype
                )
            else:
                anisotropy_gap = reliability_context[..., 3:4]
                camera_count = (
                    reliability_context[..., 1:2] * config.num_cameras
                )
                overlap_gate = torch.sigmoid(
                    (
                        config.yaw_anisotropy_threshold
                        - anisotropy_gap
                    )
                    / config.yaw_anisotropy_temperature
                )
                yaw_gate = torch.where(
                    camera_count >= 2,
                    overlap_gate,
                    torch.ones_like(overlap_gate),
                )
                yaw_gate = yaw_gate * observation.supported[..., None].to(
                    yaw_gate.dtype
                )
            if config.mask_unsupported_updates:
                support = observation.supported[..., None]
                yaw_raw_delta = torch.where(
                    support, yaw_raw_delta, torch.zeros_like(yaw_raw_delta)
                )
                yaw_gate = torch.where(
                    support, yaw_gate, torch.zeros_like(yaw_gate)
                )
            yaw_delta = yaw_gate * yaw_raw_delta
            shape_anchor = _compose_yaw(shape_anchor, yaw_delta)
        updated = (
            center_anchor
            if legacy_yaw
            else _compose_yaw(center_anchor, yaw_delta)
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
            radial_proposal=radial_proposal,
            radial_gate=radial_gate,
            radial_gate_logit=radial_gate_logit,
            radial_safe=radial_safe,
            tangent_delta=tangent_delta,
            yaw_raw_delta=yaw_raw_delta,
            yaw_gate=yaw_gate,
            yaw_gate_logit=yaw_gate_logit,
            feedback=feedback,
            reliability_context=reliability_context,
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
        sigma_points = base.new_tensor(
            _LEGACY_SIGMA_POINTS
            if self.cfg.architecture_profile == "legacy"
            else _COMPACT_SIGMA_POINTS
        )

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

        quality_raw = (
            self.quality_head(final.feedback).tanh()
            if self.quality_head is not None
            else torch.zeros_like(final.feedback[..., :1])
        )
        if self.cfg.enable_quality_score:
            quality_delta = self.cfg.max_quality_delta_logit * quality_raw
            if self.cfg.architecture_profile == "legacy":
                quality_delta = (
                    quality_delta
                    * final.radial_gate.detach()[..., None]
                )
        else:
            quality_delta = quality_raw * 0.0
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
            searched_candidate_probabilities=first.candidates.probabilities,
            searched_candidate_logits=first.candidates.logits,
            searched_candidate_supported=first.candidates.supported,
            total_radial_delta=total_radial,
            total_tangent_delta=total_tangent,
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
            iteration_observation_supported=torch.stack(
                [step.observation.supported for step in steps],
                dim=2,
            ),
            iteration_radial_proposal=torch.stack(
                [step.radial_proposal for step in steps],
                dim=2,
            ),
            iteration_radial_gate=torch.stack(
                [step.radial_gate for step in steps],
                dim=2,
            ),
            iteration_radial_gate_logit=torch.stack(
                [step.radial_gate_logit for step in steps],
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
            iteration_yaw_raw_delta=torch.stack(
                [step.yaw_raw_delta for step in steps],
                dim=2,
            ),
            iteration_yaw_gate=torch.stack(
                [step.yaw_gate for step in steps],
                dim=2,
            ),
            iteration_yaw_gate_logit=torch.stack(
                [step.yaw_gate_logit for step in steps],
                dim=2,
            ),
            iteration_yaw_delta=torch.stack(
                [step.shape_yaw_delta for step in steps],
                dim=2,
            ),
            iteration_reliability_context=torch.stack(
                [step.reliability_context for step in steps],
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
