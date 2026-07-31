"""Typed outputs shared by ROSA modules."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ObservationEncoding:
    """Camera- and point-aggregated evidence for each object query."""

    object_feature: Tensor
    point_features: Tensor
    camera_weights: Tensor
    valid: Tensor
    supported: Tensor
    cross_camera_consistency: Tensor
    camera_weight_mean: Tensor
    camera_geometry_summary: Tensor


@dataclass
class ROSAOutput:
    """Refined boxes and compact diagnostics used by training and analysis."""

    final_anchor: Tensor
    shape1_anchor: Tensor
    candidate_probabilities: Tensor
    candidate_logits: Tensor
    candidate_features: Tensor
    candidate_supported: Tensor
    radial_raw: Tensor
    radial_gate: Tensor
    radial_safe: Tensor
    tangent_delta: Tensor
    shape1_size_delta: Tensor
    shape1_yaw_delta: Tensor
    shape2_size_delta: Tensor
    shape2_yaw_delta: Tensor
    quality_delta: Tensor
    initial_supported: Tensor
    posterior_entropy: Tensor
    posterior_margin: Tensor
    distribution_mean: Tensor | None = None
    distribution_std: Tensor | None = None
    candidate_offsets: Tensor | None = None
    iteration_base_anchors: Tensor | None = None
    iteration_anchors: Tensor | None = None
    iteration_distribution_mean: Tensor | None = None
    iteration_distribution_std: Tensor | None = None
    iteration_candidate_offsets: Tensor | None = None
    iteration_candidate_probabilities: Tensor | None = None
    iteration_candidate_supported: Tensor | None = None
    iteration_radial_safe: Tensor | None = None
    iteration_tangent_delta: Tensor | None = None
    iteration_camera_weight_mean: Tensor | None = None
    iteration_camera_geometry_summary: Tensor | None = None


__all__ = ["ObservationEncoding", "ROSAOutput"]
