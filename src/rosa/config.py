"""Validated configuration for the released ROSA refiner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class ROSAConfig:
    """Hyperparameters for adaptive two-pass fisheye refinement."""

    embed_dims: int = 256
    hidden_dims: int = 128
    num_keypoints: int = 13
    num_cameras: int = 4
    num_levels: int = 4
    refine_topk: int = 100
    num_refinement_steps: int = 2

    initial_log_range_std: float = 0.05
    min_log_range_std: float = 0.01
    max_log_range_std: float = 0.50
    min_candidate_range_m: float = 0.50
    max_candidate_range_m: float = 48.0
    radial_step_m: float = 0.50

    max_log_size_delta: float = 0.20
    max_yaw_delta_rad: float = 0.25
    max_tangent_delta_m: float = 0.75
    max_second_log_size_delta: float = 0.05
    max_second_yaw_delta_rad: float = 0.08
    max_quality_delta_logit: float = 1.0
    keep_threshold_m: float = 0.50
    candidate_target_sigma_m: float = 0.50

    shape_loss_weight: float = 1.0
    distribution_nll_weight: float = 0.25
    candidate_loss_weight: float = 0.50
    radial_offset_loss_weight: float = 0.50
    center_move_loss_weight: float = 1.0
    center_keep_loss_weight: float = 1.0
    tangent_loss_weight: float = 0.25
    quality_loss_weight: float = 0.0
    intermediate_step_loss_weight: float = 0.50

    enable_shape_refinement: bool = True
    enable_tangent: bool = True
    enable_radial: bool = True
    enable_quality_score: bool = True
    enable_rfl: bool = True
    enable_rosa_geometry: bool = True
    uniform_camera_fusion: bool = False
    uniform_point_fusion: bool = False
    force_radial_gate_one: bool = False
    fixed_metric_radial_hypotheses: bool = False
    detach_candidate_geometry: bool = True
    export_diagnostics: bool = False

    # Published checkpoints used this historical validity rule.
    projection_contract: str = "legacy_front_only"

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_mapping(cls, value: Mapping | None) -> "ROSAConfig":
        """Build a config while accepting names from the original runs."""

        raw = dict(value or {})
        aliases = {
            "enable_shape1": "enable_shape_refinement",
            "max_shape2_log_size_delta": "max_second_log_size_delta",
            "max_shape2_yaw_delta_rad": "max_second_yaw_delta_rad",
        }
        for old_name, new_name in aliases.items():
            if old_name in raw:
                if new_name in raw:
                    raise ValueError(f"both {old_name} and {new_name} were provided")
                raw[new_name] = raw.pop(old_name)

        removed = {
            "radial_proposal_mode": "learned_log_gaussian",
            "second_step_candidate_search": False,
            "enable_shape2_feedback": True,
        }
        for name, required in removed.items():
            if name in raw and raw.pop(name) != required:
                raise ValueError(
                    f"{name}={required!r} is required by the release model"
                )

        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown ROSA options: {unknown}")
        return cls(**raw)

    def _validate(self) -> None:
        if (
            self.embed_dims <= 0
            or self.hidden_dims <= 0
            or self.num_keypoints != 13
            or self.num_cameras != 4
            or self.num_levels <= 0
            or self.refine_topk <= 0
        ):
            raise ValueError("invalid ROSA dimensions")
        if self.num_refinement_steps not in (1, 2):
            raise ValueError("num_refinement_steps must be one or two")
        if self.projection_contract not in (
            "legacy_front_only",
            "full_fov",
        ):
            raise ValueError("invalid projection contract")
        if not (
            self.min_log_range_std
            < self.initial_log_range_std
            <= self.max_log_range_std
        ):
            raise ValueError("invalid log-range standard deviation bounds")
        if self.min_candidate_range_m >= self.max_candidate_range_m:
            raise ValueError("invalid candidate range bounds")

        positive_fields = (
            "initial_log_range_std",
            "min_log_range_std",
            "max_log_range_std",
            "min_candidate_range_m",
            "max_candidate_range_m",
            "radial_step_m",
            "max_log_size_delta",
            "max_yaw_delta_rad",
            "max_tangent_delta_m",
            "max_second_log_size_delta",
            "max_second_yaw_delta_rad",
            "max_quality_delta_logit",
            "keep_threshold_m",
            "candidate_target_sigma_m",
            "distribution_nll_weight",
            "intermediate_step_loss_weight",
        )
        for name in positive_fields:
            number = float(getattr(self, name))
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{name} must be finite and positive")

    # Read-only compatibility properties used by the original MM config.
    @property
    def radial_proposal_mode(self) -> str:
        return "learned_log_gaussian"

    @property
    def second_step_candidate_search(self) -> bool:
        return False

    @property
    def enable_shape1(self) -> bool:
        return self.enable_shape_refinement

    @property
    def max_shape2_log_size_delta(self) -> float:
        return self.max_second_log_size_delta

    @property
    def max_shape2_yaw_delta_rad(self) -> float:
        return self.max_second_yaw_delta_rad


__all__ = ["ROSAConfig"]
