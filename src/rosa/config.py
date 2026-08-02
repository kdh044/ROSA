"""Validated configuration for the released ROSA refiner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

_BALANCED_PROFILE = {
    "hidden_dims": 64,
    "max_candidate_range_m": 68.0,
    "yaw_keep_loss_weight": 0.25,
    "yaw_improvement_loss_weight": 0.50,
    "center_gate_loss_weight": 0.25,
    "enable_quality_score": False,
    "quality_loss_weight": 0.0,
    "mask_unsupported_updates": True,
    "train_disabled_branch_aux": False,
    "center_update_mode": "posterior_gate",
    "yaw_update_mode": "guarded_shared",
    "yaw_update_steps": "both",
    "yaw_gate_mode": "support",
    "architecture_profile": "compact",
    "candidate_target_mode": "fixed_sigma",
    "center_target_mode": "sequential",
    "center_supervision_mode": "move_or_keep",
    "projection_contract": "full_fov",
    "anchor_z_is_bottom": False,
}

_DENSE_RFL_TOPOLOGY_CONTRACT = "rosa.rfl.dense_outer_product.v2"
_FACTORIZED_RFL_TOPOLOGY_CONTRACT = "rosa.rfl.signed_gated_cp16.v3"
_RETIRED_COMPACT_PROFILES = frozenset(("compact_v2", "compact_v2_fast"))

# Users choose one behavior profile.  The individual values remain available
# only for release reproduction and controlled ablations.
_PROFILES = {
    "published": {},
    "legacy_release": {},
    "balanced": _BALANCED_PROFILE,
    "fast": {
        **_BALANCED_PROFILE,
        "refine_topk": 50,
        "num_refinement_steps": 1,
    },
}


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
    yaw_loss_beta_rad: float = 0.10
    yaw_keep_threshold_rad: float = math.radians(5.0)
    center_gate_margin_m: float = 0.05
    yaw_anisotropy_threshold: float = 0.40
    yaw_anisotropy_temperature: float = 0.05

    shape_loss_weight: float = 1.0
    yaw_loss_weight: float = 1.0
    yaw_keep_loss_weight: float = 0.0
    yaw_improvement_loss_weight: float = 0.0
    distribution_nll_weight: float = 0.25
    candidate_loss_weight: float = 0.50
    radial_offset_loss_weight: float = 0.50
    center_gate_loss_weight: float = 0.0
    center_move_loss_weight: float = 1.0
    center_keep_loss_weight: float = 1.0
    tangent_loss_weight: float = 0.25
    quality_loss_weight: float = 0.0
    intermediate_step_loss_weight: float = 0.50
    yaw_class_loss_weights: tuple[float, float, float, float, float, float] | None = (
        None
    )

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
    mask_unsupported_updates: bool = False
    train_disabled_branch_aux: bool = True
    export_diagnostics: bool = False
    anchor_z_is_bottom: bool = False

    center_update_mode: str = "legacy_candidate_gate"
    yaw_update_mode: str = "legacy_shared"
    yaw_update_steps: str = "both"
    yaw_gate_mode: str = "none"
    architecture_profile: str = "legacy"
    candidate_target_mode: str = "legacy_spacing"
    center_target_mode: str = "legacy_base"
    center_supervision_mode: str = "legacy_conflict"

    # Published checkpoints used this historical validity rule.
    projection_contract: str = "legacy_front_only"

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_mapping(cls, value: Mapping | None) -> "ROSAConfig":
        """Build a config while accepting names from the original runs."""

        raw = dict(value or {})
        raw.pop("_delete_", None)
        profile = raw.pop("profile", None)
        if profile is not None:
            if profile in _RETIRED_COMPACT_PROFILES:
                raise ValueError(
                    f"ROSA profile {profile!r} names the retired dense compact "
                    "v2 topology; use its pinned v2 source and checkpoint"
                )
            if profile not in _PROFILES:
                raise ValueError(f"unknown ROSA profile: {profile!r}")
            raw = {**_PROFILES[profile], **raw}
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
        if self.center_update_mode not in (
            "legacy_candidate_gate",
            "posterior_gate",
        ):
            raise ValueError("invalid center update mode")
        if self.yaw_update_mode not in (
            "legacy_shared",
            "guarded_shared",
        ):
            raise ValueError("invalid yaw update mode")
        if self.yaw_update_steps not in ("off", "first_only", "both"):
            raise ValueError("invalid yaw update steps")
        if self.yaw_gate_mode not in (
            "none",
            "support",
            "anisotropy_gap",
        ):
            raise ValueError("invalid yaw gate mode")
        if self.architecture_profile not in ("legacy", "compact"):
            raise ValueError("invalid architecture profile")
        if self.candidate_target_mode not in (
            "legacy_spacing",
            "fixed_sigma",
        ):
            raise ValueError("invalid candidate target mode")
        if self.center_target_mode not in (
            "legacy_base",
            "sequential",
        ):
            raise ValueError("invalid center target mode")
        if self.center_supervision_mode not in (
            "legacy_conflict",
            "move_or_keep",
        ):
            raise ValueError("invalid center supervision mode")
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
            "yaw_loss_beta_rad",
            "yaw_keep_threshold_rad",
            "yaw_anisotropy_temperature",
            "distribution_nll_weight",
            "intermediate_step_loss_weight",
        )
        for name in positive_fields:
            number = float(getattr(self, name))
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{name} must be finite and positive")

        nonnegative_fields = (
            "yaw_loss_weight",
            "yaw_keep_loss_weight",
            "yaw_improvement_loss_weight",
            "center_gate_loss_weight",
            "center_gate_margin_m",
            "yaw_anisotropy_threshold",
            "shape_loss_weight",
            "candidate_loss_weight",
            "radial_offset_loss_weight",
            "center_move_loss_weight",
            "center_keep_loss_weight",
            "tangent_loss_weight",
            "quality_loss_weight",
        )
        for name in nonnegative_fields:
            number = float(getattr(self, name))
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{name} must be finite and non-negative")

        if self.yaw_class_loss_weights is not None:
            if (
                self.architecture_profile != "compact"
                or self.yaw_update_mode == "legacy_shared"
            ):
                raise ValueError(
                    "yaw_class_loss_weights requires compact wrapped-yaw training"
                )
            try:
                weights = tuple(float(value) for value in self.yaw_class_loss_weights)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "yaw_class_loss_weights must contain six finite positive values"
                ) from error
            if len(weights) != 6 or any(
                not math.isfinite(value) or value <= 0 for value in weights
            ):
                raise ValueError(
                    "yaw_class_loss_weights must contain six finite positive values"
                )
            object.__setattr__(self, "yaw_class_loss_weights", weights)

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

    @property
    def rfl_topology_contract(self) -> str:
        """Return the architecture-owned RFL checkpoint topology."""

        if self.architecture_profile == "compact":
            return _FACTORIZED_RFL_TOPOLOGY_CONTRACT
        return _DENSE_RFL_TOPOLOGY_CONTRACT


__all__ = ["ROSAConfig"]
