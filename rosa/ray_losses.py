"""Auxiliary supervision for the spherical state of matched 3D queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ray_state import RayState, log_range_gaussian_nll, normalize_direction


@dataclass(frozen=True)
class RayAuxiliaryLossConfig:
    angular_weight: float = 0.25
    log_range_l1_weight: float = 0.25
    range_nll_weight: float = 0.10
    beta: float = 0.10
    range_uncertainty_objective: str = "gaussian_nll"

    def __post_init__(self) -> None:
        if (
            min(
                self.angular_weight,
                self.log_range_l1_weight,
                self.range_nll_weight,
            )
            < 0.0
        ):
            raise ValueError("ray auxiliary loss weights must be non-negative")
        if self.beta <= 0.0:
            raise ValueError("smooth-L1 beta must be positive")
        if self.range_uncertainty_objective not in (
            "gaussian_nll",
            "detached_excess_nll",
        ):
            raise ValueError("unknown range uncertainty objective")


class FisheyeRayAuxiliaryLoss(nn.Module):
    """Supervise direction, continuous range, and predicted uncertainty."""

    def __init__(self, config: RayAuxiliaryLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or RayAuxiliaryLossConfig()

    def forward(
        self,
        state: RayState,
        target_anchor: Tensor,
        matched: Tensor,
    ) -> dict[str, Tensor]:
        state.validate()
        if (
            target_anchor.shape[:-1] != state.direction.shape[:-1]
            or target_anchor.shape[-1] < 3
        ):
            raise ValueError("target_anchor must align with state and contain XYZ")
        if (
            matched.shape != state.direction.shape[:-1]
            or matched.dtype is not torch.bool
        ):
            raise ValueError("matched must be a boolean state-leading mask")

        target_center = target_anchor[..., :3]
        target_direction = normalize_direction(target_center)
        predicted_direction = normalize_direction(state.direction)
        cosine = (predicted_direction * target_direction).sum(dim=-1).clamp(-1.0, 1.0)
        angular = 1.0 - cosine

        target_log_range = (
            torch.linalg.vector_norm(target_center, dim=-1, keepdim=True)
            .clamp_min(1e-6)
            .log()
        )
        log_range_l1 = F.smooth_l1_loss(
            state.log_range,
            target_log_range,
            reduction="none",
            beta=self.config.beta,
        ).squeeze(-1)
        range_nll = log_range_gaussian_nll(state, target_center).squeeze(-1)

        weight = matched.to(dtype=state.log_range.dtype)
        denominator = weight.sum().clamp_min(1.0)

        def reduce(value: Tensor) -> Tensor:
            return (value * weight).sum() / denominator

        return {
            "loss_ray_angular": reduce(angular) * self.config.angular_weight,
            "loss_ray_log_range": reduce(log_range_l1)
            * self.config.log_range_l1_weight,
            "loss_ray_range_nll": reduce(range_nll) * self.config.range_nll_weight,
        }


class FisheyeHybridRayAuxiliaryLoss(nn.Module):
    """Supervise a hybrid centre without bounded uncertainty state.

    Direction and log-range act directly on the physical centre.  Uncertainty
    is a per-decoder auxiliary readout, so it needs no hand-selected initial
    value or physical minimum/maximum and is never propagated as box state.
    """

    def __init__(self, config: RayAuxiliaryLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or RayAuxiliaryLossConfig()

    def forward(
        self,
        direction: Tensor,
        log_range: Tensor,
        raw_log_range_std: Tensor,
        target_anchor: Tensor,
        matched: Tensor,
    ) -> dict[str, Tensor]:
        leading = direction.shape[:-1]
        if direction.shape[-1] != 3:
            raise ValueError("direction must end in XYZ")
        if log_range.shape != leading + (1,):
            raise ValueError("log_range must align with direction")
        if raw_log_range_std.shape != leading + (1,):
            raise ValueError("raw_log_range_std must align with direction")
        if target_anchor.shape[:-1] != leading or target_anchor.shape[-1] < 3:
            raise ValueError("target_anchor must align and contain XYZ")
        if matched.shape != leading or matched.dtype is not torch.bool:
            raise ValueError("matched must be a boolean state-leading mask")

        target_center = target_anchor[..., :3]
        target_direction = normalize_direction(target_center)
        cosine = (
            (normalize_direction(direction) * target_direction)
            .sum(dim=-1)
            .clamp(-1.0, 1.0)
        )
        angular = 1.0 - cosine

        target_log_range = (
            torch.linalg.vector_norm(target_center, dim=-1, keepdim=True)
            .clamp_min(1e-6)
            .log()
        )
        log_range_l1 = F.smooth_l1_loss(
            log_range,
            target_log_range,
            reduction="none",
            beta=self.config.beta,
        ).squeeze(-1)

        compute_dtype = (
            torch.float32
            if log_range.dtype in (torch.float16, torch.bfloat16)
            else log_range.dtype
        )
        # Machine epsilon is numerical protection, not a tuned physical bound.
        std = (
            F.softplus(raw_log_range_std.to(compute_dtype))
            + torch.finfo(compute_dtype).eps
        )
        error = target_log_range.to(compute_dtype) - log_range.to(compute_dtype)
        if self.config.range_uncertainty_objective == "gaussian_nll":
            range_nll = (0.5 * (error / std).square() + std.log()).squeeze(-1)
        else:
            # Gaussian NLL is unbounded below as both mean error and sigma
            # approach zero.  Subtract its per-example optimum and detach the
            # observed error: the resulting excess NLL is non-negative,
            # scale-free, and trains uncertainty calibration without pulling
            # the detector representation toward sigma collapse.
            target_std = error.detach().abs() + torch.finfo(compute_dtype).eps
            log_ratio = std.log() - target_std.log()
            range_nll = (0.5 * torch.exp(-2.0 * log_ratio) + log_ratio - 0.5).squeeze(
                -1
            )

        weight = matched.to(dtype=compute_dtype)
        denominator = weight.sum().clamp_min(1.0)

        def reduce(value: Tensor) -> Tensor:
            return (value.to(compute_dtype) * weight).sum() / denominator

        return {
            "loss_ray_angular": reduce(angular) * self.config.angular_weight,
            "loss_ray_log_range": reduce(log_range_l1)
            * self.config.log_range_l1_weight,
            "loss_ray_range_nll": reduce(range_nll) * self.config.range_nll_weight,
        }


__all__ = [
    "FisheyeHybridRayAuxiliaryLoss",
    "FisheyeRayAuxiliaryLoss",
    "RayAuxiliaryLossConfig",
]
