"""Auxiliary ray losses on the centre direction, log-range and its uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ray import normalize_direction


@dataclass(frozen=True)
class RayAuxiliaryLossConfig:
    angular_weight: float = 0.25
    log_range_l1_weight: float = 0.25
    range_nll_weight: float = 0.10
    beta: float = 0.10


class FisheyeHybridRayAuxiliaryLoss(nn.Module):
    """Losses on matched queries: ray angle, log-range, and range uncertainty.

    The uncertainty is an auxiliary output and is not part of the box state.
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
        # Machine epsilon for numerical stability.
        std = (
            F.softplus(raw_log_range_std.to(compute_dtype))
            + torch.finfo(compute_dtype).eps
        )
        error = target_log_range.to(compute_dtype) - log_range.to(compute_dtype)
        # Excess Gaussian NLL over its per-example optimum, with the error
        # detached: non-negative and trains only the uncertainty.
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
    "RayAuxiliaryLossConfig",
]
