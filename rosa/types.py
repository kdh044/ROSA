"""Typed inputs shared by ROSA modules."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class RayFeatureRelation:
    """Calibrated unit rays and their validity, consumed by RFL."""

    ray: Tensor
    valid: Tensor


__all__ = ["RayFeatureRelation"]
