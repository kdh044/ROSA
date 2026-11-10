"""Unit-ray helpers: normalization, tangent frames, exp map, yaw rotation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

_EPS = 1e-6


def _require_last_dim(value: Tensor, width: int, name: str) -> None:
    if not torch.is_tensor(value) or value.ndim < 1 or value.shape[-1] != width:
        shape = getattr(value, "shape", None)
        raise ValueError(f"{name} must end in {width} values, got {shape}")


def normalize_direction(direction: Tensor) -> Tensor:
    """Return finite unit directions with a deterministic forward fallback."""

    _require_last_dim(direction, 3, "direction")
    finite = torch.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.vector_norm(finite, dim=-1, keepdim=True)
    fallback = torch.zeros_like(finite)
    fallback[..., 0] = 1.0
    return torch.where(norm > _EPS, finite / norm.clamp_min(_EPS), fallback)


def spherical_tangent_basis(direction: Tensor) -> tuple[Tensor, Tensor]:
    """Construct a stable orthonormal tangent frame around a unit ray.

    ``e1`` follows increasing azimuth away from the poles.  At a pole the
    reference axis changes deterministically, avoiding a zero cross product.
    ``e2`` completes the right-handed frame in the tangent plane.
    """

    ray = normalize_direction(direction)
    up = torch.zeros_like(ray)
    up[..., 2] = 1.0
    side = torch.zeros_like(ray)
    side[..., 1] = 1.0
    reference = torch.where(ray[..., 2:3].abs() < 0.95, up, side)
    e1 = F.normalize(torch.cross(reference, ray, dim=-1), dim=-1, eps=_EPS)
    e2 = F.normalize(torch.cross(ray, e1, dim=-1), dim=-1, eps=_EPS)
    return e1, e2


def spherical_exp_map(direction: Tensor, tangent_delta: Tensor) -> Tensor:
    """Move unit rays by a two-axis angular delta using the sphere exp map."""

    _require_last_dim(tangent_delta, 2, "tangent_delta")
    ray = normalize_direction(direction)
    if tangent_delta.shape[:-1] != ray.shape[:-1]:
        raise ValueError("direction and tangent_delta batch shapes must match")
    e1, e2 = spherical_tangent_basis(ray)
    tangent = tangent_delta[..., 0:1] * e1 + tangent_delta[..., 1:2] * e2
    angle = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    # torch.sinc(x / pi) is sin(x) / x and is stable at zero.
    moved = ray * angle.cos() + tangent * torch.sinc(angle / math.pi)
    return normalize_direction(moved)


def rotate_sin_cos(yaw: Tensor, delta: Tensor) -> Tensor:
    """Rotate a ``[sin(yaw), cos(yaw)]`` vector by ``delta`` radians."""

    _require_last_dim(yaw, 2, "yaw")
    if delta.shape != yaw.shape[:-1] + (1,):
        raise ValueError("delta must have shape yaw.shape[:-1] + (1,)")
    sin_yaw, cos_yaw = yaw.unbind(dim=-1)
    sin_delta = delta.squeeze(-1).sin()
    cos_delta = delta.squeeze(-1).cos()
    rotated = torch.stack(
        (
            sin_yaw * cos_delta + cos_yaw * sin_delta,
            cos_yaw * cos_delta - sin_yaw * sin_delta,
        ),
        dim=-1,
    )
    return F.normalize(rotated, dim=-1, eps=_EPS)












__all__ = [
    "normalize_direction",
    "rotate_sin_cos",
    "spherical_exp_map",
    "spherical_tangent_basis",
]
