"""Spherical query state used by the native-fisheye decoder.

The V2 decoder does not carry a box as its primary localization state.  A
query lives on ``S^2 x R+``: a unit ego-frame direction and a positive
Euclidean range.  Box shape, heading, and velocity remain attached to that
ray so the state can be decoded to the ordinary Sparse4D 11-value box
carrier at every decoder boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
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
    """Rotate a ``[sin(yaw), cos(yaw)]`` carrier without angle unwrapping."""

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


def compose_sin_cos(yaw: Tensor, delta_yaw: Tensor) -> Tensor:
    """Compose two normalized ``[sin, cos]`` yaw carriers."""

    _require_last_dim(yaw, 2, "yaw")
    _require_last_dim(delta_yaw, 2, "delta_yaw")
    if yaw.shape != delta_yaw.shape:
        raise ValueError("yaw and delta_yaw shapes must match")
    sin_yaw, cos_yaw = F.normalize(yaw, dim=-1, eps=_EPS).unbind(dim=-1)
    sin_delta, cos_delta = F.normalize(delta_yaw, dim=-1, eps=_EPS).unbind(dim=-1)
    return torch.stack(
        (
            sin_yaw * cos_delta + cos_yaw * sin_delta,
            cos_yaw * cos_delta - sin_yaw * sin_delta,
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class RayState:
    """Structured localization state for one decoder stage.

    Shapes share the same leading dimensions, conventionally ``[B, Q]``.
    ``yaw`` is relative to the horizontal bearing of ``direction`` and
    ``velocity`` stores radial, tangential, and vertical components.
    """

    direction: Tensor
    log_range: Tensor
    log_range_std: Tensor
    log_size: Tensor
    yaw: Tensor
    velocity: Tensor

    def validate(self) -> "RayState":
        _require_last_dim(self.direction, 3, "direction")
        _require_last_dim(self.log_range, 1, "log_range")
        _require_last_dim(self.log_range_std, 1, "log_range_std")
        _require_last_dim(self.log_size, 3, "log_size")
        _require_last_dim(self.yaw, 2, "yaw")
        _require_last_dim(self.velocity, 3, "velocity")
        leading = self.direction.shape[:-1]
        for name in (
            "log_range",
            "log_range_std",
            "log_size",
            "yaw",
            "velocity",
        ):
            if getattr(self, name).shape[:-1] != leading:
                raise ValueError(f"{name} leading shape does not match direction")
        return self

    @property
    def range(self) -> Tensor:
        return self.log_range.exp()

    @property
    def range_std(self) -> Tensor:
        return self.log_range_std.exp()

    @property
    def center(self) -> Tensor:
        return normalize_direction(self.direction) * self.range

    def to_sparse_anchor(self) -> Tensor:
        """Decode to ``[x,y,z,logw,logl,logh,sin,cos,vx,vy,vz]``."""

        self.validate()
        direction = normalize_direction(self.direction)
        center = direction * self.range

        horizontal = direction[..., :2]
        horizontal_norm = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
        radial = torch.where(
            horizontal_norm > _EPS,
            horizontal / horizontal_norm.clamp_min(_EPS),
            torch.tensor((1.0, 0.0), device=direction.device, dtype=direction.dtype),
        )
        tangent = torch.stack((-radial[..., 1], radial[..., 0]), dim=-1)

        sin_bearing = radial[..., 1]
        cos_bearing = radial[..., 0]
        sin_relative, cos_relative = F.normalize(self.yaw, dim=-1, eps=_EPS).unbind(
            dim=-1
        )
        global_yaw = torch.stack(
            (
                sin_bearing * cos_relative + cos_bearing * sin_relative,
                cos_bearing * cos_relative - sin_bearing * sin_relative,
            ),
            dim=-1,
        )

        velocity_xy = (
            self.velocity[..., 0:1] * radial + self.velocity[..., 1:2] * tangent
        )
        velocity = torch.cat((velocity_xy, self.velocity[..., 2:3]), dim=-1)
        return torch.cat((center, self.log_size, global_yaw, velocity), dim=-1)

    @classmethod
    def from_sparse_anchor(
        cls,
        anchor: Tensor,
        *,
        range_std: float = 0.5,
    ) -> "RayState":
        """Encode an ordinary 11-value anchor into the spherical carrier."""

        _require_last_dim(anchor, 11, "anchor")
        if range_std <= 0.0:
            raise ValueError("range_std must be positive")
        center = anchor[..., :3]
        distance = torch.linalg.vector_norm(center, dim=-1, keepdim=True)
        direction = normalize_direction(center)
        log_range = distance.clamp_min(_EPS).log()
        log_range_std = torch.full_like(log_range, math.log(range_std))

        radial = F.normalize(direction[..., :2], dim=-1, eps=_EPS)
        radial = torch.where(
            torch.linalg.vector_norm(direction[..., :2], dim=-1, keepdim=True) > _EPS,
            radial,
            torch.tensor((1.0, 0.0), device=anchor.device, dtype=anchor.dtype),
        )
        tangent = torch.stack((-radial[..., 1], radial[..., 0]), dim=-1)
        sin_global, cos_global = F.normalize(anchor[..., 6:8], dim=-1, eps=_EPS).unbind(
            dim=-1
        )
        sin_bearing = radial[..., 1]
        cos_bearing = radial[..., 0]
        relative_yaw = torch.stack(
            (
                sin_global * cos_bearing - cos_global * sin_bearing,
                cos_global * cos_bearing + sin_global * sin_bearing,
            ),
            dim=-1,
        )
        velocity_xy = anchor[..., 8:10]
        local_velocity = torch.cat(
            (
                (velocity_xy * radial).sum(dim=-1, keepdim=True),
                (velocity_xy * tangent).sum(dim=-1, keepdim=True),
                anchor[..., 10:11],
            ),
            dim=-1,
        )
        return cls(
            direction=direction,
            log_range=log_range,
            log_range_std=log_range_std,
            log_size=anchor[..., 3:6],
            yaw=relative_yaw,
            velocity=local_velocity,
        )


@dataclass(frozen=True)
class RayRefinementConfig:
    """Numerically bounded update contract shared by all decoder stages."""

    embed_dims: int = 256
    hidden_dims: int = 256
    min_range_m: float = 0.5
    max_range_m: float = 68.0
    initial_range_m: float = 18.0
    min_range_std_m: float = 0.1
    max_range_std_m: float = 20.0
    initial_range_std_m: float = 8.0
    max_angular_delta_rad: float = 0.35
    max_log_range_delta: float = 0.50
    max_log_size_delta: float = 1.25
    max_yaw_delta_rad: float = math.pi
    max_velocity_delta_mps: float = 8.0
    initial_size_m: tuple[float, float, float] = (2.0, 4.0, 1.7)

    def __post_init__(self) -> None:
        if self.embed_dims <= 0 or self.hidden_dims <= 0:
            raise ValueError("refinement dimensions must be positive")
        if not 0.0 < self.min_range_m < self.max_range_m:
            raise ValueError("invalid range interval")
        if not self.min_range_m <= self.initial_range_m <= self.max_range_m:
            raise ValueError("initial_range_m must lie inside the range interval")
        if not 0.0 < self.min_range_std_m <= self.max_range_std_m:
            raise ValueError("invalid range-std interval")
        if not all(value > 0.0 for value in self.initial_size_m):
            raise ValueError("initial box size must be positive")


class FisheyeRayStateRefinement(nn.Module):
    """Create D0 state or update D1-D5 state in angular/range coordinates."""

    output_dims = 12  # angle2, range1, std1, size3, yaw2, velocity3

    def __init__(
        self,
        config: RayRefinementConfig | None = None,
        *,
        initialize: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or RayRefinementConfig()
        self.initialize = bool(initialize)
        self.trunk = nn.Sequential(
            nn.Linear(self.config.embed_dims, self.config.hidden_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.config.hidden_dims),
            nn.Linear(self.config.hidden_dims, self.output_dims),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        final = self.trunk[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if self.initialize:
            # A unit cosine makes the initial relative-yaw carrier valid while
            # every other zero output exactly reproduces its declared prior.
            final.bias.data[8] = 1.0

    def _bounded_initial_range(self, raw: Tensor) -> Tensor:
        cfg = self.config
        low = math.log(cfg.min_range_m)
        high = math.log(cfg.max_range_m)
        initial = math.log(cfg.initial_range_m)
        fraction = (initial - low) / (high - low)
        bias = math.log(fraction / (1.0 - fraction))
        return low + (high - low) * torch.sigmoid(raw + bias)

    def _bounded_std(self, raw: Tensor, base: Tensor | None) -> Tensor:
        cfg = self.config
        low = math.log(cfg.min_range_std_m)
        high = math.log(cfg.max_range_std_m)
        if base is None:
            initial = math.log(cfg.initial_range_std_m)
            fraction = (initial - low) / (high - low)
            bias = math.log(fraction / (1.0 - fraction))
            return low + (high - low) * torch.sigmoid(raw + bias)
        return (base + raw.tanh()).clamp(min=low, max=high)

    def forward(
        self,
        feature: Tensor,
        direction: Tensor,
        state: RayState | None = None,
    ) -> RayState:
        if feature.ndim != 3 or feature.shape[-1] != self.config.embed_dims:
            raise ValueError(
                "feature must be [B,Q,embed_dims], got " f"{tuple(feature.shape)}"
            )
        _require_last_dim(direction, 3, "direction")
        if direction.shape[:-1] != feature.shape[:-1]:
            raise ValueError("direction and feature leading shapes must match")
        if self.initialize != (state is None):
            expected = "no state" if self.initialize else "an existing state"
            raise ValueError(f"this refinement stage requires {expected}")

        cfg = self.config
        raw = self.trunk(feature.float()).to(dtype=feature.dtype)
        angular = raw[..., 0:2].tanh() * cfg.max_angular_delta_rad
        new_direction = spherical_exp_map(direction, angular)

        if state is None:
            log_range = self._bounded_initial_range(raw[..., 2:3])
            log_range_std = self._bounded_std(raw[..., 3:4], None)
            size_prior = raw.new_tensor(cfg.initial_size_m).log()
            log_size = size_prior + raw[..., 4:7].tanh() * cfg.max_log_size_delta
            yaw = F.normalize(raw[..., 7:9], dim=-1, eps=_EPS)
            velocity = raw[..., 9:12].tanh() * cfg.max_velocity_delta_mps
        else:
            state.validate()
            log_range = (
                state.log_range + raw[..., 2:3].tanh() * cfg.max_log_range_delta
            ).clamp(min=math.log(cfg.min_range_m), max=math.log(cfg.max_range_m))
            log_range_std = self._bounded_std(raw[..., 3:4], state.log_range_std)
            log_size = state.log_size + raw[..., 4:7].tanh() * cfg.max_log_size_delta
            # Both yaw outputs participate in the update.  Adding the unit
            # cosine makes a zero raw output an exact identity rotation.
            yaw_delta = raw[..., 7:9] + raw.new_tensor((0.0, 1.0))
            yaw = compose_sin_cos(state.yaw, yaw_delta)
            velocity = (
                state.velocity + raw[..., 9:12].tanh() * cfg.max_velocity_delta_mps
            )

        return RayState(
            direction=new_direction,
            log_range=log_range,
            log_range_std=log_range_std,
            log_size=log_size,
            yaw=yaw,
            velocity=velocity,
        ).validate()


def log_range_gaussian_nll(state: RayState, target_center: Tensor) -> Tensor:
    """Per-query Gaussian NLL for matched Euclidean object range."""

    state.validate()
    _require_last_dim(target_center, 3, "target_center")
    if target_center.shape[:-1] != state.direction.shape[:-1]:
        raise ValueError("target_center and state leading shapes must match")
    target_log_range = (
        torch.linalg.vector_norm(target_center, dim=-1, keepdim=True)
        .clamp_min(_EPS)
        .log()
    )
    inv_std = (-state.log_range_std).exp()
    residual = (target_log_range - state.log_range) * inv_std
    return 0.5 * residual.square() + state.log_range_std


__all__ = [
    "FisheyeRayStateRefinement",
    "RayRefinementConfig",
    "RayState",
    "compose_sin_cos",
    "log_range_gaussian_nll",
    "normalize_direction",
    "rotate_sin_cos",
    "spherical_exp_map",
    "spherical_tangent_basis",
]
