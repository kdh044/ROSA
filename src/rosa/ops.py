"""Small identity-safe tensor operations used by ROSA."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def orthonormal_bev_radial_basis(center_xy: Tensor) -> tuple[Tensor, Tensor]:
    """Return a deterministic orthonormal radial/tangent frame in BEV.

    The radial direction is undefined exactly at the ego origin.  Falling
    back to the positive x-axis keeps center refinement trainable there and
    avoids the all-zero "basis" returned by ``F.normalize`` for a zero
    vector.  Non-finite detector boxes use the same fail-safe frame rather
    than propagating NaNs through candidate construction and diagnostics.
    """

    if center_xy.ndim < 1 or center_xy.shape[-1] != 2:
        raise ValueError("BEV centers must end in an XY pair")
    if not center_xy.is_floating_point():
        raise TypeError("BEV centers must be floating point")

    finite = torch.isfinite(center_xy).all(dim=-1, keepdim=True)
    safe_center = torch.where(finite, center_xy, torch.zeros_like(center_xy))
    norm = torch.linalg.vector_norm(safe_center, dim=-1, keepdim=True)
    radial = F.normalize(safe_center, dim=-1, eps=1e-6)
    fallback = torch.zeros_like(radial)
    fallback[..., 0] = 1.0
    radial = torch.where(finite & norm.gt(1e-6), radial, fallback)
    tangent = torch.stack((-radial[..., 1], radial[..., 0]), dim=-1)
    return radial, tangent


class _ExactZeroDetachedBaseAdd(torch.autograd.Function):
    """Add a residual while keeping an exact-zero update bit-identical."""

    @staticmethod
    def forward(ctx, base: Tensor, delta: Tensor) -> Tensor:
        del ctx
        if base.shape != delta.shape or base.dtype != delta.dtype:
            raise ValueError("base and delta must share shape and dtype")
        return torch.where(delta == 0, base, base + delta)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        del ctx
        return None, grad_output


def exact_zero_detached_base_add(base: Tensor, delta: Tensor) -> Tensor:
    """Return ``base + delta`` with gradient assigned only to ``delta``."""

    return _ExactZeroDetachedBaseAdd.apply(base, delta)


class _ExactZeroDetachedBaseYawRotation(torch.autograd.Function):
    """Rotate ``[sin(yaw), cos(yaw)]`` and preserve exact zero updates."""

    @staticmethod
    def forward(ctx, base: Tensor, delta_yaw: Tensor) -> Tensor:
        if base.shape[-1] != 2:
            raise ValueError("yaw encoding must end in [sin, cos]")
        if delta_yaw.shape != base.shape[:-1] + (1,):
            raise ValueError("delta_yaw must contain one scalar per yaw")
        if base.dtype != delta_yaw.dtype or base.device != delta_yaw.device:
            raise ValueError("base and delta_yaw must share dtype and device")
        sine, cosine = base[..., :1], base[..., 1:]
        sin_delta, cos_delta = delta_yaw.sin(), delta_yaw.cos()
        rotated = torch.cat(
            (
                sine * cos_delta + cosine * sin_delta,
                cosine * cos_delta - sine * sin_delta,
            ),
            dim=-1,
        )
        output = torch.where(delta_yaw == 0, base, rotated)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (rotated,) = ctx.saved_tensors
        grad_delta = (
            grad_output[..., :1] * rotated[..., 1:]
            - grad_output[..., 1:] * rotated[..., :1]
        )
        return None, grad_delta


def exact_zero_detached_base_yaw_rotation(base: Tensor, delta_yaw: Tensor) -> Tensor:
    """Rotate an encoded yaw while detaching the detector-owned base."""

    return _ExactZeroDetachedBaseYawRotation.apply(base, delta_yaw)


def yaw_vector_to_angle(yaw: Tensor) -> Tensor:
    """Decode an encoded ``[sin(yaw), cos(yaw)]`` vector."""

    if yaw.shape[-1] != 2:
        raise ValueError("yaw encoding must end in [sin, cos]")
    return torch.atan2(yaw[..., 0], yaw[..., 1])


def wrapped_angle_difference(prediction: Tensor, target: Tensor) -> Tensor:
    """Return the signed shortest angular difference between yaw vectors."""

    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("yaw tensors must share shape [...,2]")
    prediction = F.normalize(prediction, dim=-1, eps=1e-6)
    target = F.normalize(target, dim=-1, eps=1e-6)
    sine = prediction[..., 0] * target[..., 1] - prediction[..., 1] * target[..., 0]
    cosine = (prediction * target).sum(dim=-1)
    return torch.atan2(sine, cosine)


def wrap_angle(angle: Tensor) -> Tensor:
    """Map a scalar angle to the differentiable principal interval [-pi, pi]."""

    return torch.atan2(angle.sin(), angle.cos())


def bitwise_equal(left: Tensor, right: Tensor) -> bool:
    """Compare floating tensors including signed zero and NaN bit patterns."""

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    integer_dtype = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
    }.get(left.dtype)
    if integer_dtype is None:
        return bool(torch.equal(left, right))
    return bool(
        torch.equal(
            left.contiguous().view(integer_dtype),
            right.contiguous().view(integer_dtype),
        )
    )


__all__ = [
    "bitwise_equal",
    "exact_zero_detached_base_add",
    "exact_zero_detached_base_yaw_rotation",
    "orthonormal_bev_radial_basis",
    "wrap_angle",
    "wrapped_angle_difference",
    "yaw_vector_to_angle",
]
