"""Projection cues for camera/level weighting.

The fisheye Jacobian at each projected point gives the local pixels per
radian; dividing an FPN stride by its singular values gives the angular size
of one feature cell, which is compared with the box's expected angular extent.
"""

from __future__ import annotations

import torch
from torch import Tensor


def angular_fpn_geometry(
    tangent: Tensor,
    jacobian: Tensor,
    lens_margin: Tensor,
    overlap: Tensor,
    valid: Tensor,
    level_strides: Tensor,
    log_angular_extent: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build the distortion-aware descriptor and FPN compatibility score.

    The singular values of the fisheye Jacobian are pixels per radian along
    its local principal axes.  Dividing an FPN stride by those values gives
    the angular footprint of one feature cell.  Compatibility compares that
    footprint to the query's expected angular object extent.
    """

    if tangent.ndim != 4 or tangent.shape[-1] != 2:
        raise ValueError("tangent must be [B,Q,C,2]")
    prefix = tangent.shape[:-1]
    if jacobian.shape != prefix + (2, 2):
        raise ValueError("jacobian must align with tangent and end in [2,2]")
    if lens_margin.shape != prefix + (1,) or overlap.shape != prefix + (1,):
        raise ValueError("lens_margin and overlap must be [B,Q,C,1]")
    if valid.shape != prefix or valid.dtype is not torch.bool:
        raise ValueError("valid must be boolean [B,Q,C]")
    if level_strides.ndim != 1 or level_strides.numel() == 0:
        raise ValueError("level_strides must be a non-empty vector")
    if log_angular_extent.shape != prefix[:2] + (2,):
        raise ValueError("log_angular_extent must be [B,Q,2]")

    # Stop gradients through the singular values: they coincide on
    # isotropic regions, where the derivative is undefined (sqrt'(0)).
    safe_jacobian = torch.nan_to_num(
        jacobian.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).detach()
    # Closed-form singular values of a 2x2 matrix.
    a, b = safe_jacobian[..., 0, :].unbind(dim=-1)
    c, d = safe_jacobian[..., 1, :].unbind(dim=-1)
    trace = a.square() + b.square() + c.square() + d.square()
    determinant_squared = (a * d - b * c).square()
    discriminant = (trace.square() - 4.0 * determinant_squared).clamp_min(0.0)
    root = discriminant.sqrt()
    singular = torch.stack(
        (
            ((trace + root) * 0.5).clamp_min(1e-8).sqrt(),
            ((trace - root) * 0.5).clamp_min(1e-8).sqrt(),
        ),
        dim=-1,
    ).clamp_min(1e-4)
    log_singular = singular.log()
    log_stride = level_strides.float().log()
    log_cell_extent = log_stride.view(1, 1, 1, -1, 1) - log_singular[..., None, :]
    expected = log_angular_extent.float()[..., None, None, :]
    compatibility = -(log_cell_extent - expected).square().mean(dim=-1)

    levels = level_strides.numel()
    tangent_l = tangent.float()[..., None, :].expand(*prefix, levels, 2)
    singular_l = log_singular[..., None, :].expand(*prefix, levels, 2)
    anisotropy = (
        (log_singular[..., 0:1] - log_singular[..., 1:2])
        .abs()[..., None, :]
        .expand(*prefix, levels, 1)
    )
    margin_l = (
        torch.nan_to_num(lens_margin.float())
        .clamp(0.0, 1.0)[..., None, :]
        .expand(*prefix, levels, 1)
    )
    overlap_l = (
        torch.nan_to_num(overlap.float())
        .clamp(0.0, 1.0)[..., None, :]
        .expand(*prefix, levels, 1)
    )
    stride_l = log_stride.view(1, 1, 1, levels, 1).expand(*prefix, levels, 1)
    descriptor = torch.cat(
        (
            tangent_l,
            singular_l,
            anisotropy,
            margin_l,
            overlap_l,
            stride_l,
            compatibility[..., None],
        ),
        dim=-1,
    )
    descriptor = torch.where(
        valid[..., None, None], descriptor, torch.zeros_like(descriptor)
    )
    compatibility = torch.where(
        valid[..., None], compatibility, torch.zeros_like(compatibility)
    )
    return descriptor, compatibility




__all__ = ["angular_fpn_geometry"]
