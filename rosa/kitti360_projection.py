"""Pixel projection for KITTI-360's rectified pinhole and MEI fisheye cameras."""

from __future__ import annotations


import torch
from torch import Tensor
import torch.nn.functional as F


def _native_pixels(ray_or_xyz: Tensor, metadata: dict) -> tuple[Tensor, Tensor]:
    """Return native pixels and projection-domain validity."""

    intrinsic = metadata["camera_intrinsic"].to(
        device=ray_or_xyz.device, dtype=ray_or_xyz.dtype
    )
    distortion = metadata["mei_distortion"].to(
        device=ray_or_xyz.device, dtype=ray_or_xyz.dtype
    )
    model_id = metadata["camera_model_id"].to(
        device=ray_or_xyz.device, dtype=torch.long
    )
    x, y, z = ray_or_xyz.unbind(dim=-1)
    fx = intrinsic[..., 0, 0][:, :, None, None]
    fy = intrinsic[..., 1, 1][:, :, None, None]
    cx = intrinsic[..., 0, 2][:, :, None, None]
    cy = intrinsic[..., 1, 2][:, :, None, None]

    pinhole_z = z.clamp_min(1e-8)
    pinhole = torch.stack(
        (fx * x / pinhole_z + cx, fy * y / pinhole_z + cy), dim=-1
    )
    pinhole_valid = z.gt(0.0)

    unit = F.normalize(ray_or_xyz, dim=-1, eps=1e-8)
    ux, uy, uz = unit.unbind(dim=-1)
    # MEI parameters in the calibration-file order [k1, k2, p1, p2, xi].
    k1 = distortion[..., 0][:, :, None, None]
    k2 = distortion[..., 1][:, :, None, None]
    p1 = distortion[..., 2][:, :, None, None]
    p2 = distortion[..., 3][:, :, None, None]
    xi = distortion[..., 4][:, :, None, None]
    denominator = uz + xi
    safe_denominator = torch.where(
        denominator.abs().gt(1e-8), denominator, torch.ones_like(denominator)
    )
    mx = ux / safe_denominator
    my = uy / safe_denominator
    radius_squared = mx.square() + my.square()
    radial = 1.0 + k1 * radius_squared + k2 * radius_squared.square()
    # Like the KITTI-360 devkit (CameraFisheye.cam2image), only the radial
    # terms are applied; p1/p2 are ignored.
    del p1, p2
    distorted_x = mx * radial
    distorted_y = my * radial
    mei = torch.stack(
        (fx * distorted_x + cx, fy * distorted_y + cy), dim=-1
    )
    # Visibility is decided by the sign of z, as in the devkit.  With
    # xi > 1 the MEI model maps even rear-facing rays to finite pixels.
    mei_valid = z.gt(0.0)

    is_mei = model_id[:, :, None, None].eq(1)
    return torch.where(is_mei[..., None], mei, pinhole), torch.where(
        is_mei, mei_valid, pinhole_valid
    )


def _network_pixels(ray_or_xyz: Tensor, metadata: dict) -> tuple[Tensor, Tensor]:
    native, valid = _native_pixels(ray_or_xyz, metadata)
    affine = metadata["image_affine"].to(
        device=ray_or_xyz.device, dtype=ray_or_xyz.dtype
    )
    linear = affine[..., :2, :2][:, :, None, None]
    translation = affine[..., :2, 2][:, :, None, None]
    output = torch.matmul(linear, native[..., None]).squeeze(-1) + translation
    return output, valid


def mixed_camera_project_rays_to_pixels(
    camera_rays: Tensor,
    metadata: dict,
) -> tuple[Tensor, Tensor]:
    """Project camera-frame rays to network-input pixels.

    Args:
        camera_rays: Unit or non-unit rays shaped ``[B,Q,C,3]``.

    Returns:
        Pixel coordinates ``[B,Q,C,2]`` and model-domain validity
        ``[B,Q,C]``. Image-bound validity is deliberately left to the caller.
    """

    if camera_rays.ndim != 4 or camera_rays.shape[-1] != 3:
        raise ValueError("camera_rays must be [B,Q,C,3]")
    camera_major = camera_rays.permute(0, 2, 1, 3)[:, :, :, None, :]
    pixels, valid = _network_pixels(camera_major, metadata)
    return (
        pixels.squeeze(3).permute(0, 2, 1, 3).contiguous(),
        valid.squeeze(3).permute(0, 2, 1).contiguous(),
    )


