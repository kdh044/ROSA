"""Native OCamCalib projection used by the released ROSA experiment."""

from __future__ import annotations

import torch
from torch import Tensor

PROJECTION_CONTRACTS = ("legacy_front_only", "full_fov")


def _horner(coefficients: Tensor, value: Tensor) -> Tensor:
    terms = coefficients.unbind(dim=-1)
    output = terms[-1].expand_as(value).clone()
    for coefficient in reversed(terms[:-1]):
        output = output * value + coefficient.expand_as(value)
    return output


def _camera_xyz(points: Tensor, lidar2cam: Tensor) -> Tensor:
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must be [B,N,P,3]")
    if lidar2cam.ndim != 4 or lidar2cam.shape[-2:] != (4, 4):
        raise ValueError("lidar2cam must be [B,C,4,4]")
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    return torch.matmul(
        lidar2cam[:, :, None, None],
        homogeneous[:, None, ..., None],
    ).squeeze(-1)[..., :3]


def _half_fov_radians(
    raw_fov, *, batch: int, cameras: int, reference: Tensor
) -> Tensor:
    fov = torch.as_tensor(raw_fov, device=reference.device, dtype=reference.dtype)
    if fov.numel() == 1:
        fov = fov.reshape(1, 1).expand(batch, cameras)
    elif fov.ndim == 1 and fov.shape[0] == batch:
        fov = fov[:, None].expand(batch, cameras)
    elif fov.shape != (batch, cameras):
        raise ValueError("ocam_lens_fov_deg must be scalar, [B], or [B,C]")
    if bool((~torch.isfinite(fov) | (fov <= 0) | (fov > 360)).any()):
        raise ValueError("ocam_lens_fov_deg must be finite and in (0,360]")
    return torch.deg2rad(fov * 0.5)[:, :, None, None]


def ocam_project_points(
    points: Tensor,
    metas: dict,
    image_wh: Tensor,
    *,
    contract: str = "legacy_front_only",
) -> Tensor:
    """Project lidar-frame points to normalized native-fisheye pixels.

    ``legacy_front_only`` exactly preserves the projection validity used to
    train the published checkpoints.  It adds a front-plane test and therefore
    clips the >180° part of a 220° lens.  ``full_fov`` removes only that test
    and follows the declared lens cone.  Existing checkpoints must not be
    evaluated under the corrected contract and compared as if unchanged.
    """

    if contract not in PROJECTION_CONTRACTS:
        raise ValueError(f"unknown OCam projection contract: {contract}")
    for name in ("lidar2cam", "ocam_intrinsic", "ocam_pol"):
        if name not in metas:
            raise KeyError(f"missing OCam metadata: {name}")
    lidar2cam = metas["lidar2cam"]
    intrinsic = metas["ocam_intrinsic"]
    polynomial = metas["ocam_pol"]
    if image_wh.shape != lidar2cam.shape[:2] + (2,):
        raise ValueError("image_wh must be [B,C,2]")

    camera_xyz = _camera_xyz(points, lidar2cam)
    finite = torch.isfinite(camera_xyz).all(dim=-1)
    safe_xyz = torch.where(finite[..., None], camera_xyz, torch.zeros_like(camera_xyz))
    x, y, z = safe_xyz.unbind(dim=-1)
    planar_norm = torch.sqrt(x.square() + y.square() + 1e-12)
    theta_ocam = torch.atan2(z, planar_norm)
    expanded_polynomial = polynomial[:, :, None, None]
    rho = _horner(expanded_polynomial, theta_ocam)
    scale = torch.where(
        planar_norm > 1e-8,
        rho / planar_norm.clamp_min(1e-8),
        torch.zeros_like(rho),
    )
    image_x, image_y = x * scale, y * scale

    a11 = intrinsic[..., 1, 0][:, :, None, None]
    a12 = intrinsic[..., 1, 1][:, :, None, None]
    a13 = intrinsic[..., 0, 2][:, :, None, None]
    a21 = intrinsic[..., 0, 0][:, :, None, None]
    a22 = intrinsic[..., 0, 1][:, :, None, None]
    a23 = intrinsic[..., 1, 2][:, :, None, None]
    pixel_u = a11 * image_x + a12 * image_y + a13
    pixel_v = a21 * image_x + a22 * image_y + a23

    half_fov = _half_fov_radians(
        metas.get("ocam_lens_fov_deg", 220.0),
        batch=points.shape[0],
        cameras=lidar2cam.shape[1],
        reference=points,
    )
    incidence = torch.atan2(planar_norm, -z)
    valid = finite & (incidence <= half_fov)
    if contract == "legacy_front_only":
        valid = valid & (z < 0)

    width = image_wh[..., 0][:, :, None, None]
    height = image_wh[..., 1][:, :, None, None]
    valid = (
        valid
        & torch.isfinite(pixel_u)
        & torch.isfinite(pixel_v)
        & pixel_u.ge(0)
        & pixel_u.lt(width)
        & pixel_v.ge(0)
        & pixel_v.lt(height)
    )
    normalized_u = pixel_u / width.clamp_min(1)
    normalized_v = pixel_v / height.clamp_min(1)
    sentinel = torch.full_like(normalized_u, 2.0)
    return torch.stack(
        (
            torch.where(valid, normalized_u, sentinel),
            torch.where(valid, normalized_v, sentinel),
        ),
        dim=-1,
    )


def ocam_projection_scales(unit_ray: Tensor, metas: dict) -> tuple[Tensor, Tensor]:
    """Return local radial/tangential pixel-per-radian scales."""

    polynomial = metas["ocam_pol"].to(device=unit_ray.device, dtype=unit_ray.dtype)
    ray = torch.nan_to_num(unit_ray)
    x, y, z = ray.unbind(dim=-1)
    horizontal = torch.sqrt(x.square() + y.square() + 1e-12)
    theta = torch.atan2(z, horizontal)
    incidence = torch.atan2(horizontal, -z)
    coefficients = polynomial.unbind(dim=-1)
    rho = coefficients[-1][:, :, None, None].expand_as(theta).clone()
    for coefficient in reversed(coefficients[:-1]):
        rho = rho * theta + coefficient[:, :, None, None]
    derivative = torch.zeros_like(theta)
    for index, coefficient in enumerate(coefficients[1:], start=1):
        derivative = derivative + index * coefficient[:, :, None, None] * theta.pow(
            index - 1
        )
    radial = derivative.abs()
    tangential = torch.where(
        incidence.sin().abs() > 1e-4,
        rho.abs() / incidence.sin().abs().clamp_min(1e-4),
        radial,
    )
    radial = torch.nan_to_num(radial, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-6)
    tangential = torch.nan_to_num(
        tangential, nan=1.0, posinf=1.0, neginf=1.0
    ).clamp_min(1e-6)
    return radial, tangential


__all__ = [
    "PROJECTION_CONTRACTS",
    "ocam_project_points",
    "ocam_projection_scales",
]
