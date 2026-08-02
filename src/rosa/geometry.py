"""Native OCamCalib projection used by the released ROSA experiment."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

PROJECTION_CONTRACTS = ("legacy_front_only", "full_fov")


def _horner(coefficients: Tensor, value: Tensor) -> Tensor:
    terms = coefficients.unbind(dim=-1)
    output = terms[-1].expand_as(value).clone()
    for coefficient in reversed(terms[:-1]):
        output = output * value + coefficient.expand_as(value)
    return output


def _camera_xyz(points: Tensor, lidar2cam: Tensor) -> Tensor:
    if lidar2cam.ndim != 4 or lidar2cam.shape[-2:] != (4, 4):
        raise ValueError("lidar2cam must be [B,C,4,4]")
    if points.shape[-1] != 3:
        raise ValueError("points must end in XYZ coordinates")
    if points.ndim == 5:
        if points.shape[1] != lidar2cam.shape[1]:
            raise ValueError(
                "camera-conditioned points must be [B,C,N,P,3] with the "
                "same camera count as lidar2cam"
            )
        # Preserve the historical 4x4 matmul shape and arithmetic exactly,
        # while constructing the constant homogeneous coordinate in one
        # allocation instead of allocating ``ones_like`` and then ``cat``.
        homogeneous = F.pad(points, (0, 1), mode="constant", value=1.0)
        return torch.matmul(
            lidar2cam[:, :, None, None],
            homogeneous[..., None],
        ).squeeze(-1)[..., :3]
    if points.ndim != 4:
        raise ValueError("points must be [B,N,P,3] or [B,C,N,P,3]")
    homogeneous = F.pad(points, (0, 1), mode="constant", value=1.0)
    return torch.matmul(
        lidar2cam[:, :, None, None],
        homogeneous[:, None, ..., None],
    ).squeeze(-1)[..., :3]


def _sanitize_camera_xyz(camera_xyz: Tensor) -> tuple[Tensor, Tensor]:
    """Make undefined rays harmless before differentiable projection math."""

    finite_and_bounded = torch.isfinite(camera_xyz).all(dim=-1) & (
        camera_xyz.detach().abs().amax(dim=-1) < 1e4
    )
    bounded = torch.where(
        finite_and_bounded[..., None],
        camera_xyz,
        torch.zeros_like(camera_xyz),
    )
    radius_squared = (bounded * bounded).sum(dim=-1)
    geometry_valid = finite_and_bounded & radius_squared.gt(1e-6)
    # OCam's forward axis is -Z. Invalid positions are evaluated at this
    # finite ray and masked after projection, preventing NaN gradients at the
    # camera origin without changing any valid projection.
    fallback = torch.zeros_like(camera_xyz)
    fallback[..., 2] = -1.0
    safe = torch.where(geometry_valid[..., None], bounded, fallback)
    return safe, geometry_valid


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
    if isinstance(raw_fov, (int, float)):
        scalar_fov = float(raw_fov)
        if not math.isfinite(scalar_fov) or not 0 < scalar_fov <= 360:
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
    safe_xyz, geometry_valid = _sanitize_camera_xyz(camera_xyz)
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
    valid = geometry_valid & (incidence <= half_fov)
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


def ocam_projection_scales(
    unit_ray: Tensor,
    metas: dict,
    *,
    contract: str = "legacy_front_only",
) -> tuple[Tensor, Tensor]:
    """Return local radial/tangential pixel-per-radian scales.

    The published descriptor omitted OCam's affine image transform.  Preserve
    that behavior under ``legacy_front_only`` and apply the calibrated affine
    gains only for the corrected ``full_fov`` contract.
    """

    if contract not in PROJECTION_CONTRACTS:
        raise ValueError(f"unknown OCam projection contract: {contract}")

    polynomial = metas["ocam_pol"].to(device=unit_ray.device, dtype=unit_ray.dtype)
    ray = torch.nan_to_num(unit_ray)
    x, y, z = ray.unbind(dim=-1)
    horizontal_squared = x.square() + y.square()
    horizontal = torch.sqrt(horizontal_squared + 1e-12)
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
    if contract == "full_fov":
        intrinsic = metas["ocam_intrinsic"].to(
            device=unit_ray.device,
            dtype=unit_ray.dtype,
        )
        radial_x = x / horizontal
        radial_y = y / horizontal
        on_axis = horizontal_squared <= 1e-12
        radial_x = torch.where(on_axis, torch.ones_like(radial_x), radial_x)
        radial_y = torch.where(on_axis, torch.zeros_like(radial_y), radial_y)
        tangential_x = -radial_y
        tangential_y = radial_x

        a11 = intrinsic[..., 1, 0][:, :, None, None]
        a12 = intrinsic[..., 1, 1][:, :, None, None]
        a21 = intrinsic[..., 0, 0][:, :, None, None]
        a22 = intrinsic[..., 0, 1][:, :, None, None]

        radial_gain = torch.sqrt(
            (a11 * radial_x + a12 * radial_y).square()
            + (a21 * radial_x + a22 * radial_y).square()
        )
        tangential_gain = torch.sqrt(
            (a11 * tangential_x + a12 * tangential_y).square()
            + (a21 * tangential_x + a22 * tangential_y).square()
        )
        radial = radial * radial_gain
        tangential = tangential * tangential_gain

    radial = torch.nan_to_num(radial, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-6)
    tangential = torch.nan_to_num(
        tangential, nan=1.0, posinf=1.0, neginf=1.0
    ).clamp_min(1e-6)
    return radial, tangential


def ocam_full_fov_boundary(
    unit_ray: Tensor,
    radial_scale: Tensor,
    metas: dict,
    image_wh: Tensor,
) -> Tensor:
    """Return normalized support margin to the declared OCam lens cone."""

    x, y, z = torch.nan_to_num(unit_ray).unbind(dim=-1)
    horizontal = torch.sqrt(x.square() + y.square() + 1e-12)
    incidence = torch.atan2(horizontal, -z)
    half_fov = _half_fov_radians(
        metas.get("ocam_lens_fov_deg", 220.0),
        batch=unit_ray.shape[0],
        cameras=unit_ray.shape[1],
        reference=unit_ray,
    )
    angular_margin = (half_fov - incidence).clamp_min(0.0)
    minimum_extent = image_wh.to(
        device=unit_ray.device,
        dtype=unit_ray.dtype,
    ).amin(
        dim=-1
    )[:, :, None, None]
    boundary = 2.0 * radial_scale * angular_margin / minimum_extent.clamp_min(1.0)
    return torch.nan_to_num(
        boundary,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)


__all__ = [
    "PROJECTION_CONTRACTS",
    "ocam_full_fov_boundary",
    "ocam_project_points",
    "ocam_projection_scales",
]
