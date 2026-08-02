"""Native OCamCalib projection used by the released ROSA experiment."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

PROJECTION_CONTRACTS = ("legacy_front_only", "full_fov")


def _horner(coefficients: Tensor, value: Tensor) -> Tensor:
    terms = coefficients.unbind(dim=-1)
    if len(terms) == 1:
        return terms[0].expand_as(value).clone()
    # The first multiply already materializes a full value-shaped tensor.
    # Keep the leading coefficient as a broadcast view until that operation
    # instead of cloning a full temporary that is immediately discarded.
    output = terms[-1]
    for coefficient in reversed(terms[:-1]):
        output = output * value + coefficient.expand_as(value)
    return output


def _horner_inference_storage_reuse(
    coefficients: Tensor,
    value: Tensor,
) -> Tensor:
    """Evaluate Horner terms in one owned buffer during no-grad inference."""

    if torch.is_grad_enabled():
        raise RuntimeError(
            "OCam inference storage reuse requires torch.no_grad() or "
            "torch.inference_mode()"
        )
    terms = coefficients.unbind(dim=-1)
    if len(terms) == 1:
        return terms[0].expand_as(value).clone()
    output = terms[-1] * value
    output.add_(terms[-2].expand_as(value))
    for coefficient in reversed(terms[:-2]):
        output.mul_(value).add_(coefficient.expand_as(value))
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
        # Keep the historical reduction order as the default exact path.
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


def _camera_xyz_affine(points: Tensor, lidar2cam: Tensor) -> Tensor:
    """Evaluate the same rigid transform with a tolerance-gated 3x3 path."""

    if lidar2cam.ndim != 4 or lidar2cam.shape[-2:] != (4, 4):
        raise ValueError("lidar2cam must be [B,C,4,4]")
    if points.shape[-1] != 3:
        raise ValueError("points must end in XYZ coordinates")
    batch, cameras = lidar2cam.shape[:2]
    rotation = lidar2cam[..., :3, :3]
    if points.ndim == 5:
        if points.shape[1] != cameras:
            raise ValueError(
                "camera-conditioned points must be [B,C,N,P,3] with the "
                "same camera count as lidar2cam"
            )
        rotated = torch.bmm(
            rotation.reshape(batch * cameras, 3, 3),
            points.reshape(batch * cameras, -1, 3).transpose(1, 2),
        )
    else:
        if points.ndim != 4:
            raise ValueError("points must be [B,N,P,3] or [B,C,N,P,3]")
        rotated = torch.bmm(
            rotation.reshape(batch, cameras * 3, 3),
            points.reshape(batch, -1, 3).transpose(1, 2),
        )
    camera_xyz = rotated.reshape(
        batch,
        cameras,
        3,
        points.shape[-3],
        points.shape[-2],
    ).movedim(2, -1)
    # ``camera_xyz`` owns fresh bmm storage, so translation needs no new output.
    return camera_xyz.add_(lidar2cam[..., :3, 3][:, :, None, None, :])


def _sanitize_camera_xyz(camera_xyz: Tensor) -> tuple[Tensor, Tensor]:
    """Make undefined rays harmless before differentiable projection math."""

    finite_and_bounded = torch.isfinite(camera_xyz).all(dim=-1) & (
        camera_xyz.detach().abs().amax(dim=-1) < 1e4
    )
    bounded = torch.where(
        finite_and_bounded[..., None],
        camera_xyz,
        0.0,
    )
    radius_squared = (bounded * bounded).sum(dim=-1)
    geometry_valid = finite_and_bounded & radius_squared.gt(1e-6)
    # OCam's forward axis is -Z. Invalid positions are evaluated at this
    # finite ray and masked after projection, preventing NaN gradients at the
    # camera origin without changing any valid projection.
    safe = torch.where(geometry_valid[..., None], bounded, 0.0)
    safe[..., 2].masked_fill_(~geometry_valid, -1.0)
    return safe, geometry_valid


def _sanitize_camera_xyz_inference_storage_reuse(
    camera_xyz: Tensor,
) -> tuple[Tensor, Tensor]:
    """Sanitize an owned transform output in place for ordinary inference."""

    if torch.is_grad_enabled():
        raise RuntimeError(
            "OCam inference storage reuse requires torch.no_grad() or "
            "torch.inference_mode()"
        )
    finite_and_bounded = torch.isfinite(camera_xyz).all(dim=-1) & (
        camera_xyz.detach().abs().amax(dim=-1) < 1e4
    )
    # ``camera_xyz`` is the freshly allocated affine/matmul result and never
    # aliases a caller input.  Reuse it instead of retaining separate
    # bounded and safe full-size XYZ tensors.
    finite_and_bounded.logical_not_()
    camera_xyz.masked_fill_(finite_and_bounded[..., None], 0.0)
    finite_and_bounded.logical_not_()
    radius_squared = (camera_xyz * camera_xyz).sum(dim=-1)
    geometry_valid = finite_and_bounded & radius_squared.gt(1e-6)
    # Reuse the validity mask itself while filling invalid entries, then
    # restore its public polarity before returning it.
    geometry_valid.logical_not_()
    camera_xyz.masked_fill_(geometry_valid[..., None], 0.0)
    camera_xyz[..., 2].masked_fill_(geometry_valid, -1.0)
    geometry_valid.logical_not_()
    return camera_xyz, geometry_valid


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
    camera_transform: str = "homogeneous_4x4",
    inference_storage_reuse: bool = False,
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
    if not isinstance(inference_storage_reuse, bool):
        raise TypeError("inference_storage_reuse must be a bool")
    if inference_storage_reuse and torch.is_grad_enabled():
        raise RuntimeError(
            "OCam inference storage reuse requires torch.no_grad() or "
            "torch.inference_mode()"
        )
    for name in ("lidar2cam", "ocam_intrinsic", "ocam_pol"):
        if name not in metas:
            raise KeyError(f"missing OCam metadata: {name}")
    lidar2cam = metas["lidar2cam"]
    intrinsic = metas["ocam_intrinsic"]
    polynomial = metas["ocam_pol"]
    if image_wh.shape != lidar2cam.shape[:2] + (2,):
        raise ValueError("image_wh must be [B,C,2]")

    if camera_transform == "homogeneous_4x4":
        camera_xyz = _camera_xyz(points, lidar2cam)
    elif camera_transform == "affine_3x3":
        camera_xyz = _camera_xyz_affine(points, lidar2cam)
    else:
        raise ValueError(f"unknown OCam camera transform: {camera_transform}")
    if inference_storage_reuse:
        safe_xyz, geometry_valid = _sanitize_camera_xyz_inference_storage_reuse(
            camera_xyz
        )
    else:
        safe_xyz, geometry_valid = _sanitize_camera_xyz(camera_xyz)
    x, y, z = safe_xyz.unbind(dim=-1)
    planar_norm = torch.sqrt(x.square() + y.square() + 1e-12)
    theta_ocam = torch.atan2(z, planar_norm)
    expanded_polynomial = polynomial[:, :, None, None]
    if inference_storage_reuse:
        rho = _horner_inference_storage_reuse(
            expanded_polynomial,
            theta_ocam,
        )
        rho.div_(planar_norm.clamp_min(1e-8))
        # Spell the complementary predicate directly.  Materializing
        # ``~(planar_norm > eps)`` creates a second full-size bool tensor in
        # every decoder stage for no semantic benefit.
        rho.masked_fill_(planar_norm <= 1e-8, 0.0)
        scale = rho
    else:
        rho = _horner(expanded_polynomial, theta_ocam)
        scale = torch.where(
            planar_norm > 1e-8,
            rho / planar_norm.clamp_min(1e-8),
            0.0,
        )
    reuse_image_xy = inference_storage_reuse and (
        x.dtype == scale.dtype and y.dtype == scale.dtype
    )
    if reuse_image_xy:
        # ``safe_xyz`` is owned by this call in the reuse path.  Projection no
        # longer needs x/y after their planar norm and theta have been formed,
        # so retain the products in those two slots instead of allocating two
        # additional full-size tensors.  Mixed polynomial/point dtypes retain
        # the parent's ordinary promotion through the fallback below.
        image_x = safe_xyz[..., 0].mul_(scale)
        image_y = safe_xyz[..., 1].mul_(scale)
    else:
        image_x, image_y = x * scale, y * scale

    a11 = intrinsic[..., 1, 0][:, :, None, None]
    a12 = intrinsic[..., 1, 1][:, :, None, None]
    a13 = intrinsic[..., 0, 2][:, :, None, None]
    a21 = intrinsic[..., 0, 0][:, :, None, None]
    a22 = intrinsic[..., 0, 1][:, :, None, None]
    a23 = intrinsic[..., 1, 2][:, :, None, None]
    if inference_storage_reuse:
        pixel_u = a11 * image_x
        pixel_u.add_(a12 * image_y).add_(a13)
        pixel_v = a21 * image_x
        pixel_v.add_(a22 * image_y).add_(a23)
    else:
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
    # In-place division is exact only when it does not suppress PyTorch's
    # normal dtype promotion (for example FP32 pixels divided by FP64 image
    # extents).  Preserve the generic parent semantics for mixed dtypes while
    # retaining the allocation-free official homogeneous-dtype path.
    reuse_normalization = inference_storage_reuse and (
        pixel_u.dtype == width.dtype and pixel_v.dtype == height.dtype
    )
    if reuse_normalization:
        normalized_u = pixel_u.div_(width.clamp_min(1))
        normalized_v = pixel_v.div_(height.clamp_min(1))
        projected = normalized_u.new_empty((*normalized_u.shape, 2))
        sentinel = normalized_u.new_tensor(2.0)
        torch.where(
            valid,
            normalized_u,
            sentinel,
            out=projected[..., 0],
        )
        torch.where(
            valid,
            normalized_v,
            sentinel,
            out=projected[..., 1],
        )
        return projected
    normalized_u = pixel_u / width.clamp_min(1)
    normalized_v = pixel_v / height.clamp_min(1)
    return torch.stack(
        (
            torch.where(valid, normalized_u, 2.0),
            torch.where(valid, normalized_v, 2.0),
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
