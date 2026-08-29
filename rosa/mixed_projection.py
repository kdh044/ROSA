"""Projection for KITTI-360's mixed pinhole and unified-MEI cameras.

The camera backend is intentionally separate from ROSA's learned logic.  It
only converts detector-frame 3D points into normalized network-input pixels
and exposes the local pixel-per-radian scale used by the existing observation
descriptor.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def _camera_xyz(points: Tensor, lidar2cam: Tensor) -> Tensor:
    """Rigid-transform points into camera-local xyz via a [B,C,4,4] matrix.

    Ported verbatim from ``rosa.geometry._camera_xyz`` (a sibling source
    tree this port doesn't include) since only this one helper is needed
    here.
    """
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


MIXED_CAMERA_METADATA = (
    "lidar2cam",
    "camera_model_id",
    "camera_intrinsic",
    "mei_distortion",
    "image_affine",
)


def _validate_metadata(metadata: dict, image_wh: Tensor) -> None:
    missing = [name for name in MIXED_CAMERA_METADATA if name not in metadata]
    if missing:
        raise KeyError(f"mixed-camera metadata missing: {missing}")
    lidar2cam = metadata["lidar2cam"]
    if image_wh.shape != lidar2cam.shape[:2] + (2,):
        raise ValueError("image_wh must be [B,C,2]")
    if metadata["camera_model_id"].shape != lidar2cam.shape[:2]:
        raise ValueError("camera_model_id must be [B,C]")
    if metadata["camera_intrinsic"].shape != lidar2cam.shape[:2] + (3, 3):
        raise ValueError("camera_intrinsic must be [B,C,3,3]")
    if metadata["mei_distortion"].shape != lidar2cam.shape[:2] + (5,):
        raise ValueError("mei_distortion must be [B,C,5]")
    if metadata["image_affine"].shape != lidar2cam.shape[:2] + (3, 3):
        raise ValueError("image_affine must be [B,C,3,3]")


def _safe_camera_xyz(camera_xyz: Tensor) -> tuple[Tensor, Tensor]:
    finite = camera_xyz.detach().abs().amax(dim=-1).lt(1e4)
    bounded = torch.where(finite[..., None], camera_xyz, 0.0)
    geometry_valid = finite & bounded.square().sum(dim=-1).gt(1e-6)
    fallback = torch.zeros_like(bounded)
    fallback[..., 2] = 1.0
    return torch.where(geometry_valid[..., None], bounded, fallback), geometry_valid


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
    # FishBEVOD/DAPETR keeps KITTI-360's MEI vector in the calibration-file
    # order [k1, k2, p1, p2, xi].  Keep that order at the public metadata
    # boundary so the converted nuScenes records can be consumed verbatim.
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
    # KITTI-360's official devkit (kitti360scripts.helpers.project.
    # CameraFisheye.cam2image) never references p1/p2 at all -- only the
    # radial (k1, k2) terms. Matching that exactly (verified against real
    # frames in projection_parity_check.py) rather than adding an
    # unofficial tangential correction the ground truth doesn't apply.
    del p1, p2
    distorted_x = mx * radial
    distorted_y = my * radial
    mei = torch.stack(
        (fx * distorted_x + cx, fy * distorted_y + cy), dim=-1
    )
    # Visibility must be decided by the sign of z, not by the model-domain
    # condition z/||P|| + xi > 0.  KITTI-360's mirror parameter is xi = 2.21
    # (image_02) and 2.55 (image_03), and |z/||P||| <= 1, so that condition
    # holds for *every* direction including the one pointing straight out the
    # back of the camera.  Such a ray still lands at a finite radius and can
    # fall inside the image, which silently marks objects on the far side of
    # the vehicle as observed.  KITTI-360's own devkit culls on z instead:
    # ``CameraFisheye.cam2image`` returns ``norm * z / |z|`` as depth and the
    # official viewer keeps ``depth > 0``.
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


def mixed_camera_project_points_with_xyz(
    points: Tensor,
    metadata: dict,
    image_wh: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project points and retain raw camera-frame XYZ for RFL."""

    _validate_metadata(metadata, image_wh)
    camera_xyz = _camera_xyz(points, metadata["lidar2cam"])
    safe_xyz, geometry_valid = _safe_camera_xyz(camera_xyz)
    pixels, domain_valid = _network_pixels(safe_xyz, metadata)
    width = image_wh[..., 0][:, :, None, None]
    height = image_wh[..., 1][:, :, None, None]
    valid = (
        geometry_valid
        & domain_valid
        & torch.isfinite(pixels).all(dim=-1)
        & pixels[..., 0].ge(0.0)
        & pixels[..., 0].lt(width)
        & pixels[..., 1].ge(0.0)
        & pixels[..., 1].lt(height)
    )
    normalized = pixels / image_wh[:, :, None, None, :].clamp_min(1.0)
    invalid = torch.full_like(normalized, 2.0)
    return torch.where(valid[..., None], normalized, invalid), camera_xyz


def mixed_camera_project_points(
    points: Tensor,
    metadata: dict,
    image_wh: Tensor,
) -> Tensor:
    return mixed_camera_project_points_with_xyz(points, metadata, image_wh)[0]


def mixed_camera_projection_scales(
    ray: Tensor,
    metadata: dict,
    *,
    epsilon: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Estimate local radial/tangential pixel-per-radian scales."""

    ray = F.normalize(torch.nan_to_num(ray), dim=-1, eps=1e-8)
    x, y, z = ray.unbind(dim=-1)
    horizontal = torch.sqrt(x.square() + y.square())
    safe_horizontal = horizontal.clamp_min(1e-8)
    radial = torch.stack(
        (z * x / safe_horizontal, z * y / safe_horizontal, -horizontal), dim=-1
    )
    radial_fallback = torch.zeros_like(radial)
    radial_fallback[..., 0] = 1.0
    radial = F.normalize(
        torch.where(horizontal[..., None].gt(1e-6), radial, radial_fallback),
        dim=-1,
        eps=1e-8,
    )
    tangential = torch.stack(
        (-y / safe_horizontal, x / safe_horizontal, torch.zeros_like(z)), dim=-1
    )
    tangential_fallback = torch.zeros_like(tangential)
    tangential_fallback[..., 1] = 1.0
    tangential = F.normalize(
        torch.where(
            horizontal[..., None].gt(1e-6), tangential, tangential_fallback
        ),
        dim=-1,
        eps=1e-8,
    )
    cosine = math.cos(float(epsilon))
    sine = math.sin(float(epsilon))

    def scale(direction: Tensor) -> Tensor:
        positive = F.normalize(ray * cosine + direction * sine, dim=-1, eps=1e-8)
        negative = F.normalize(ray * cosine - direction * sine, dim=-1, eps=1e-8)
        positive_pixel = _network_pixels(positive, metadata)[0]
        negative_pixel = _network_pixels(negative, metadata)[0]
        return torch.linalg.vector_norm(
            positive_pixel - negative_pixel, dim=-1
        ) / (2.0 * float(epsilon))

    return tuple(
        torch.nan_to_num(value, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-6)
        for value in (scale(radial), scale(tangential))
    )
