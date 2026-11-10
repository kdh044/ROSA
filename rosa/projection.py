"""Projection of box keypoints into the native camera images.

``Projector`` applies the rig transform and computes the
projection cues; a lens backend maps camera rays to pixels: OCam for
Fisheye3DOD, pinhole + MEI for KITTI-360.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pinhole_mei import pinhole_mei_project_rays
from .ray import normalize_direction, spherical_tangent_basis


@dataclass(frozen=True)
class ObservationGeometry:
    """Calibrated per-query geometry, ``[B,Q,C,...]``.

    ``rays`` are camera-frame unit rays, ``tangent`` the observation in the
    query's local tangent frame, and ``jacobian`` maps a local angular
    displacement (radians) to native-image pixels.
    """

    rays: Tensor
    tangent: Tensor
    jacobian: Tensor
    lens_margin: Tensor
    overlap: Tensor
    valid: Tensor


@dataclass(frozen=True)
class LensProjection:
    """Output of a lens backend."""

    uv: Tensor
    jacobian: Tensor
    lens_margin: Tensor
    valid: Tensor


@dataclass(frozen=True)
class Projection:
    """Normalized pixels plus the projection cues of every camera."""

    uv: Tensor
    geometry: ObservationGeometry


def _expand_fov(raw_fov, reference: Tensor) -> Tensor:
    batch, _, cameras = reference.shape[:3]
    fov = torch.as_tensor(raw_fov, device=reference.device, dtype=reference.dtype)
    if fov.numel() == 1:
        fov = fov.reshape(1, 1).expand(batch, cameras)
    elif fov.ndim == 1 and fov.shape[0] == batch:
        fov = fov[:, None].expand(batch, cameras)
    elif tuple(fov.shape) != (batch, cameras):
        raise ValueError("lens_fov_deg must be scalar, [B], or [B,C]")
    return torch.deg2rad(fov * 0.5)[:, None, :]


def _polynomial_and_derivative(
    coefficients: Tensor, value: Tensor
) -> tuple[Tensor, Tensor]:
    """Evaluate an ascending-power polynomial and its derivative."""

    expanded = coefficients[:, None]
    terms = expanded.unbind(dim=-1)
    output = terms[-1].expand_as(value)
    for coefficient in reversed(terms[:-1]):
        output = output * value + coefficient
    if len(terms) == 1:
        return output, torch.zeros_like(output)
    derivative = (len(terms) - 1) * terms[-1].expand_as(value)
    for power in range(len(terms) - 2, 0, -1):
        derivative = derivative * value + power * terms[power]
    return output, derivative


class OCamProjection(nn.Module):
    """Full-FOV OCam backend with an analytic pixel/radian Jacobian."""

    required_metadata = (
        "ocam_intrinsic",
        "ocam_pol",
    )

    def project_rays(
        self,
        camera_rays: Tensor,
        metadata: dict,
        image_wh: Tensor,
    ) -> LensProjection:
        if camera_rays.ndim != 4 or camera_rays.shape[-1] != 3:
            raise ValueError("camera_rays must be [B,Q,C,3]")
        batch, _, cameras = camera_rays.shape[:3]
        if tuple(image_wh.shape) != (batch, cameras, 2):
            raise ValueError("image_wh must be [B,C,2]")

        ray = normalize_direction(camera_rays)
        dtype = ray.dtype
        device = ray.device
        intrinsic = metadata["ocam_intrinsic"].to(device=device, dtype=dtype)
        polynomial = metadata["ocam_pol"].to(device=device, dtype=dtype)
        if tuple(intrinsic.shape) != (batch, cameras, 3, 3):
            raise ValueError("ocam_intrinsic must be [B,C,3,3]")
        if polynomial.shape[:2] != (batch, cameras):
            raise ValueError("ocam_pol must be [B,C,K]")

        x, y, z = ray.unbind(dim=-1)
        planar = torch.sqrt(x.square() + y.square() + 1e-12)
        theta = torch.atan2(z, planar)
        rho, rho_derivative = _polynomial_and_derivative(polynomial, theta)
        scale = torch.where(
            planar > 1e-8,
            rho / planar.clamp_min(1e-8),
            torch.zeros_like(rho),
        )
        image_x = x * scale
        image_y = y * scale

        a11 = intrinsic[:, None, :, 1, 0]
        a12 = intrinsic[:, None, :, 1, 1]
        a13 = intrinsic[:, None, :, 0, 2]
        a21 = intrinsic[:, None, :, 0, 0]
        a22 = intrinsic[:, None, :, 0, 1]
        a23 = intrinsic[:, None, :, 1, 2]
        pixel_u = a11 * image_x + a12 * image_y + a13
        pixel_v = a21 * image_x + a22 * image_y + a23
        width = image_wh[:, None, :, 0].to(dtype=dtype, device=device)
        height = image_wh[:, None, :, 1].to(dtype=dtype, device=device)
        uv = torch.stack((pixel_u / width, pixel_v / height), dim=-1)

        incidence = torch.atan2(planar, -z)
        half_fov = _expand_fov(
            metadata.get(
                "fisheye_lens_fov_deg", metadata.get("ocam_lens_fov_deg", 220.0)
            ),
            ray,
        )
        finite = torch.isfinite(camera_rays).all(dim=-1)
        valid = (
            finite
            & incidence.le(half_fov)
            & uv[..., 0].ge(0.0)
            & uv[..., 0].le(1.0)
            & uv[..., 1].ge(0.0)
            & uv[..., 1].le(1.0)
        )

        radial_x = torch.where(planar > 1e-8, x / planar, torch.ones_like(x))
        radial_y = torch.where(planar > 1e-8, y / planar, torch.zeros_like(y))
        tangent_x = -radial_y
        tangent_y = radial_x
        tangential_scale = torch.where(
            incidence.sin().abs() > 1e-4,
            rho / incidence.sin().abs().clamp_min(1e-4),
            rho_derivative,
        )
        radial_u = (a11 * radial_x + a12 * radial_y) * rho_derivative
        radial_v = (a21 * radial_x + a22 * radial_y) * rho_derivative
        tangent_u = (a11 * tangent_x + a12 * tangent_y) * tangential_scale
        tangent_v = (a21 * tangent_x + a22 * tangent_y) * tangential_scale
        jacobian = torch.stack(
            (
                torch.stack((radial_u, tangent_u), dim=-1),
                torch.stack((radial_v, tangent_v), dim=-1),
            ),
            dim=-2,
        )
        jacobian = torch.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0)

        border_margin = (
            2.0
            * torch.minimum(
                torch.minimum(uv[..., 0], 1.0 - uv[..., 0]),
                torch.minimum(uv[..., 1], 1.0 - uv[..., 1]),
            )
        ).clamp(0.0, 1.0)
        angular_margin = ((half_fov - incidence) / half_fov.clamp_min(1e-6)).clamp(
            0.0, 1.0
        )
        lens_margin = torch.minimum(border_margin, angular_margin)[..., None]
        safe_uv = torch.where(valid[..., None], uv, torch.zeros_like(uv))
        safe_jacobian = torch.where(
            valid[..., None, None], jacobian, torch.zeros_like(jacobian)
        )
        lens_margin = torch.where(
            valid[..., None], lens_margin, torch.zeros_like(lens_margin)
        )
        return LensProjection(
            uv=safe_uv,
            jacobian=safe_jacobian,
            lens_margin=lens_margin,
            valid=valid,
        )


class PinholeMEIProjection(nn.Module):
    """KITTI-360 backend: pinhole cameras and unified (MEI) fisheye cameras."""

    required_metadata = (
        "camera_model_id",
        "camera_intrinsic",
        "mei_distortion",
        "image_affine",
    )

    def __init__(self, jacobian_epsilon: float = 1e-3) -> None:
        super().__init__()
        self.jacobian_epsilon = float(jacobian_epsilon)

    def project_rays(
        self,
        camera_rays: Tensor,
        metadata: dict,
        image_wh: Tensor,
    ) -> LensProjection:
        if camera_rays.ndim != 4 or camera_rays.shape[-1] != 3:
            raise ValueError("camera_rays must be [B,Q,C,3]")
        batch, _, cameras = camera_rays.shape[:3]
        if tuple(image_wh.shape) != (batch, cameras, 2):
            raise ValueError("image_wh must be [B,C,2]")

        ray = normalize_direction(camera_rays)
        pixels, domain_valid = pinhole_mei_project_rays(ray, metadata)
        width = image_wh[:, None, :, 0].to(device=ray.device, dtype=ray.dtype)
        height = image_wh[:, None, :, 1].to(device=ray.device, dtype=ray.dtype)
        uv = torch.stack((pixels[..., 0] / width, pixels[..., 1] / height), dim=-1)
        valid = (
            domain_valid
            & torch.isfinite(camera_rays).all(dim=-1)
            & torch.isfinite(uv).all(dim=-1)
            & uv[..., 0].ge(0.0)
            & uv[..., 0].le(1.0)
            & uv[..., 1].ge(0.0)
            & uv[..., 1].le(1.0)
        )

        e1, e2 = spherical_tangent_basis(ray)
        epsilon = self.jacobian_epsilon
        cosine = math.cos(epsilon)
        sine = math.sin(epsilon)

        def derivative(tangent: Tensor) -> Tensor:
            positive = normalize_direction(ray * cosine + tangent * sine)
            negative = normalize_direction(ray * cosine - tangent * sine)
            positive_pixel = pinhole_mei_project_rays(
                positive, metadata
            )[0]
            negative_pixel = pinhole_mei_project_rays(
                negative, metadata
            )[0]
            return (positive_pixel - negative_pixel) / (2.0 * epsilon)

        jacobian = torch.stack((derivative(e1), derivative(e2)), dim=-1)
        jacobian = torch.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0)
        border_margin = (
            2.0
            * torch.minimum(
                torch.minimum(uv[..., 0], 1.0 - uv[..., 0]),
                torch.minimum(uv[..., 1], 1.0 - uv[..., 1]),
            )
        ).clamp(0.0, 1.0)[..., None]
        return LensProjection(
            uv=torch.where(valid[..., None], uv, torch.zeros_like(uv)),
            jacobian=torch.where(
                valid[..., None, None], jacobian, torch.zeros_like(jacobian)
            ),
            lens_margin=torch.where(
                valid[..., None], border_margin, torch.zeros_like(border_margin)
            ),
            valid=valid,
        )


def _spherical_log_coordinates(origin: Tensor, target: Tensor) -> Tensor:
    """Express the geodesic from ``origin`` to ``target`` in its tangent frame."""

    origin = normalize_direction(origin)
    target = normalize_direction(target)
    if origin.shape != target.shape:
        raise ValueError("origin and target ray shapes must match")
    e1, e2 = spherical_tangent_basis(origin)
    cosine = (origin * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent_components = torch.cat(
        (
            (target * e1).sum(dim=-1, keepdim=True),
            (target * e2).sum(dim=-1, keepdim=True),
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(tangent_components, dim=-1, keepdim=True)
    # atan2 avoids the infinite derivative of acos(cosine) at |cosine|=1.
    # The logarithm is undefined at exact alignment/antipodes; return zero there.
    angle = torch.atan2(sine, cosine)
    regular = sine > 1e-7
    coordinates = tangent_components * (angle / sine.clamp_min(1e-7))
    return torch.where(regular, coordinates, torch.zeros_like(coordinates))


class Projector(nn.Module):
    """Transform ego rays/points and apply an interchangeable lens backend."""

    def __init__(
        self,
        lens: nn.Module | None = None,
        *,
        camera_forward: tuple[float, float, float] = (0.0, 0.0, -1.0),
    ) -> None:
        super().__init__()
        forward = torch.tensor(camera_forward, dtype=torch.float32)
        if forward.shape != (3,) or not torch.isfinite(forward).all():
            raise ValueError("camera_forward must be a finite 3-vector")
        if torch.linalg.vector_norm(forward).le(1e-6):
            raise ValueError("camera_forward must be non-zero")
        self.register_buffer("camera_forward", F.normalize(forward, dim=0), persistent=False)
        self.lens = lens or OCamProjection()

    def forward(
        self,
        direction: Tensor,
        metadata: dict,
        *,
        points: Tensor | None = None,
    ) -> Projection:
        if direction.ndim != 3 or direction.shape[-1] != 3:
            raise ValueError("direction must be [B,Q,3]")
        transform = metadata["lidar2cam"].to(
            device=direction.device, dtype=direction.dtype
        )
        image_wh = metadata["image_wh"].to(
            device=direction.device, dtype=direction.dtype
        )
        batch, queries = direction.shape[:2]
        if (
            transform.ndim != 4
            or transform.shape[0] != batch
            or transform.shape[-2:] != (4, 4)
        ):
            raise ValueError("lidar2cam must be [B,C,4,4]")
        cameras = transform.shape[1]
        if points is not None and tuple(points.shape) != (batch, queries, 3):
            raise ValueError("points must be [B,Q,3]")

        rotation = transform[..., :3, :3]
        source = normalize_direction(direction) if points is None else points
        camera_xyz = torch.einsum("bcij,bqj->bqci", rotation, source)
        if points is not None:
            camera_xyz = camera_xyz + transform[:, None, :, :3, 3]
        camera_geometry_valid = torch.isfinite(camera_xyz).all(
            dim=-1
        ) & torch.linalg.vector_norm(camera_xyz, dim=-1).gt(1e-6)
        camera_rays = normalize_direction(camera_xyz)
        lens = self.lens.project_rays(camera_rays, metadata, image_wh)
        valid = lens.valid & camera_geometry_valid
        lens = LensProjection(
            uv=torch.where(valid[..., None], lens.uv, torch.zeros_like(lens.uv)),
            jacobian=torch.where(
                valid[..., None, None],
                lens.jacobian,
                torch.zeros_like(lens.jacobian),
            ),
            lens_margin=torch.where(
                valid[..., None],
                lens.lens_margin,
                torch.zeros_like(lens.lens_margin),
            ),
            valid=valid,
        )

        query_direction = normalize_direction(direction)
        camera_forward = self.camera_forward.to(device=direction.device, dtype=direction.dtype)
        optical_axis_ego = torch.einsum("bcji,j->bci", rotation, camera_forward)
        tangent = _spherical_log_coordinates(
            query_direction[:, :, None, :].expand(-1, -1, cameras, -1),
            optical_axis_ego[:, None, :, :].expand(-1, queries, -1, -1),
        )
        valid_count = lens.valid.sum(dim=2, keepdim=True)
        overlap = (valid_count - 1).clamp_min(0).to(direction.dtype) / max(
            cameras - 1, 1
        )
        overlap = overlap.expand(-1, -1, cameras)[..., None]
        overlap = torch.where(lens.valid[..., None], overlap, torch.zeros_like(overlap))
        geometry = ObservationGeometry(
            rays=camera_rays,
            tangent=tangent,
            jacobian=lens.jacobian,
            lens_margin=lens.lens_margin,
            overlap=overlap,
            valid=lens.valid,
        )
        return Projection(uv=lens.uv, geometry=geometry)


def build_projector(projection_type: str) -> Projector:
    """Projector for the dataset's camera model."""

    if projection_type == "ocam":
        return Projector()
    if projection_type == "pinhole_mei":
        return Projector(PinholeMEIProjection(), camera_forward=(0.0, 0.0, 1.0))
    raise ValueError(f"unsupported projection_type={projection_type!r}")


__all__ = [
    "PinholeMEIProjection",
    "build_projector",
    "LensProjection",
    "Projection",
    "Projector",
    "OCamProjection",
]
