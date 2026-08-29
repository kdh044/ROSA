"""Camera-model-agnostic projection for sparse native-fisheye ray queries.

``NativeFisheyeProjector`` owns rig transforms and spherical relations while a
small backend owns only the lens map.  OCam is the first backend because it is
the calibration shipped by Fisheye3DOD; the decoder never branches on OCam and
can accept another calibrated fisheye model implementing the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .fisheye_evidence import FisheyeObservationGeometry
from .ray_state import normalize_direction, spherical_tangent_basis
from .sampling import DeterministicBilinearSampler, FusedBilinearSampler


@dataclass(frozen=True)
class LensProjection:
    """Output contract of a native fisheye lens backend."""

    uv: Tensor
    jacobian: Tensor
    lens_margin: Tensor
    valid: Tensor


@dataclass(frozen=True)
class NativeFisheyeProjection:
    """Rig-aware projection consumed by the unified evidence attention."""

    uv: Tensor
    geometry: FisheyeObservationGeometry


class FisheyeProjectionModel(nn.Module, ABC):
    """Interface implemented by OCam, KB, Mei, or Double-Sphere backends."""

    @abstractmethod
    def project_rays(
        self,
        camera_rays: Tensor,
        metadata: dict,
        image_wh: Tensor,
    ) -> LensProjection:
        """Project camera-frame unit rays ``[B,Q,C,3]`` to native pixels."""


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

    if coefficients.ndim != 3:
        raise ValueError("OCam polynomial must be [B,C,K]")
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


class OCamFisheyeProjection(FisheyeProjectionModel):
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
        missing = [name for name in self.required_metadata if name not in metadata]
        if missing:
            raise KeyError(f"missing OCam metadata: {missing}")

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
    # The logarithm is direction-ambiguous at exact alignment/antipodes; the
    # previous forward contract returned zero there, so retain that value and
    # deliberately stop its undefined gradient.  Ordinary rays keep the exact
    # geodesic angle and differentiable tangent direction.
    angle = torch.atan2(sine, cosine)
    regular = sine > 1e-7
    coordinates = tangent_components * (angle / sine.clamp_min(1e-7))
    return torch.where(regular, coordinates, torch.zeros_like(coordinates))


class NativeFisheyeProjector(nn.Module):
    """Transform ego rays/points and apply an interchangeable lens backend."""

    def __init__(self, lens: FisheyeProjectionModel | None = None) -> None:
        super().__init__()
        self.lens = lens or OCamFisheyeProjection()

    def forward(
        self,
        direction: Tensor,
        metadata: dict,
        *,
        points: Tensor | None = None,
    ) -> NativeFisheyeProjection:
        if direction.ndim != 3 or direction.shape[-1] != 3:
            raise ValueError("direction must be [B,Q,3]")
        if "lidar2cam" not in metadata or "image_wh" not in metadata:
            raise KeyError("fisheye projection requires lidar2cam and image_wh")
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
        camera_forward = direction.new_tensor((0.0, 0.0, -1.0))
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
        geometry = FisheyeObservationGeometry(
            rays=camera_rays,
            tangent=tangent,
            jacobian=lens.jacobian,
            lens_margin=lens.lens_margin,
            overlap=overlap,
            valid=lens.valid,
        )
        return NativeFisheyeProjection(uv=lens.uv, geometry=geometry)


def sample_native_fisheye_fpn(
    fpn: Sequence[Tensor],
    uv: Tensor,
    *,
    sampler: nn.Module | None = None,
) -> Tensor:
    """Sample one ray coordinate per camera and FPN level exactly once."""

    if not fpn:
        raise ValueError("fpn must contain at least one feature level")
    if uv.ndim != 4 or uv.shape[-1] != 2:
        raise ValueError("uv must be [B,Q,C,2]")
    batch, queries, cameras = uv.shape[:3]
    sampler = sampler or DeterministicBilinearSampler()
    sample_uv = uv[:, :, :, None, None, :]
    if (
        isinstance(sampler, FusedBilinearSampler)
        and not uv.requires_grad
        and all(not feature.requires_grad for feature in fpn)
    ):
        # One optimized grid-sampling kernel per level, with a single prepared
        # grid shared by all levels.  Training deliberately falls through to
        # the deterministic gather implementation: CUDA grid-sample backward
        # is nondeterministic and the C0 comparison seals deterministic mode.
        sampled = sampler._sample_multiscale_prepared_stacked(fpn, sample_uv)
        return sampled.squeeze(4).permute(0, 1, 3, 2, 4).contiguous()
    if type(sampler) is FusedBilinearSampler:
        # The training path keeps deterministic gather backward, but prepares
        # the shared query/camera grid only once for the complete FPN.  Exact
        # type admission preserves custom sampler instrumentation semantics.
        sampled_levels = sampler.fallback.sample_multiscale(fpn, sample_uv)
        return torch.stack(
            [sampled[:, :, :, 0, 0, :] for sampled in sampled_levels],
            dim=3,
        )
    levels = []
    channels = None
    for feature in fpn:
        if feature.ndim != 5 or feature.shape[:2] != (batch, cameras):
            raise ValueError("each FPN level must be [B,C,D,H,W]")
        if channels is None:
            channels = feature.shape[2]
        elif feature.shape[2] != channels:
            raise ValueError("all FPN levels must share their channel width")
        sampled = sampler(feature, sample_uv)
        levels.append(sampled[:, :, :, 0, 0, :])
    return torch.stack(levels, dim=3)


__all__ = [
    "FisheyeProjectionModel",
    "LensProjection",
    "NativeFisheyeProjection",
    "NativeFisheyeProjector",
    "OCamFisheyeProjection",
    "sample_native_fisheye_fpn",
]
