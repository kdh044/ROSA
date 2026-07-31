"""Projection descriptors and multiscale feature sampling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import ocam_project_points, ocam_projection_scales
from .sampling import DeterministicBilinearSampler


def _camera_xyz(points: Tensor, lidar2cam: Tensor) -> Tensor:
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must be [B,N,P,3]")
    homogeneous = torch.cat(
        (points, torch.ones_like(points[..., :1])),
        dim=-1,
    )
    return torch.matmul(
        lidar2cam[:, :, None, None],
        homogeneous[:, None, ..., None],
    ).squeeze(-1)[..., :3]


def project_and_describe(
    keypoints: Tensor,
    anchor: Tensor,
    metadata: dict,
    projection_contract: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Project keypoints and encode calibrated observation geometry."""

    required = (
        "lidar2cam",
        "image_wh",
        "ocam_intrinsic",
        "ocam_pol",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise KeyError(f"OCam camera metadata missing: {missing}")

    projected = ocam_project_points(
        keypoints,
        metadata,
        metadata["image_wh"],
        contract=projection_contract,
    )
    valid = (
        torch.isfinite(projected).all(dim=-1)
        & projected[..., 0].ge(0)
        & projected[..., 0].le(1)
        & projected[..., 1].ge(0)
        & projected[..., 1].le(1)
    )
    safe_uv = torch.where(
        valid[..., None],
        torch.nan_to_num(projected),
        torch.zeros_like(projected),
    )
    camera_xyz = _camera_xyz(keypoints, metadata["lidar2cam"])
    ray = F.normalize(
        torch.nan_to_num(camera_xyz),
        dim=-1,
        eps=1e-6,
    )
    radial_scale, tangential_scale = ocam_projection_scales(
        ray,
        metadata,
    )
    boundary = (
        torch.minimum(
            torch.minimum(safe_uv[..., 0], 1.0 - safe_uv[..., 0]),
            torch.minimum(safe_uv[..., 1], 1.0 - safe_uv[..., 1]),
        )
        .mul(2.0)
        .clamp(0.0, 1.0)
    )

    positive_inf = torch.full_like(safe_uv, float("inf"))
    negative_inf = torch.full_like(safe_uv, float("-inf"))
    minimum_uv = torch.where(
        valid[..., None],
        safe_uv,
        positive_inf,
    ).amin(dim=3)
    maximum_uv = torch.where(
        valid[..., None],
        safe_uv,
        negative_inf,
    ).amax(dim=3)
    footprint = (maximum_uv - minimum_uv).clamp_min(0.0) * metadata["image_wh"][
        :, :, None, :
    ]
    short_axis = footprint.amin(dim=-1)
    short_axis = torch.where(
        valid.sum(dim=3) >= 2,
        short_axis,
        torch.zeros_like(short_axis),
    )
    short_axis = torch.log1p(short_axis).clamp(max=6.0) / 6.0
    short_axis = short_axis[..., None].expand_as(boundary)

    center = anchor[..., :3]
    size = anchor[..., 3:6].exp().clamp_min(1e-3)
    yaw_sin = anchor[..., 6]
    yaw_cos = anchor[..., 7]
    local = keypoints - center[..., None, :]
    local_x = yaw_cos[..., None] * local[..., 0] + yaw_sin[..., None] * local[..., 1]
    local_y = -yaw_sin[..., None] * local[..., 0] + yaw_cos[..., None] * local[..., 1]
    local = torch.stack((local_x, local_y, local[..., 2]), dim=-1)
    local = local / size[..., None, :]
    local = local[:, None].expand(
        -1,
        ray.shape[1],
        -1,
        -1,
        -1,
    )

    log_radial = radial_scale.log().clamp(-8.0, 8.0) / 8.0
    log_tangential = tangential_scale.log().clamp(-8.0, 8.0) / 8.0
    anisotropy = (radial_scale / tangential_scale).log().clamp(-4.0, 4.0) / 4.0
    geometry = torch.cat(
        (
            ray,
            (safe_uv - 0.5) * 2.0,
            boundary[..., None],
            log_radial[..., None],
            log_tangential[..., None],
            anisotropy[..., None],
            short_axis[..., None],
            valid[..., None].to(ray.dtype),
            local,
        ),
        dim=-1,
    )
    geometry = geometry.permute(0, 2, 1, 3, 4).contiguous()
    valid = valid.permute(0, 2, 1, 3).contiguous()
    uv = safe_uv.permute(0, 2, 1, 3, 4).contiguous()

    relation = torch.zeros(
        geometry.shape[:-1] + (19,),
        device=geometry.device,
        dtype=geometry.dtype,
    )
    relation[..., 8:11] = geometry[..., :3]
    relation[..., 13] = valid.to(geometry.dtype)
    relation = relation.permute(0, 2, 1, 3, 4).contiguous()
    return uv, geometry, relation, valid


def sample_multiscale_features(
    fpn: Sequence[Tensor],
    uv: Tensor,
    query: Tensor,
    sampler: DeterministicBilinearSampler,
    level_head: nn.Linear,
) -> Tensor:
    """Sample each FPN level and fuse it with query-specific weights."""

    if len(fpn) != level_head.out_features:
        raise ValueError(
            f"expected {level_head.out_features} FPN levels, " f"got {len(fpn)}"
        )
    sample_uv = uv.unsqueeze(4)
    sampled = [sampler(feature, sample_uv).squeeze(4) for feature in fpn]
    levels = torch.stack(sampled, dim=2)
    weights = torch.softmax(
        level_head(query).float(),
        dim=-1,
    ).to(levels.dtype)
    return (levels * weights[:, :, :, None, None, None]).sum(dim=2)


__all__ = ["project_and_describe", "sample_multiscale_features"]
