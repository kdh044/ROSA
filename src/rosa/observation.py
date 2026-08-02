"""Projection descriptors and multiscale feature sampling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils import _python_dispatch

from . import geometry as _geometry
from .geometry import (
    _ocam_project_points_with_raw_camera_xyz,
    ocam_full_fov_boundary,
    ocam_project_points,
    ocam_projection_scales,
)
from .sampling import FusedBilinearSampler
from .types import RayFeatureRelation


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


_RELEASED_OBSERVATION_CAMERA_XYZ = _camera_xyz
_RELEASED_OCAM_PROJECT_POINTS = ocam_project_points
_RELEASED_OCAM_PROJECT_WITH_XYZ = _ocam_project_points_with_raw_camera_xyz
_RELEASED_GEOMETRY_CAMERA_XYZ = _geometry._camera_xyz
_RELEASED_OCAM_PROJECT_IMPL = _geometry._ocam_project_points_impl
_RELEASED_FUSED_SAMPLER_FORWARD = FusedBilinearSampler.forward
_RELEASED_PREPARE_GRID = FusedBilinearSampler._prepare_grid
_RELEASED_PREPARED_MULTISCALE = FusedBilinearSampler._sample_multiscale_prepared
_RELEASED_PREPARED_STACKED = FusedBilinearSampler._sample_multiscale_prepared_stacked
# This isolated candidate branch enables the eager optimization for sealed
# CUDA A/B validation.  Promotion to the release branch still requires raw-bit
# full-model parity plus the preregistered latency and memory gates.
_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE = True


def _override_mode_is_active_or_unknown() -> bool:
    """Fail closed when PyTorch override-mode state cannot be trusted."""

    torch_function_mode_state = getattr(
        torch.overrides,
        "_is_torch_function_mode_enabled",
        None,
    )
    if not callable(torch_function_mode_state):
        return True
    try:
        torch_function_mode_active = torch_function_mode_state()
    except Exception:
        return True
    if type(torch_function_mode_active) is not bool:
        return True
    dispatch_mode_state = getattr(
        _python_dispatch,
        "_is_in_non_infra_torch_dispatch_mode",
        None,
    )
    if type(dispatch_mode_state) is bool:
        dispatch_mode_active = dispatch_mode_state
    elif callable(dispatch_mode_state):
        try:
            dispatch_mode_active = dispatch_mode_state()
        except Exception:
            return True
        if type(dispatch_mode_active) is not bool:
            return True
    else:
        return True
    return torch_function_mode_active or dispatch_mode_active


def _can_reuse_camera_transform(keypoints: Tensor, metadata: dict) -> bool:
    if (
        not _ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE
        or torch.is_grad_enabled()
        or _override_mode_is_active_or_unknown()
        or type(metadata) is not dict
        or type(keypoints) is not torch.Tensor
        or keypoints.ndim != 4
        or keypoints.shape[-1] != 3
        or _camera_xyz is not _RELEASED_OBSERVATION_CAMERA_XYZ
        or ocam_project_points is not _RELEASED_OCAM_PROJECT_POINTS
        or _ocam_project_points_with_raw_camera_xyz
        is not _RELEASED_OCAM_PROJECT_WITH_XYZ
        or _geometry._camera_xyz is not _RELEASED_GEOMETRY_CAMERA_XYZ
        or _geometry._ocam_project_points_impl is not _RELEASED_OCAM_PROJECT_IMPL
        or torch.jit.is_scripting()
        or torch.jit.is_tracing()
    ):
        return False
    tensors = (
        keypoints,
        metadata.get("lidar2cam"),
        metadata.get("image_wh"),
        metadata.get("ocam_intrinsic"),
        metadata.get("ocam_pol"),
    )
    if (
        not all(type(value) is torch.Tensor for value in tensors)
        or not all(value.layout == torch.strided for value in tensors)
        or not all(not value.is_quantized for value in tensors)
    ):
        return False
    if torch._dynamo.is_compiling():
        # Dynamo specializes exact Tensor inputs by layout and dispatch key.
        # Named tensors are not accepted by torch.compile; keep their eager
        # fallback below without attempting to inspect ``Tensor.names`` while
        # a full graph is being captured.
        return True
    return bool(all(all(name is None for name in value.names) for value in tensors))


def project_and_describe(
    keypoints: Tensor,
    anchor: Tensor,
    metadata: dict,
    projection_contract: str,
    *,
    anchor_z_is_bottom: bool = False,
    compact_relation: bool = False,
) -> tuple[Tensor, Tensor, Tensor | RayFeatureRelation, Tensor]:
    """Project keypoints and encode calibrated observation geometry."""

    if not isinstance(compact_relation, bool):
        raise TypeError("compact_relation must be a bool")

    required = (
        "lidar2cam",
        "image_wh",
        "ocam_intrinsic",
        "ocam_pol",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise KeyError(f"OCam camera metadata missing: {missing}")

    reuse_projection_storage = bool(
        not torch.is_grad_enabled() and not torch._dynamo.is_compiling()
    )
    if _can_reuse_camera_transform(keypoints, metadata):
        projected, camera_xyz = _ocam_project_points_with_raw_camera_xyz(
            keypoints,
            metadata,
            metadata["image_wh"],
            contract=projection_contract,
            inference_storage_reuse=reuse_projection_storage,
        )
    else:
        projected = ocam_project_points(
            keypoints,
            metadata,
            metadata["image_wh"],
            contract=projection_contract,
            inference_storage_reuse=reuse_projection_storage,
        )
        camera_xyz = None
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
    if camera_xyz is None:
        camera_xyz = _camera_xyz(keypoints, metadata["lidar2cam"])
    ray = F.normalize(
        torch.nan_to_num(camera_xyz),
        dim=-1,
        eps=1e-6,
    )
    radial_scale, tangential_scale = ocam_projection_scales(
        ray,
        metadata,
        contract=projection_contract,
    )
    boundary = (
        torch.minimum(
            torch.minimum(safe_uv[..., 0], 1.0 - safe_uv[..., 0]),
            torch.minimum(safe_uv[..., 1], 1.0 - safe_uv[..., 1]),
        )
        .mul(2.0)
        .clamp(0.0, 1.0)
    )
    if projection_contract == "full_fov":
        boundary = torch.minimum(
            boundary,
            ocam_full_fov_boundary(
                ray,
                radial_scale,
                metadata,
                metadata["image_wh"],
            ),
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

    size = anchor[..., 3:6].exp().clamp_min(1e-3)
    center = anchor[..., :3]
    if anchor_z_is_bottom:
        center = center.clone()
        center[..., 2] = center[..., 2] + size[..., 2] * 0.5
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

    if compact_relation:
        # Preserve the historical carrier's independent ownership so an RFL
        # pre-hook cannot mutate the geometry/validity returned to ROSA.
        relation = RayFeatureRelation(
            ray=geometry[..., :3].clone(),
            valid=valid.clone(),
        )
    else:
        relation = torch.zeros(
            geometry.shape[:-1] + (19,),
            device=geometry.device,
            dtype=geometry.dtype,
        )
        relation[..., 8:11] = geometry[..., :3]
        relation[..., 13] = valid.to(geometry.dtype)
        relation = relation.permute(0, 2, 1, 3, 4).contiguous()
    return uv, geometry, relation, valid


def _can_use_prepared_multiscale(
    fpn: Sequence[Tensor],
    uv: Tensor,
    sampler: nn.Module,
    *,
    private_eager_inference: bool = False,
) -> bool:
    """Select one-grid sampling only on an unobserved inference surface."""

    if not isinstance(private_eager_inference, bool):
        raise TypeError("private_eager_inference must be a bool")

    compiling = torch._dynamo.is_compiling()
    if not compiling:
        if (
            not private_eager_inference
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
            or _override_mode_is_active_or_unknown()
        ):
            return False

    if (
        type(fpn) not in (list, tuple)
        or len(fpn) != 4
        or type(sampler) is not FusedBilinearSampler
        or sampler.training
        or torch.is_grad_enabled()
        or getattr(sampler, "_compiled_call_impl", None) is not None
        or "forward" in sampler.__dict__
        or type(sampler).forward is not _RELEASED_FUSED_SAMPLER_FORWARD
        or "_prepare_grid" in sampler.__dict__
        or type(sampler)._prepare_grid is not _RELEASED_PREPARE_GRID
        or "_sample_multiscale_prepared" in sampler.__dict__
        or type(sampler)._sample_multiscale_prepared
        is not _RELEASED_PREPARED_MULTISCALE
        or type(uv) is not torch.Tensor
        or uv.ndim != 5
        or uv.shape[-1] != 2
        or uv.requires_grad
        or any(type(feature) is not torch.Tensor for feature in fpn)
        or any(feature.ndim != 5 or feature.requires_grad for feature in fpn)
    ):
        return False
    if (
        sampler._forward_pre_hooks
        or sampler._forward_hooks
        or torch.nn.modules.module._global_forward_pre_hooks
        or torch.nn.modules.module._global_forward_hooks
    ):
        return False
    if (
        private_eager_inference
        and not compiling
        and (
            "_sample_multiscale_prepared_stacked" in sampler.__dict__
            or type(sampler)._sample_multiscale_prepared_stacked
            is not _RELEASED_PREPARED_STACKED
        )
    ):
        return False

    batch, cameras, channels = fpn[0].shape[:3]
    tensors = (*fpn, uv)
    if (
        uv.shape[0] != batch
        or uv.shape[2] != cameras
        or any(feature.shape[:3] != (batch, cameras, channels) for feature in fpn)
        or any(
            feature.dtype != uv.dtype or feature.device != uv.device for feature in fpn
        )
        or any(
            tensor.layout != torch.strided or tensor.is_quantized for tensor in tensors
        )
    ):
        return False
    return True


def _sample_multiscale_features_impl(
    fpn: Sequence[Tensor],
    uv: Tensor,
    query: Tensor,
    sampler: nn.Module,
    level_head: nn.Linear,
    *,
    private_prepared_inference: bool = False,
) -> Tensor:
    """Internal implementation with an explicit private eager opt-in."""

    if len(fpn) != level_head.out_features:
        raise ValueError(
            f"expected {level_head.out_features} FPN levels, " f"got {len(fpn)}"
        )
    sample_uv = uv.unsqueeze(4)
    prepared = _can_use_prepared_multiscale(
        fpn,
        uv,
        sampler,
        private_eager_inference=private_prepared_inference,
    )
    if prepared and private_prepared_inference and not torch._dynamo.is_compiling():
        # The final contiguous level tensor is private to this hook-free eager
        # path.  Fill it level by level so four grid-sample outputs are not all
        # retained until a later stack allocation.
        levels = sampler._sample_multiscale_prepared_stacked(fpn, sample_uv)
    else:
        if prepared:
            sampled = [
                value.squeeze(4)
                for value in sampler._sample_multiscale_prepared(fpn, sample_uv)
            ]
        else:
            sampled = [sampler(feature, sample_uv).squeeze(4) for feature in fpn]
        levels = torch.stack(sampled, dim=2)
        # ``stack`` owns the fused level storage.  Drop the four per-level
        # outputs before allocating any subsequent tensor.
        del sampled
    weights = torch.softmax(
        level_head(query).float(),
        dim=-1,
    ).to(levels.dtype)
    expanded_weights = weights[:, :, :, None, None, None]
    if (
        not torch.is_grad_enabled()
        and type(levels) is torch.Tensor
        and levels.layout == torch.strided
        and levels._base is None
        and not levels.is_quantized
        and (torch._dynamo.is_compiling() or all(name is None for name in levels.names))
    ):
        # ``levels`` is the private result of ``stack``.  Evaluation can reuse
        # that storage for the weighted reduction instead of materializing a
        # second tensor of the same [B,N,L,C,P,D] size.
        levels.mul_(expanded_weights)
        return levels.sum(dim=2)
    return (levels * expanded_weights).sum(dim=2)


def sample_multiscale_features(
    fpn: Sequence[Tensor],
    uv: Tensor,
    query: Tensor,
    sampler: nn.Module,
    level_head: nn.Linear,
) -> Tensor:
    """Sample each FPN level and fuse it with query-specific weights."""

    return _sample_multiscale_features_impl(
        fpn,
        uv,
        query,
        sampler,
        level_head,
    )


def _sample_multiscale_features_prepared_inference(
    fpn: Sequence[Tensor],
    uv: Tensor,
    query: Tensor,
    sampler: nn.Module,
    level_head: nn.Linear,
) -> Tensor:
    """Use one prepared grid on the refiner's private eager inference edge."""

    return _sample_multiscale_features_impl(
        fpn,
        uv,
        query,
        sampler,
        level_head,
        private_prepared_inference=True,
    )


__all__ = ["project_and_describe", "sample_multiscale_features"]
