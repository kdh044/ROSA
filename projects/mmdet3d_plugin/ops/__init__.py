"""Python entry points of the deformable-aggregation CUDA extension.

Build it once with ``python setup.py build_ext --inplace`` in this directory.
The ``rosa_*`` and ``*_rfl_rosa_*`` kernels are inference-only fused paths.
"""
import torch

try:
    from . import deformable_aggregation_ext
    from .deformable_aggregation import DeformableAggregationPerCameraFunction
except ImportError as exc:
    raise ImportError(
        "deformable_aggregation_ext is not built; run "
        "`python setup.py build_ext --inplace` in projects/mmdet3d_plugin/ops"
    ) from exc


def deformable_aggregation_per_camera_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    """Return ``(fused [B,N,D], views [B,N,C,D])`` in one CUDA call."""

    return DeformableAggregationPerCameraFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def deformable_aggregation_rfl_rosa_inference(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
    camera_logits,
    camera_rays,
    camera_valid,
    query_heads,
    appearance_weight,
    ray_weight,
    output_weight,
    key_rfl_weight,
    reliability_rfl_weight,
    points_per_query,
):
    """Fused camera aggregation path (inference only)."""

    if torch.is_grad_enabled():
        raise RuntimeError("streaming RFL/ROSA CUDA path is inference-only")
    return deformable_aggregation_ext.deformable_aggregation_rfl_rosa_forward(
        feature_maps.contiguous().float(),
        spatial_shape.contiguous().int(),
        scale_start_index.contiguous().int(),
        sampling_location.contiguous().float(),
        weights.contiguous().float(),
        camera_logits.contiguous().float(),
        camera_rays.contiguous().float(),
        camera_valid.contiguous().bool(),
        query_heads.contiguous().float(),
        appearance_weight.contiguous().float(),
        ray_weight.contiguous().float(),
        output_weight.contiguous().float(),
        key_rfl_weight.contiguous().float(),
        reliability_rfl_weight.contiguous().float(),
        int(points_per_query),
    )


def rosa_hybrid_state_update_inference(
    direction,
    log_range,
    log_size,
    yaw,
    velocity,
    raw,
    update_gate_logits,
    face_yaw,
    face_yaw_gate_logits,
):
    """Fuse the post-head box state update into one CUDA launch."""

    if torch.is_grad_enabled():
        raise RuntimeError("fused hybrid-state update is inference-only")
    values = (
        direction,
        log_range,
        log_size,
        yaw,
        velocity,
        raw,
        update_gate_logits,
        face_yaw,
        face_yaw_gate_logits,
    )
    return deformable_aggregation_ext.rosa_hybrid_state_update_forward(
        *(value.contiguous().float() for value in values)
    )


def rosa_state_geometry_inputs_inference(
    direction, log_range, log_size, yaw, velocity
):
    """Build the SH state encoding and angular extent in one CUDA launch."""
    if torch.is_grad_enabled():
        raise RuntimeError("fused state geometry inputs are inference-only")
    return deformable_aggregation_ext.rosa_state_geometry_inputs_forward(
        *(value.contiguous().float() for value in (
            direction, log_range, log_size, yaw, velocity
        ))
    )


def rosa_build_keypoints_project_geometry_inference(
    direction, log_range, log_size, yaw, learned, lidar2cam, image_wh,
    intrinsic, polynomial, half_fov, log_extent, level_strides,
    points_per_query, fixed_scale,
):
    """Build all 13 box points and their calibrated geometry in CUDA."""
    if torch.is_grad_enabled():
        raise RuntimeError("fused keypoint projection is inference-only")
    values = (
        direction, log_range, log_size, yaw, learned, lidar2cam, image_wh,
        intrinsic, polynomial, half_fov, log_extent, level_strides,
    )
    return deformable_aggregation_ext.rosa_build_keypoints_project_geometry_forward(
        *(value.contiguous().float() for value in values),
        int(points_per_query), float(fixed_scale),
    )


def rosa_point_attention_inference(query, key, reliability, values, valid):
    """Fuse 13-point logits, masked softmax, and value reduction."""
    if torch.is_grad_enabled():
        raise RuntimeError("fused point attention is inference-only")
    return deformable_aggregation_ext.rosa_point_attention_forward(
        query.contiguous().float(), key.contiguous().float(),
        reliability.contiguous().float(), values.contiguous().float(),
        valid.contiguous(),
    )


def feature_maps_format(feature_maps):
    """Flatten FPN levels into Sparse4D's ``(col_feats, spatial_shape, start_index)``."""
    if isinstance(feature_maps[0], (list, tuple)):
        formated = [feature_maps_format(x) for x in feature_maps]
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []

    col_feats = []
    for i, feat in enumerate(feature_maps):
        spatial_shape.append(feat.shape[-2:])
        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )

    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)

    feature_maps = [
        col_feats,
        spatial_shape,
        scale_start_index,
    ]
    return feature_maps
