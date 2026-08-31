import torch

try:
    from .deformable_aggregation import (
        DeformableAggregationFunction,
        DeformableAggregationPerCameraDeterministicFunction,
        DeformableAggregationPerCameraFunction,
    )
    from . import deformable_aggregation_ext  # noqa: F401

    _DAF_BUILT = True
except Exception:
    DeformableAggregationFunction = None
    DeformableAggregationPerCameraDeterministicFunction = None
    DeformableAggregationPerCameraFunction = None
    _DAF_BUILT = False


def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    if not _DAF_BUILT:
        raise RuntimeError(
            "deformable_aggregation_ext is not built. Run "
            "`python projects/mmdet3d_plugin/ops/setup.py develop` in this package."
        )
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def deformable_aggregation_per_camera_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    """Return ``(fused [B,N,D], views [B,N,C,D])`` in one CUDA call."""

    if not _DAF_BUILT or DeformableAggregationPerCameraFunction is None:
        raise RuntimeError(
            "deformable_aggregation_ext is not built. Run "
            "`python projects/mmdet3d_plugin/ops/setup.py develop` in this package."
        )
    if not hasattr(
        deformable_aggregation_ext,
        "deformable_aggregation_per_camera_forward",
    ):
        raise RuntimeError(
            "The loaded deformable_aggregation_ext does not contain the "
            "single-pass per-camera API. Rebuild it before enabling "
            "ray_object_single_pass_daf."
        )
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
    """Run the released V49 camera path without a global camera carrier.

    This is deliberately forward-only.  Callers must retain the ordinary
    per-camera autograd implementation for training.
    """

    if not _DAF_BUILT or not hasattr(
        deformable_aggregation_ext,
        "deformable_aggregation_rfl_rosa_forward",
    ):
        raise RuntimeError(
            "The loaded deformable_aggregation_ext lacks the V49 streaming "
            "RFL/ROSA entry point. Rebuild the isolated extension."
        )
    if torch.is_grad_enabled():
        raise RuntimeError("streaming V49 CUDA is inference-only")
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


def v49_hybrid_state_update_inference(
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
    """Fuse the released V49 post-head state update into one CUDA launch."""

    if not _DAF_BUILT or not hasattr(
        deformable_aggregation_ext,
        "v49_hybrid_state_update_forward",
    ):
        raise RuntimeError(
            "The loaded deformable_aggregation_ext lacks the V49 fused "
            "hybrid-state update entry point. Rebuild the isolated extension."
        )
    if torch.is_grad_enabled():
        raise RuntimeError("fused V49 hybrid-state update is inference-only")
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
    return deformable_aggregation_ext.v49_hybrid_state_update_forward(
        *(value.contiguous().float() for value in values)
    )


def v49_project_geometry_inference(
    points,
    lidar2cam,
    image_wh,
    intrinsic,
    polynomial,
    half_fov,
    log_extent,
    level_strides,
    points_per_query,
):
    """Fused calibrated projection and ROSA geometry descriptor for V49."""
    if not _DAF_BUILT or not hasattr(
        deformable_aggregation_ext, "v49_project_geometry_forward"
    ):
        raise RuntimeError("isolated extension lacks V49 fused projection")
    if torch.is_grad_enabled():
        raise RuntimeError("fused V49 projection is inference-only")
    values = (
        points, lidar2cam, image_wh, intrinsic, polynomial, half_fov,
        log_extent, level_strides,
    )
    return deformable_aggregation_ext.v49_project_geometry_forward(
        *(value.contiguous().float() for value in values), int(points_per_query)
    )


def v49_state_geometry_inputs_inference(
    direction, log_range, log_size, yaw, velocity
):
    """Build V49's SH state carrier and angular extent in one CUDA launch."""
    if not _DAF_BUILT or not hasattr(
        deformable_aggregation_ext, "v49_state_geometry_inputs_forward"
    ):
        raise RuntimeError("isolated extension lacks V49 state geometry inputs")
    if torch.is_grad_enabled():
        raise RuntimeError("fused V49 state geometry inputs are inference-only")
    return deformable_aggregation_ext.v49_state_geometry_inputs_forward(
        *(value.contiguous().float() for value in (
            direction, log_range, log_size, yaw, velocity
        ))
    )


def v49_build_keypoints_project_geometry_inference(
    direction, log_range, log_size, yaw, learned, lidar2cam, image_wh,
    intrinsic, polynomial, half_fov, log_extent, level_strides,
    points_per_query, fixed_scale,
):
    """Build all 13 box points and their calibrated geometry in CUDA."""
    if not _DAF_BUILT or not hasattr(
        deformable_aggregation_ext,
        "v49_build_keypoints_project_geometry_forward",
    ):
        raise RuntimeError("isolated extension lacks fused V49 keypoints")
    if torch.is_grad_enabled():
        raise RuntimeError("fused V49 keypoint projection is inference-only")
    values = (
        direction, log_range, log_size, yaw, learned, lidar2cam, image_wh,
        intrinsic, polynomial, half_fov, log_extent, level_strides,
    )
    return deformable_aggregation_ext.v49_build_keypoints_project_geometry_forward(
        *(value.contiguous().float() for value in values),
        int(points_per_query), float(fixed_scale),
    )


def v49_point_attention_inference(query, key, reliability, values, valid):
    """Fuse V49's 13-point logits, masked softmax, and value reduction."""
    if not _DAF_BUILT or not hasattr(
        deformable_aggregation_ext, "v49_point_attention_forward"
    ):
        raise RuntimeError("isolated extension lacks fused V49 point attention")
    if torch.is_grad_enabled():
        raise RuntimeError("fused V49 point attention is inference-only")
    return deformable_aggregation_ext.v49_point_attention_forward(
        query.contiguous().float(), key.contiguous().float(),
        reliability.contiguous().float(), values.contiguous().float(),
        valid.contiguous(),
    )


def deformable_aggregation_per_camera_deterministic_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    """Return fixed-order outputs and use an atomic-free backward."""

    if (
        not _DAF_BUILT
        or DeformableAggregationPerCameraDeterministicFunction is None
        or not hasattr(
            deformable_aggregation_ext,
            "deformable_aggregation_per_camera_deterministic_backward",
        )
    ):
        raise RuntimeError(
            "deformable_aggregation_ext was not built with the isolated "
            "deterministic backward candidate"
        )
    return DeformableAggregationPerCameraDeterministicFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def feature_maps_format(feature_maps, inverse=False):
    import torch

    if inverse:
        col_feats, spatial_shape, scale_start_index = feature_maps
        num_cams, num_levels = spatial_shape.shape[:2]

        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        split_size = split_size.cpu().numpy().tolist()

        idx = 0
        cam_split = [1]
        cam_split_size = [sum(split_size[0])]
        for i in range(num_cams - 1):
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        spatial_shape = spatial_shape.cpu().numpy().tolist()
        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

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
