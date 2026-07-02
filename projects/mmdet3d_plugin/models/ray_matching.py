"""Sparse multi-view ray matching helpers for GT-based checks.

Shape convention:
    feature_maps[l]:     (B, N_cam, C_l, H_l, W_l)
    centers_lidar:      (B, N_obj, 3)
    candidate centers:  (B, N_obj, K, 3)
    projections:        (B, N_cam, N_obj, K, 2), normalized image coords
    sampled feats:      (B, N_cam, N_obj, K, C_l)

For sparse keypoint matching, candidate centers are expanded to
    candidate keypoints: (B, N_obj, K, N_key, 3)
and keypoint scores are reduced back to object-candidate curves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .projection import kb_project_points, ocam_project_points


@dataclass
class RayMatchingResult:
    candidate_ranges: torch.Tensor
    candidate_points: torch.Tensor
    points_2d: torch.Tensor
    valid_mask: torch.Tensor
    reference_camera: torch.Tensor
    gt_range: torch.Tensor
    matching_confidence: torch.Tensor
    projection_quality: torch.Tensor
    reliability: torch.Tensor
    visible_count: torch.Tensor


def project_points_with_mask(
    points_lidar: torch.Tensor,
    metas: Dict,
    image_wh: torch.Tensor,
    projection_type: str = "kb_fisheye",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project lidar points and return normalized coords plus validity mask.

    Args:
        points_lidar: (B, N_obj, K, 3)
    Returns:
        points_2d: (B, N_cam, N_obj, K, 2)
        valid: (B, N_cam, N_obj, K)
    """
    if projection_type == "kb_fisheye":
        points_2d = kb_project_points(points_lidar, metas, image_wh)
    elif projection_type == "ocam_fisheye":
        points_2d = ocam_project_points(points_lidar, metas, image_wh)
    else:
        raise ValueError(f"unsupported projection_type={projection_type}")
    valid = (points_2d[..., 0] >= 0.0) & (points_2d[..., 0] <= 1.0)
    valid = valid & (points_2d[..., 1] >= 0.0) & (points_2d[..., 1] <= 1.0)
    return points_2d, valid


def choose_reference_camera(
    center_points_2d: torch.Tensor,
    center_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the most central valid view for each object.

    Args:
        center_points_2d: (B, N_cam, N_obj, 1, 2)
        center_valid: (B, N_cam, N_obj, 1)
    Returns:
        ref_cam: (B, N_obj)
        visible_count: (B, N_obj)
    """
    pts = center_points_2d[..., 0, :]
    valid = center_valid[..., 0]
    dist = torch.linalg.norm(pts - 0.5, dim=-1)
    dist = torch.where(valid, dist, torch.full_like(dist, 1e6))
    ref_cam = dist.argmin(dim=1)
    visible_count = valid.sum(dim=1)
    return ref_cam, visible_count


def make_range_candidates(
    centers_lidar: torch.Tensor,
    lidar2cam: torch.Tensor,
    ref_cam: torch.Tensor,
    offsets: torch.Tensor,
    min_range: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create candidate 3D points along each object's reference-camera ray.

    The first GT-based implementation derives the reference ray from the GT
    center transformed into the selected reference camera. This is equivalent
    to using the inverse-projected reference pixel for this check.

    Args:
        centers_lidar: (B, N_obj, 3)
        lidar2cam: (B, N_cam, 4, 4)
        ref_cam: (B, N_obj)
        offsets: (K,)
    Returns:
        ranges: (B, N_obj, K)
        points_lidar: (B, N_obj, K, 3)
    """
    B, N_obj = centers_lidar.shape[:2]
    K = offsets.numel()
    device = centers_lidar.device
    dtype = centers_lidar.dtype
    ones = centers_lidar.new_ones(B, N_obj, 1)
    centers_h = torch.cat([centers_lidar, ones], dim=-1)

    ranges = centers_lidar.new_empty(B, N_obj, K)
    points = centers_lidar.new_empty(B, N_obj, K, 3)
    cam2lidar = torch.linalg.inv(lidar2cam)

    for b in range(B):
        for n in range(N_obj):
            c = int(ref_cam[b, n].item())
            p_cam_h = lidar2cam[b, c] @ centers_h[b, n]
            p_cam = p_cam_h[:3]
            gt_range = torch.linalg.norm(p_cam).clamp(min=min_range)
            direction = p_cam / gt_range
            cand_ranges = (gt_range + offsets.to(device=device, dtype=dtype)).clamp(
                min=min_range
            )
            cand_cam = direction[None, :] * cand_ranges[:, None]
            cand_h = torch.cat([cand_cam, cand_cam.new_ones(K, 1)], dim=-1)
            cand_lidar = (cam2lidar[b, c] @ cand_h.T).T[:, :3]
            ranges[b, n] = cand_ranges
            points[b, n] = cand_lidar
    return ranges, points


def decoded_box_centers(boxes_lidar: torch.Tensor) -> torch.Tensor:
    """Return geometric box centers for decoded LiDAR boxes.

    The KB20 pkl stores boxes as [x, y, z_bottom, length, width, height, yaw].
    Sparse ray matching should use the geometric center, not the bottom point.
    """
    centers = boxes_lidar[..., :3].clone()
    centers[..., 2] = centers[..., 2] + boxes_lidar[..., 5] * 0.5
    return centers


def make_box_keypoints(
    boxes_lidar: torch.Tensor,
    candidate_centers: torch.Tensor,
    fixed_scales: torch.Tensor,
) -> torch.Tensor:
    """Create Sparse4D-style fixed box keypoints for each range candidate.

    Args:
        boxes_lidar: (B, N_obj, >=7), decoded LiDAR boxes
            [x, y, z_bottom, length, width, height, yaw, ...].
        candidate_centers: (B, N_obj, K, 3)
        fixed_scales: (N_key, 3), e.g. SparseBox3DKeyPointsGenerator.fix_scale.
    Returns:
        keypoints: (B, N_obj, K, N_key, 3)
    """
    sizes = boxes_lidar[..., None, 3:6]
    local = fixed_scales.to(
        device=boxes_lidar.device, dtype=boxes_lidar.dtype
    )[None, None] * sizes
    yaw = boxes_lidar[..., 6]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    x = local[..., 0]
    y = local[..., 1]
    z = local[..., 2]
    rotated = torch.stack(
        [
            cos_yaw[..., None] * x - sin_yaw[..., None] * y,
            sin_yaw[..., None] * x + cos_yaw[..., None] * y,
            z,
        ],
        dim=-1,
    )
    return candidate_centers[..., None, :] + rotated[:, :, None]


def sample_feature_map(
    feature_map: torch.Tensor,
    points_2d: torch.Tensor,
) -> torch.Tensor:
    """Sample one feature level at normalized image coordinates.

    Args:
        feature_map: (B, N_cam, C, H, W)
        points_2d: (B, N_cam, N_obj, K, 2), normalized to [0, 1]
    Returns:
        sampled: (B, N_cam, N_obj, K, C)
    """
    B, N_cam, C = feature_map.shape[:3]
    N_obj, K = points_2d.shape[2:4]
    grid = points_2d * 2.0 - 1.0
    sampled = F.grid_sample(
        feature_map.reshape(B * N_cam, C, *feature_map.shape[-2:]),
        grid.reshape(B * N_cam, N_obj, K, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.reshape(B, N_cam, C, N_obj, K).permute(0, 1, 3, 4, 2)
    return sampled


def _projection_quality(points_2d: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    center_dist = torch.linalg.norm(points_2d - 0.5, dim=-1)
    max_dist = 0.5 ** 0.5
    quality = (1.0 - center_dist / max_dist).clamp(min=0.0, max=1.0)
    return torch.where(valid_mask, quality, torch.zeros_like(quality))


def compute_gt_ray_matching(
    feature_maps: List[torch.Tensor],
    centers_lidar: torch.Tensor,
    metas: Dict,
    offsets: torch.Tensor,
    feature_level: int = 0,
    projection_type: str = "kb_fisheye",
) -> RayMatchingResult:
    """Compute GT-based range reliability curves for object centers."""
    image_wh = metas["image_wh"]
    center_points = centers_lidar[:, :, None, :]
    center_2d, center_valid = project_points_with_mask(
        center_points, metas, image_wh, projection_type=projection_type
    )
    ref_cam, visible_count = choose_reference_camera(center_2d, center_valid)
    candidate_ranges, candidate_points = make_range_candidates(
        centers_lidar, metas["lidar2cam"], ref_cam, offsets
    )
    points_2d, valid_mask = project_points_with_mask(
        candidate_points, metas, image_wh, projection_type=projection_type
    )

    feature_map = feature_maps[feature_level]
    sampled = sample_feature_map(feature_map, points_2d)
    center_sampled = sample_feature_map(feature_map, center_2d)

    B, N_cam, N_obj, K = valid_mask.shape
    matching = centers_lidar.new_full((B, N_obj, K), float("nan"))
    proj_quality = centers_lidar.new_zeros(B, N_obj, K)
    reliability = centers_lidar.new_full((B, N_obj, K), float("nan"))
    quality_per_view = _projection_quality(points_2d, valid_mask)

    for b in range(B):
        for n in range(N_obj):
            ref = int(ref_cam[b, n].item())
            ref_feat = center_sampled[b, ref, n, 0]
            for k in range(K):
                view_mask = valid_mask[b, :, n, k].clone()
                view_mask[ref] = False
                if view_mask.sum() == 0:
                    continue
                feats = sampled[b, view_mask, n, k]
                cos = F.cosine_similarity(ref_feat[None, :], feats, dim=-1)
                matching[b, n, k] = cos.mean()
                proj_quality[b, n, k] = quality_per_view[b, view_mask, n, k].mean()
                reliability[b, n, k] = matching[b, n, k] * proj_quality[b, n, k]

    return RayMatchingResult(
        candidate_ranges=candidate_ranges,
        candidate_points=candidate_points,
        points_2d=points_2d,
        valid_mask=valid_mask,
        reference_camera=ref_cam,
        gt_range=candidate_ranges[..., offsets.numel() // 2],
        matching_confidence=matching,
        projection_quality=proj_quality,
        reliability=reliability,
        visible_count=visible_count,
    )
def compute_gt_ray_keypoint_matching(
    feature_maps: List[torch.Tensor],
    boxes_lidar: torch.Tensor,
    metas: Dict,
    offsets: torch.Tensor,
    fixed_scales: torch.Tensor,
    feature_level: int = 0,
    projection_type: str = "kb_fisheye",
    min_keypoints_per_view: int = 2,
) -> RayMatchingResult:
    """Compute GT-based reliability curves using sparse box keypoints.

    This keeps the same range-candidate setup as the center-point check, but
    compares Sparse4D-style fixed keypoints instead of a single 3D box center.
    Learnable keypoints are query-feature dependent, so this GT-only check uses
    the fixed keypoint scales from the config.
    """
    image_wh = metas["image_wh"]
    centers_lidar = decoded_box_centers(boxes_lidar)
    center_points = centers_lidar[:, :, None, :]
    center_2d, center_valid = project_points_with_mask(
        center_points, metas, image_wh, projection_type=projection_type
    )
    ref_cam, visible_count = choose_reference_camera(center_2d, center_valid)
    candidate_ranges, candidate_centers = make_range_candidates(
        centers_lidar, metas["lidar2cam"], ref_cam, offsets
    )

    candidate_keypoints = make_box_keypoints(
        boxes_lidar, candidate_centers, fixed_scales
    )
    gt_keypoints = make_box_keypoints(
        boxes_lidar, centers_lidar[:, :, None, :], fixed_scales
    )[:, :, 0]

    B, N_obj, K, N_key = candidate_keypoints.shape[:4]
    flat_candidate_keypoints = candidate_keypoints.reshape(B, N_obj, K * N_key, 3)
    keypoints_2d_flat, key_valid_flat = project_points_with_mask(
        flat_candidate_keypoints, metas, image_wh, projection_type=projection_type
    )
    keypoints_2d = keypoints_2d_flat.reshape(B, -1, N_obj, K, N_key, 2)
    key_valid = key_valid_flat.reshape(B, -1, N_obj, K, N_key)

    gt_keypoints_2d, gt_key_valid = project_points_with_mask(
        gt_keypoints[:, :, None].reshape(B, N_obj, N_key, 3),
        metas,
        image_wh,
        projection_type=projection_type,
    )
    gt_keypoints_2d = gt_keypoints_2d.reshape(B, -1, N_obj, N_key, 2)
    gt_key_valid = gt_key_valid.reshape(B, -1, N_obj, N_key)

    center_candidate_2d, center_candidate_valid = project_points_with_mask(
        candidate_centers, metas, image_wh, projection_type=projection_type
    )

    feature_map = feature_maps[feature_level]
    sampled = sample_feature_map(
        feature_map, keypoints_2d_flat.reshape(B, -1, N_obj, K * N_key, 2)
    )
    sampled = sampled.reshape(B, -1, N_obj, K, N_key, feature_map.shape[2])
    ref_sampled = sample_feature_map(
        feature_map, gt_keypoints_2d.reshape(B, -1, N_obj, N_key, 2)
    )
    ref_sampled = ref_sampled.reshape(B, -1, N_obj, N_key, feature_map.shape[2])

    matching = centers_lidar.new_full((B, N_obj, K), float("nan"))
    proj_quality = centers_lidar.new_zeros(B, N_obj, K)
    reliability = centers_lidar.new_full((B, N_obj, K), float("nan"))
    quality_per_keypoint = _projection_quality(keypoints_2d, key_valid)
    view_valid = key_valid.sum(dim=-1) >= int(min_keypoints_per_view)

    for b in range(B):
        for n in range(N_obj):
            ref = int(ref_cam[b, n].item())
            ref_feats = ref_sampled[b, ref, n]
            ref_valid = gt_key_valid[b, ref, n]
            for k in range(K):
                view_scores = []
                view_qualities = []
                for c in range(key_valid.shape[1]):
                    if c == ref or not bool(view_valid[b, c, n, k]):
                        continue
                    kp_mask = key_valid[b, c, n, k] & ref_valid
                    if int(kp_mask.sum().item()) < int(min_keypoints_per_view):
                        continue
                    feats = sampled[b, c, n, k, kp_mask]
                    cos = F.cosine_similarity(ref_feats[kp_mask], feats, dim=-1)
                    view_scores.append(cos.mean())
                    view_qualities.append(
                        quality_per_keypoint[b, c, n, k, kp_mask].mean()
                    )
                if not view_scores:
                    continue
                matching[b, n, k] = torch.stack(view_scores).mean()
                proj_quality[b, n, k] = torch.stack(view_qualities).mean()
                reliability[b, n, k] = matching[b, n, k] * proj_quality[b, n, k]

    return RayMatchingResult(
        candidate_ranges=candidate_ranges,
        candidate_points=candidate_centers,
        points_2d=center_candidate_2d,
        valid_mask=center_candidate_valid & view_valid.any(dim=1)[:, None],
        reference_camera=ref_cam,
        gt_range=candidate_ranges[..., offsets.numel() // 2],
        matching_confidence=matching,
        projection_quality=proj_quality,
        reliability=reliability,
        visible_count=visible_count,
    )
