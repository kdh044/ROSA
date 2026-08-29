"""Mixed pinhole + MEI feature sampling for the projection-based baselines.

DETR3D, Graph-DETR3D and SparseBEV all reach the image through one function --
``detr3d.detr3d_transformer._feature_sampling_ocam`` -- which projects 3D
reference points with a single shared ``OcamCamera``.  That is correct for
Fisheye3DOD (four identical OCam fisheyes) and wrong for KITTI-360, whose rig
pairs two rectified pinhole cameras with two native MEI fisheyes that share
neither a lens model nor intrinsics.

This module provides a drop-in replacement with the *same* return contract, so
the baselines' query generation, attention weights, decoder recurrence, heads,
matching and losses stay exactly the released implementation and only the
projection changes -- the same scope of edit the OCam port already made.

Two properties of KITTI-360 that the OCam path has no equivalent of, and that
every caller here therefore depends on:

* **the fisheyes are cropped.**  The network is fed a
  ``FISHEYE_CROPS``-sized window of the 1400x1400 native image rescaled to
  ``IMAGE_SIZE``.  Projection has to land in *network* pixels, not native
  ones, or the sampled features come from the wrong place.  ``image_affine``
  carries that mapping (and picks up any train-time augmentation), which is
  why projection goes through ``mixed_projection._network_pixels`` rather
  than reimplementing the lens models here.
* **MEI visibility is decided by the sign of z.**  For KITTI-360's
  ``xi = 2.21 / 2.55`` the model-domain condition ``z/||P|| + xi > 0`` is true
  for *every* direction including straight out the back, so it cannot cull
  anything; ``mixed_projection`` culls on ``z > 0`` like the official devkit.
  Relying on that shared implementation keeps the ghost-box fix in one place.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rosa.mixed_projection import MIXED_CAMERA_METADATA, _network_pixels

from .constants import IMAGE_SIZE


def _bilinear_sampler():
    """The released deterministic four-gather bilinear read, imported lazily.

    Reuse it rather than ``torch.grid_sample`` so the mixed arm and the OCam
    arm read pixels identically and any gap between them is the projection
    alone.

    The import is deferred because ``detr3d`` and ``fisheye3dod.models.petr``
    both register a ``BBox3DL1Cost`` under the same global MMEngine key and so
    cannot coexist in one process.  Only ``feature_sampling_mixed`` needs the
    sampler; ``project_points_mixed`` does not, and the RepDETR3D and SRCN3D
    ports use only the latter.  A module-level import here would drag the
    DETR3D tree into those fisheye3dod-based arms and fail at import time.
    """
    from detr3d.detr3d_transformer import _deterministic_bilinear_sample

    return _deterministic_bilinear_sample


def has_mixed_camera_metadata(img_metas) -> bool:
    """True when ``img_metas`` carries KITTI-360's mixed-rig calibration."""
    if not img_metas:
        return False
    meta = img_metas[0]
    return all(name in meta for name in MIXED_CAMERA_METADATA)


def _stack_metadata(img_metas, name: str, like: torch.Tensor) -> torch.Tensor:
    """Stack a per-sample calibration field into a batched tensor."""
    values = []
    for meta in img_metas:
        if name not in meta:
            raise KeyError(f"mixed-camera metadata missing '{name}'")
        values.append(np.asarray(meta[name]))
    stacked = np.stack(values)
    if name == "camera_model_id":
        return torch.as_tensor(stacked, dtype=torch.long, device=like.device)
    return like.new_tensor(stacked)


def _image_wh(img_metas, batch: int, cameras: int, like: torch.Tensor):
    """Network-input width/height per camera, as ``[B,C,2]``.

    Every camera is resized to ``IMAGE_SIZE`` by the loader, but read it back
    from ``img_shape`` when present so a config that changes the input size
    cannot silently keep projecting against the old one.
    """
    meta = img_metas[0]
    shapes = meta.get("img_shape")
    if shapes is None:
        width, height = IMAGE_SIZE
        per_camera = [(width, height)] * cameras
    else:
        shapes = np.asarray(shapes)
        if shapes.ndim == 1:  # one (h, w) shared by every camera
            per_camera = [(int(shapes[1]), int(shapes[0]))] * cameras
        else:
            if shapes.shape[0] != cameras:
                raise ValueError(
                    f"img_shape lists {shapes.shape[0]} cameras, features have "
                    f"{cameras}"
                )
            per_camera = [(int(s[1]), int(s[0])) for s in shapes]
    return like.new_tensor(np.asarray(per_camera)).unsqueeze(0).expand(
        batch, cameras, 2
    ).contiguous()


def project_points_mixed(points_lidar, img_metas, return_rays: bool = False):
    """Project lidar-frame points to normalized network pixels.

    The mixed-camera counterpart of ``graphdetr3d_transformer.
    _project_points_ocam``: same signature, same return contract, and like it
    this takes *unnormalized* points so callers that already decoded their
    boxes do not have to re-normalize them.

    Args:
        points_lidar: ``[B,Q,3]`` points in the lidar/ego frame.

    Returns:
        ``projected`` ``[B,C,Q,2]`` normalized to the network input (values
        outside ``[0,1]`` are simply invalid) and ``valid`` ``[B,C,Q]``, plus
        camera-frame rays ``[B,C,Q,3]`` when ``return_rays`` is set.
    """
    if points_lidar.ndim != 3 or points_lidar.shape[-1] != 3:
        raise ValueError("points_lidar must be [B,Q,3]")
    batch, num_query = points_lidar.shape[:2]
    metadata = {
        name: _stack_metadata(img_metas, name, points_lidar)
        for name in MIXED_CAMERA_METADATA
    }
    lidar2cam = metadata["lidar2cam"]
    if lidar2cam.ndim != 4 or lidar2cam.shape[-2:] != (4, 4):
        raise ValueError("lidar2cam must be [B,C,4,4]")
    cameras = lidar2cam.shape[1]
    if lidar2cam.shape[0] != batch:
        raise ValueError("lidar2cam batch disagrees with the points")

    # [B,C,Q,3] camera-frame coordinates, then a [B,C,Q,1,3] view because
    # ``_network_pixels`` indexes points as [B,C,N,P,*].
    homogeneous = torch.cat(
        (points_lidar, torch.ones_like(points_lidar[..., :1])), dim=-1
    )
    camera_xyz = torch.einsum("bcij,bqj->bcqi", lidar2cam, homogeneous)[..., :3]
    camera_rays = F.normalize(torch.nan_to_num(camera_xyz), dim=-1, eps=1e-6)

    finite = torch.isfinite(camera_xyz).all(dim=-1)
    fallback = torch.zeros_like(camera_xyz)
    fallback[..., 2] = 1.0
    safe_xyz = torch.where(finite[..., None], camera_xyz, fallback)

    pixels, domain_valid = _network_pixels(safe_xyz.unsqueeze(3), metadata)
    pixels = pixels.squeeze(3)
    domain_valid = domain_valid.squeeze(3)

    image_wh = _image_wh(img_metas, batch, cameras, points_lidar)
    width = image_wh[..., 0][:, :, None]
    height = image_wh[..., 1][:, :, None]
    projected = torch.stack(
        (pixels[..., 0] / width, pixels[..., 1] / height), dim=-1
    )
    valid = (
        finite
        & domain_valid
        & torch.isfinite(projected).all(dim=-1)
        & projected[..., 0].gt(0.0)
        & projected[..., 0].lt(1.0)
        & projected[..., 1].gt(0.0)
        & projected[..., 1].lt(1.0)
    )
    projected = torch.nan_to_num(projected)
    if return_rays:
        return projected, valid, camera_rays
    return projected, valid


def feature_sampling_mixed(
    mlvl_feats,
    ref_pt,
    pc_range,
    img_metas,
    no_sampling: bool = False,
    return_observation_geometry: bool = False,
):
    """``_feature_sampling_ocam`` with KITTI-360's per-camera lens models.

    Args:
        ref_pt: normalized reference points ``[B,Q,3]`` in ``pc_range``.

    Returns:
        Matches ``_feature_sampling_ocam`` exactly -- with ``no_sampling`` the
        normalized pixels ``[B,C,Q,2]`` and validity ``[B,C,Q,1]``, otherwise
        ``(reference_points_3d, sampled_feats, mask)`` -- plus camera-frame
        rays ``[B,C,Q,3]`` when ``return_observation_geometry`` is set.
    """
    reference_points = ref_pt.clone()
    reference_points_3d = reference_points.clone()
    for axis in range(3):
        reference_points[..., axis] = (
            reference_points[..., axis]
            * (pc_range[axis + 3] - pc_range[axis])
            + pc_range[axis]
        )

    batch, num_query = reference_points.shape[:2]
    projected, valid, camera_rays = project_points_mixed(
        reference_points, img_metas, return_rays=True
    )
    cameras = projected.shape[1]

    if no_sampling:
        if return_observation_geometry:
            return projected, valid[..., None], camera_rays
        return projected, valid[..., None]

    mask = valid.view(batch, cameras, 1, num_query, 1, 1).permute(
        0, 2, 3, 1, 4, 5
    )
    sampled_feats = []
    channels = None
    bilinear_sample = _bilinear_sampler()
    for feat in mlvl_feats:
        feature_batch, feature_cameras, channels, height_f, width_f = feat.shape
        if feature_batch != batch or feature_cameras != cameras:
            raise ValueError(
                "feature camera layout disagrees with mixed-camera metadata"
            )
        flattened = feat.reshape(batch * cameras, channels, height_f, width_f)
        level_grid = projected.reshape(batch * cameras, num_query, 2)
        sampled = bilinear_sample(flattened, level_grid)
        sampled = sampled.view(batch, cameras, channels, num_query, 1)
        sampled_feats.append(sampled.permute(0, 2, 3, 1, 4))
    sampled_feats = torch.stack(sampled_feats, dim=-1)
    sampled_feats = sampled_feats.view(
        batch, channels, num_query, cameras, 1, len(mlvl_feats)
    )
    result = (reference_points_3d, sampled_feats, mask.to(dtype=ref_pt.dtype))
    if return_observation_geometry:
        return result + (camera_rays,)
    return result
