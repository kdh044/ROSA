import os
"""StreamPETR for KITTI-360's mixed pinhole + MEI rig.

``StreamPETRHead``'s OCam port replaced the released
``lidar2img.inverse()`` unprojection with a calibrated ray grid, but that grid
comes from one shared ``OcamCamera`` and is broadcast to all four cameras.
Two changes make it correct here and nothing else moves:

* the ray grid becomes per-camera (``mixed_geometry``, shared with the PETR
  and DAPETR ports so all three arms lift features identically), and
  ``position_embeding``'s broadcast of it gains the camera axis;
* the detector's ``_location`` no longer reads ``ocam.height``.  That call
  only fixes ``memory_centers``' shape -- ``position_embeding`` uses the ray
  grid, not these pixel values -- but the attribute is ``None`` on a mixed
  head, and KITTI-360's input is 1408x376 rather than square anyway.

Temporal propagation stays off, as in the Fisheye3DOD arm: the KITTI-360
protocol here is 2 Hz, so consecutive samples are half a second and several
metres apart and the memory bank would be modelling a different scene.
"""

from __future__ import annotations

import torch
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet3d.registry import MODELS

from fisheye3dod.models.petr.streampetr_head_ocam import StreamPETRHead
from fisheye3dod.models.petr.streampetr_ocam import StreamPETROCam
from fisheye3dod.models.petr.streampetr_misc import locations, topk_gather

from .constants import CAMERAS, IMAGE_SIZE
from .mixed_geometry import MixedCameraGeometryMixin, make_mixed_native_geometry


@MODELS.register_module()
class StreamPETRMixedCameraHead(MixedCameraGeometryMixin, StreamPETRHead):
    """``StreamPETRHead`` with one camera model per camera."""

    _DUMMY_OCAM_PATH = (
        os.path.join(os.environ.get("ROSA_DATA_ROOT", "data/Fisheye3DOD"), "calib_results.txt")
    )
    _DUMMY_OCAM_FOV = 220.0

    def __init__(
        self,
        camera_calibration,
        ocam_path: str = _DUMMY_OCAM_PATH,
        ocam_fov: float = _DUMMY_OCAM_FOV,
        **kwargs,
    ) -> None:
        self._camera_calibration = self._validate_calibration(
            camera_calibration
        )
        super().__init__(ocam_path=ocam_path, ocam_fov=ocam_fov, **kwargs)
        self.ocam = None

    def _make_native_geometry(self, height, width):
        """``[C,H,W,3]`` rays only.

        The mixin returns ``(rays, valid)`` because the PETR/DAPETR heads mask
        invalid pixels; StreamPETR's caller registers the return value
        directly as its ``native_ray_grid`` buffer and has no mask, so the
        validity is dropped here rather than changing the shared helper.
        """
        rays, _valid = make_mixed_native_geometry(
            self._camera_calibration, height, width
        )
        return rays

    def position_embeding(self, data, memory_centers, topk_indexes, img_metas):
        """Copy of the parent method with the ray broadcast made per-camera.

        The parent does ``rays_cam.view(1, 1, H*W, 3).expand(B, N, ...)``,
        which hands every camera the same lens.  ``rays_cam`` is ``[C,H,W,3]``
        here, so the camera axis is real and the reshape keeps it.
        """
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)
        N = data['intrinsics'].size(1)

        intrinsic = torch.stack(
            [data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]],
            dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H * W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = (
            topk_indexes.size(1) if topk_indexes is not None else LEN)
        D = self.coords_d.shape[0]

        rays_cam = self._native_rays(
            H, W, device=memory_centers.device, dtype=memory_centers.dtype)
        if tuple(rays_cam.shape) != (N, H, W, 3):
            raise ValueError(
                f"expected per-camera rays {(N, H, W, 3)}, "
                f"got {tuple(rays_cam.shape)}"
            )
        rays_cam = rays_cam.reshape(1, N, H * W, 3).expand(B, N, H * W, 3)

        cam2lidar = data['cam2lidar'].view(B, N, 4, 4)
        rays_ego = torch.einsum(
            'bnij,bnkj->bnki', cam2lidar[..., :3, :3], rays_cam)
        origin_ego = cam2lidar[..., :3, 3]

        rays_ego = rays_ego.reshape(B, LEN, 3)
        origin_ego = origin_ego.view(B, N, 1, 3).expand(
            B, N, H * W, 3).reshape(B, LEN, 3)

        rays_ego = rays_ego.detach().view(B, LEN, 1, 3)
        origin_ego = origin_ego.detach().view(B, LEN, 1, 3)
        rays_ego = topk_gather(rays_ego, topk_indexes).repeat(1, 1, D, 1)
        origin_ego = topk_gather(origin_ego, topk_indexes).repeat(1, 1, D, 1)

        coords_d = self.coords_d.view(1, 1, D, 1).repeat(
            B, num_sample_tokens, 1, 1)
        coords3d = origin_ego + rays_ego * coords_d

        coords3d = (
            coords3d - self.position_range[0:3]
        ) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D * 3)

        pos_embed = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)
        intrinsic = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        cone = torch.cat(
            [intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)
        return coords_position_embeding, cone


@MODELS.register_module()
class StreamPETRMixedCamera(StreamPETROCam):
    """``StreamPETROCam`` with the square-fisheye assumption removed."""

    def _location(self, img_feats_level, img_metas):
        B, N, C, H, W = img_feats_level.shape
        if N != len(CAMERAS):
            raise ValueError(f"expected {len(CAMERAS)} cameras, got {N}")
        # ``locations()`` only fixes memory_centers' (BN, H, W) shape --
        # ``position_embeding`` reads the calibrated ray grid, not these pixel
        # values -- but the parent takes the size from ``ocam.height``, which
        # is ``None`` on a mixed head and would not be square regardless.
        pad_w, pad_h = IMAGE_SIZE
        feat_bn = img_feats_level.flatten(0, 1)
        return locations(feat_bn, self.stride, pad_h, pad_w)[None].repeat(
            B * N, 1, 1, 1)
