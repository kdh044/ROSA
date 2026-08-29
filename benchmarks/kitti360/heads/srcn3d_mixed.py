import os
"""SRCN3D RoI head for KITTI-360's mixed pinhole + MEI rig.

``SparseRoIHead3DOCam`` reaches the image through a single
``_OCamPixelProjector``.  This module supplies a drop-in projector with the
same ``project(points_ego, cam2lidar, return_rays=False)`` interface backed by
the per-camera pinhole/MEI models, and a head subclass that installs it.
Everything else -- proposal refinement, the RoI extractor, the cascade stages,
the losses -- is the released SRCN3D.

Unlike the other ports this one returns **absolute network pixels** rather
than normalized ones, because SRCN3D turns projected box corners into RoIs via
``bbox2roi`` and the RoI extractor addresses feature maps in pixels.  Getting
that wrong would not raise; it would quietly extract every RoI from a
sub-pixel patch at the top-left corner.
"""

from __future__ import annotations

import torch

from mmdet3d.registry import MODELS

from fisheye3dod.models.petr.srcn3d_roi_head import SparseRoIHead3DOCam
from rosa.mixed_projection import MIXED_CAMERA_METADATA, _network_pixels

from .mixed_sampling import _stack_metadata


class MixedPixelProjector:
    """``_OCamPixelProjector``'s interface, per-camera pinhole/MEI behind it.

    ``img_metas`` has to be pushed in before each use (``SparseRoIHead3DOCam``
    calls ``project`` from a helper that does not forward it) and cannot be
    baked in at construction, because ``image_affine`` is per-sample:
    ``DAPETRImageAugment``/``DAPETRGlobalRotScale`` compose their augmentation
    into it, and projecting with the un-augmented affine would misplace every
    RoI on exactly the batches being augmented.
    """

    def __init__(self) -> None:
        self.img_metas = None

    def project(self, points_ego, cam2lidar, return_rays: bool = False):
        """Project ego-frame points, per camera, into network pixels.

        Args:
            points_ego: ``[B,C,Q,3]``, already broadcast per camera.
            cam2lidar: ``[B,C,4,4]``; used for the camera-frame rotation
                exactly as the released projector does.

        Returns:
            ``pixel_xy`` ``[B,C,Q,2]`` in network-image pixels and ``valid``
            ``[B,C,Q]``, plus camera-frame rays ``[B,C,Q,3]`` when asked.
        """
        if self.img_metas is None:
            raise RuntimeError("MixedPixelProjector used without img_metas")
        if points_ego.ndim != 4 or points_ego.shape[-1] != 3:
            raise ValueError("points_ego must be [B,C,Q,3]")

        rotation = cam2lidar[..., :3, :3]
        translation = cam2lidar[..., :3, 3]
        offset = points_ego - translation.unsqueeze(-2)
        points_cam = torch.einsum("bcji,bcqj->bcqi", rotation, offset)

        metadata = {
            name: _stack_metadata(self.img_metas, name, points_ego)
            for name in MIXED_CAMERA_METADATA
        }
        finite = torch.isfinite(points_cam).all(dim=-1)
        fallback = torch.zeros_like(points_cam)
        fallback[..., 2] = 1.0
        safe = torch.where(finite[..., None], points_cam, fallback)

        # ``_network_pixels`` indexes points as [B,C,N,P,*], so the singleton
        # P axis added above comes back on both returns and has to come off
        # both of them -- dropping it only from the pixels leaves the validity
        # mask one rank too wide to combine with ``finite``.
        pixels, domain_valid = _network_pixels(safe.unsqueeze(3), metadata)
        pixels = pixels.squeeze(3)
        domain_valid = domain_valid.squeeze(3)
        valid = finite & domain_valid & torch.isfinite(pixels).all(dim=-1)
        pixels = torch.nan_to_num(pixels)
        if return_rays:
            rays = torch.nn.functional.normalize(
                torch.nan_to_num(points_cam), dim=-1, eps=1e-6
            )
            return pixels, valid, rays
        return pixels, valid


@MODELS.register_module()
class SparseRoIHead3DMixed(SparseRoIHead3DOCam):
    """``SparseRoIHead3DOCam`` with the mixed-camera projector installed."""

    # The parent builds an ``OcamCamera`` in its constructor and needs some
    # parseable calibration file even though this subclass never reads its
    # values.  Point at the Fisheye3DOD calibration already vendored here
    # rather than asking every KITTI-360 config to know about an OCam detail
    # that is irrelevant to it -- the same shim ``DAPETRMixedCameraHead`` and
    # ``PETRMixedCameraHead`` use.
    _DUMMY_OCAM_PATH = (
        os.path.join(os.environ.get("ROSA_DATA_ROOT", "data/Fisheye3DOD"), "calib_results.txt")
    )
    _DUMMY_OCAM_FOV = 220.0

    def __init__(self, *args, ocam_path: str = _DUMMY_OCAM_PATH,
                 ocam_fov: float = _DUMMY_OCAM_FOV, **kwargs) -> None:
        super().__init__(*args, ocam_path=ocam_path, ocam_fov=ocam_fov, **kwargs)
        # Replaces the OCam projector the parent just built.
        self.projector = MixedPixelProjector()

    def _bbox_forward(self, stage, mlvl_feats, proposal_list, object_feats,
                      img_metas):
        self.projector.img_metas = img_metas
        try:
            return super()._bbox_forward(
                stage, mlvl_feats, proposal_list, object_feats, img_metas
            )
        finally:
            self.projector.img_metas = None
