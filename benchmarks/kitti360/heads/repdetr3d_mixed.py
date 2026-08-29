import os
"""RepDETR3D cross-attention for KITTI-360's mixed pinhole + MEI rig.

``RepDetr3DCrossAttnOCam`` inlines its own differentiable OCam
``world2cam`` and projects every camera through it.  Only that projection is
replaced here; the deformable key-point generation, the query-conditioned
weights, the temporal decoder layers and the losses stay as released.

The projection needs ``img_metas`` and ``_project_points`` does not receive
it, so ``forward`` stashes it for the duration of the call.  It cannot be read
from the constructor instead: ``image_affine`` is per-sample, because
``DAPETRImageAugment``/``DAPETRGlobalRotScale`` compose their augmentation into
it, and projecting with the un-augmented affine would put every sampled
feature in the wrong place on exactly the batches that are being augmented.
"""

from __future__ import annotations

from mmdet3d.registry import MODELS

from fisheye3dod.models.petr.repdetr3d_transformer import RepDetr3DCrossAttnOCam

from .mixed_sampling import project_points_mixed


@MODELS.register_module()
class RepDetr3DCrossAttnMixed(RepDetr3DCrossAttnOCam):
    """``RepDetr3DCrossAttnOCam`` with per-camera pinhole/MEI projection."""

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
        self._img_metas = None
        # The inherited OCam tables describe a lens this module never uses and
        # would otherwise make it look calibrated for Fisheye3DOD.
        self.ocam = None

    def _project_points(self, points_ego, cam2lidar, return_rays=False):
        """Parent signature, mixed-camera projection.

        ``cam2lidar`` is accepted and deliberately unused: the mixed projector
        reads ``lidar2cam`` from ``img_metas``, and the loader keeps the two
        exact inverses of each other through augmentation
        (``dataset.py`` rewrites both from the same transform).  Taking the
        one the projector validates keeps a single source for the extrinsic
        rather than trusting two paths to agree.
        """
        if self._img_metas is None:
            raise RuntimeError(
                "_project_points called outside forward(); img_metas unset"
            )
        if cam2lidar.ndim != 4 or cam2lidar.shape[-2:] != (4, 4):
            raise ValueError("cam2lidar must be [B,C,4,4]")
        return project_points_mixed(
            points_ego, self._img_metas, return_rays=return_rays
        )

    def forward(self, instance_feature, query_pos, feat_flatten,
                reference_points, spatial_flatten, level_start_index,
                pc_range, data, img_metas):
        self._img_metas = img_metas
        try:
            return super().forward(
                instance_feature, query_pos, feat_flatten, reference_points,
                spatial_flatten, level_start_index, pc_range, data, img_metas,
            )
        finally:
            self._img_metas = None
