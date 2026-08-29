"""Sparse4D-format packing that also carries KITTI-360's lens calibration.

``PackSparse4DInputs`` hands Sparse4D its calibration as batched *inputs*
(``projection_mat``, ``lidar2cam``, ``ocam_intrinsic``, ...), which is what
ROSA's own decoder reads.  The DETR3D-family ports read theirs from
``img_metas`` instead -- that is where ``_feature_sampling_ocam`` looks, and
the mixed replacement keeps the same contract -- so on this rig those arms
would otherwise find nothing there and fall through to the pinhole
``lidar2img`` matmul without raising.

Extending ``META_KEYS`` rather than editing the shared transform keeps
``dependencies/rosa_detector_host`` untouched; the added keys are inert for
any consumer that does not look for them, so the ROSA arm is unaffected.

``image_affine`` is the reason this cannot be baked into a config: the
augmentation composes into it per sample, so projecting with a fixed
calibration would misplace every sampled feature on augmented batches.
"""

from __future__ import annotations

from mmdet3d.registry import TRANSFORMS

from projects.mmdet3d_plugin.datasets.transforms.pack_sparse4d import (
    PackSparse4DInputs,
)

from rosa.mixed_projection import MIXED_CAMERA_METADATA


@TRANSFORMS.register_module()
class KITTI360Sparse4DPackInputs(PackSparse4DInputs):
    """``PackSparse4DInputs`` plus the mixed-camera calibration in metainfo."""

    META_KEYS = PackSparse4DInputs.META_KEYS + MIXED_CAMERA_METADATA
