"""SparseBEV for KITTI-360's mixed pinhole + MEI rig.

SparseBEV reaches the image through ``sparsebev_sampling.project_ocam``, which
normalizes its already-decoded sample points and hands them to DETR3D's shared
``_feature_sampling_ocam``.  Only that projection is wrong for KITTI-360; the
box-relative point generation, the group/level mixing, the first-valid-camera
selection and ``AdaptiveMixing`` are correct as released and are what makes
the row SparseBEV rather than something else.

``SparseBEVTransformerDecoderLayer`` constructs ``SparseBEVSampling`` directly
with no config hook, so swapping the projection through the config would mean
subclassing the transformer, the decoder and the layer purely to reach one
attribute.  Instead the detector rebinds ``project_ocam`` inside the released
module once, at construction.  That is process-wide, which is exactly right
here: an arm built from this config only ever sees mixed-camera data, one
training process runs one arm, and each DDP rank is its own process.  Naming
the detector in the config keeps the swap visible rather than hiding it in an
import side effect.

The replacement drops ``pc_range``: ``project_ocam`` normalizes into it only
because ``_feature_sampling_ocam`` immediately un-normalizes, and the mixed
projector takes lidar-frame points directly, so both steps cancel.
"""

from __future__ import annotations

from mmdet3d.registry import MODELS

from sparsebev.sparsebev import SparseBEV

from .mixed_sampling import project_points_mixed


def _project_mixed(sample_points, pc_range, img_metas, return_rays=False):
    """``project_ocam``'s signature, backed by the mixed-camera projection."""
    del pc_range
    batch = sample_points.shape[0]
    flat = sample_points.reshape(batch, -1, 3)
    return project_points_mixed(flat, img_metas, return_rays=return_rays)


def install_mixed_projection() -> None:
    """Point the released SparseBEV sampler at the mixed-camera projection."""
    import sparsebev.sparsebev_sampling as sampling_module

    if getattr(sampling_module.project_ocam, "_kitti360_mixed", False):
        return
    _project_mixed._kitti360_mixed = True
    sampling_module.project_ocam = _project_mixed


@MODELS.register_module()
class SparseBEVMixedCamera(SparseBEV):
    """``SparseBEV`` with KITTI-360's per-camera lens models."""

    def __init__(self, *args, **kwargs) -> None:
        install_mixed_projection()
        super().__init__(*args, **kwargs)
