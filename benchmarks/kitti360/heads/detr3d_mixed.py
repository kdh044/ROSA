"""DETR3D and Graph-DETR3D cross-attention for KITTI-360's mixed rig.

Both baselines reach the image through a single projection helper that assumes
one shared ``OcamCamera``.  These subclasses swap that helper for the
per-camera pinhole/MEI projection in ``mixed_sampling`` and change nothing
else: query generation, attention weights, decoder recurrence, the heads, the
matcher and every loss stay the released implementation, so a row in the table
still measures the published method rather than a rewrite of it.

``research_archive/baselines`` is shared with the Fisheye3DOD experiments and
is imported, never edited, for the same reason ``dependencies/fisheye3dod``
is: an edit there would silently move Table 1's numbers.
"""

from __future__ import annotations

import numpy as np
import torch
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet3d.registry import MODELS

from detr3d.detr3d import DETR3D
from detr3d.detr3d_transformer import Detr3DCrossAtten
from detr3d.graphdetr3d_transformer import GraphDetr3DCrossAttnOCam
from sparsebev.sparsebev import SparseBEV

from .mixed_sampling import feature_sampling_mixed, project_points_mixed


@MODELS.register_module()
class Detr3DCrossAttenMixed(Detr3DCrossAtten):
    """``Detr3DCrossAtten`` with the OCam projection replaced.

    The parent's ``forward`` calls the module-level ``feature_sampling``,
    which dispatches on ``'ocam_intrinsic' in img_metas[0]``.  KITTI-360
    samples carry mixed-rig calibration instead and would silently fall
    through to the *pinhole ``lidar2img`` matmul* branch -- a projection that
    is simply wrong for the two MEI fisheyes and, worse, would not raise.
    Overriding the sampling call makes the choice explicit and lets the mixed
    path assert on its own metadata.
    """

    def forward(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        reference_points=None,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2)
        bs, num_query, _ = query.size()

        use_portable_fusion = self.enable_ray_hierarchical_fusion
        sampled_result = feature_sampling_mixed(
            value,
            reference_points,
            self.pc_range,
            kwargs["img_metas"],
            return_observation_geometry=use_portable_fusion,
        )
        if use_portable_fusion:
            reference_points_3d, output, mask, camera_rays = sampled_result
        else:
            reference_points_3d, output, mask = sampled_result
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        if use_portable_fusion:
            observations = output.permute(0, 2, 3, 4, 5, 1).contiguous()
            valid = mask[:, 0].to(torch.bool).expand(
                bs, num_query, self.num_cams, self.num_points, self.num_levels
            )
            rays = camera_rays.permute(0, 2, 1, 3)[:, :, :, None, None]
            rays = rays.expand(
                bs, num_query, self.num_cams, self.num_points,
                self.num_levels, 3,
            )
            observations = self.ray_feature_encoder(observations, rays, valid)
            output, _ = self.hierarchical_aggregator(observations, valid, query)
            output = output.permute(1, 0, 2)
        else:
            attention_weights = self.attention_weights(query).view(
                bs, 1, num_query, self.num_cams, self.num_points,
                self.num_levels,
            )
            attention_weights = attention_weights.sigmoid() * mask
            output = output * attention_weights
            output = output.sum(-1).sum(-1).sum(-1)
            output = output.permute(2, 0, 1)

        output = self.output_proj(output)
        pos_feat = self.position_encoder(
            inverse_sigmoid(reference_points_3d)
        ).permute(1, 0, 2)
        return self.dropout(output) + inp_residual + pos_feat


@MODELS.register_module()
class GraphDetr3DCrossAttnMixed(GraphDetr3DCrossAttnOCam):
    """Graph-DETR3D's graph-structured attention on the mixed rig.

    Graph-DETR3D projects its dynamic graph points through
    ``_project_points_ocam`` rather than through DETR3D's sampler, so it needs
    its own swap.  The signature and return contract of the replacement match,
    which is why the parent's ``forward`` can be reused verbatim with only the
    module-level name rebound for the duration of the call.
    """

    def forward(self, *args, **kwargs):
        import detr3d.graphdetr3d_transformer as graph_module

        original = graph_module._project_points_ocam
        graph_module._project_points_ocam = project_points_mixed
        try:
            return super().forward(*args, **kwargs)
        finally:
            graph_module._project_points_ocam = original


class MixedProjectionMetadataMixin:
    """Skip DETR3D's OCam input marshalling; the pack already did the work.

    ``DETR3D.add_projection_metadata`` copies ``ocam_intrinsic`` / ``ocam_pol``
    / ``ocam_lens_fov_deg`` out of the batched *inputs* and onto each sample's
    meta dict, and raises when they are absent.  KITTI-360 has no OCam
    polynomial to supply -- its lenses are a rectified pinhole pair and two MEI
    fisheyes -- and ``KITTI360Sparse4DPackInputs`` already put the equivalent
    calibration straight into ``metainfo``, which is the same dict this method
    writes into.  So the marshalling is not merely inapplicable, it is
    redundant, and the only part still needed is ``pad_shape``.

    That field is read downstream and the parent derives it from ``image_wh``;
    here it comes from ``img_shape``, which the loader records per camera.
    """

    @staticmethod
    def add_projection_metadata(batch_input_metas, batch_inputs_dict):
        from rosa.mixed_projection import MIXED_CAMERA_METADATA

        for meta in batch_input_metas:
            missing = [n for n in MIXED_CAMERA_METADATA if n not in meta]
            if missing:
                raise KeyError(
                    f"mixed-camera metadata missing from img_metas: {missing}; "
                    "the pipeline must end in KITTI360Sparse4DPackInputs"
                )
            shape = np.asarray(meta["img_shape"])
            height, width = (
                (int(shape[0]), int(shape[1])) if shape.ndim == 1
                else (int(shape[0][0]), int(shape[0][1]))
            )
            meta["pad_shape"] = (height, width)
            meta["image_wh"] = np.broadcast_to(
                np.asarray([width, height], dtype=np.float32),
                (len(meta["camera_model_id"]), 2),
            ).copy()
        return batch_input_metas


@MODELS.register_module()
class DETR3DMixedCamera(MixedProjectionMetadataMixin, DETR3D):
    """``DETR3D`` on the mixed rig; see the mixin for what changes."""


@MODELS.register_module()
class SparseBEVMixedCameraDetector(MixedProjectionMetadataMixin, SparseBEV):
    """``SparseBEV`` on the mixed rig, with the mixed projection installed."""

    def __init__(self, *args, **kwargs) -> None:
        from .sparsebev_mixed import install_mixed_projection

        install_mixed_projection()
        super().__init__(*args, **kwargs)
