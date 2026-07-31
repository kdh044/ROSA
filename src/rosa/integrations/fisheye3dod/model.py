"""Sparse4D wrapper for released gravity-centred Fisheye3DOD boxes."""

from __future__ import annotations

import torch
from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.models.sparse4d import Sparse4D


@MODELS.register_module()
class Fisheye3DODGravityCenterSparse4D(Sparse4D):
    """Use gravity centres in state space without adding parameters."""

    @staticmethod
    def _build_metas(inputs, data_samples=None):
        metas = Sparse4D._build_metas(inputs, data_samples)
        if data_samples is None:
            return metas
        gravity_boxes = []
        for sample in data_samples:
            ground_truth = getattr(sample, "gt_instances_3d", None)
            if ground_truth is None or "bboxes_3d" not in ground_truth:
                continue
            boxes = ground_truth.bboxes_3d
            tensor = boxes.tensor.clone()
            tensor[..., :3] = boxes.gravity_center.to(
                device=tensor.device, dtype=tensor.dtype
            )
            gravity_boxes.append(tensor)
        if gravity_boxes:
            metas["gt_bboxes_3d"] = gravity_boxes
        return metas

    def simple_test(self, data):
        outputs = super().simple_test(data)
        for output in outputs:
            prediction = output["img_bbox"]
            boxes = prediction["boxes_3d"]
            if isinstance(boxes, LiDARInstance3DBoxes):
                continue
            if not isinstance(boxes, torch.Tensor):
                raise TypeError(f"decoded boxes must be a tensor, got {type(boxes)}")
            prediction["boxes_3d"] = LiDARInstance3DBoxes(
                boxes,
                box_dim=boxes.shape[-1],
                origin=(0.5, 0.5, 0.5),
            )
        return outputs


__all__ = ["Fisheye3DODGravityCenterSparse4D"]
