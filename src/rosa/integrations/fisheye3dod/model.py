"""Sparse4D wrapper for released gravity-centred Fisheye3DOD boxes."""

from __future__ import annotations

import operator

import torch
from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.data_preprocessor.sparse4d_preprocessor import (
    Sparse4DDataPreprocessor,
)
from projects.mmdet3d_plugin.models.sparse4d import Sparse4D

from rosa.integrations.sparse4d import ROSAOfficialResidualNativeBaselineHead


@MODELS.register_module()
class Fisheye3DODSingleFrameDataPreprocessor(Sparse4DDataPreprocessor):
    """Keep temporal-only metadata on CPU during single-frame prediction.

    Sparse4D's common preprocessor otherwise copies these tensors to CUDA and
    ``_build_metas`` immediately materializes them back on the host.  They
    remain unchanged during training, and the paired model rejects this
    preprocessor when temporal instance slots are enabled.
    """

    _CPU_PREDICTION_KEYS = ("T_global", "T_global_inv", "timestamp")

    def forward(self, data, training: bool = False):
        if isinstance(data, list):
            if training:
                return super().forward(data, training=True)
            raise ValueError(
                "single-frame CPU metadata prediction does not support TTA/list input"
            )
        if training or not isinstance(data, dict):
            return super().forward(data, training=training)
        inputs = data.get("inputs")
        if not isinstance(inputs, dict):
            return super().forward(data, training=training)

        host_metadata = {
            name: inputs[name] for name in self._CPU_PREDICTION_KEYS if name in inputs
        }
        if not host_metadata:
            return super().forward(data, training=training)
        for name, value in host_metadata.items():
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ValueError(
                    f"single-frame prediction metadata {name} must enter on CPU"
                )

        device_data = dict(data)
        device_inputs = dict(inputs)
        for name in host_metadata:
            device_inputs.pop(name)
        device_data["inputs"] = device_inputs
        processed = super().forward(device_data, training=training)
        processed_inputs = processed["inputs"]
        restored_inputs = {
            name: (
                host_metadata[name] if name in host_metadata else processed_inputs[name]
            )
            for name in inputs
        }
        restored_inputs.update(
            (name, value)
            for name, value in processed_inputs.items()
            if name not in restored_inputs
        )
        processed["inputs"] = restored_inputs
        return processed


def _require_temporal_free_metadata_path(head) -> None:
    if type(head) is not ROSAOfficialResidualNativeBaselineHead:
        raise TypeError(
            "single-frame CPU metadata is sealed to the exact "
            "ROSAOfficialResidualNativeBaselineHead"
        )
    instance_bank = getattr(head, "instance_bank", None)
    raw_count = getattr(instance_bank, "num_temp_instances", None)
    if isinstance(raw_count, bool):
        raise TypeError("num_temp_instances must be an exact integer zero")
    try:
        count = operator.index(raw_count)
    except TypeError as error:
        raise TypeError("num_temp_instances must be an exact integer zero") from error
    if count != 0:
        raise ValueError(
            "Fisheye3DODSingleFrameDataPreprocessor requires num_temp_instances=0"
        )
    operation_order = getattr(head, "operation_order", None)
    if not isinstance(operation_order, (list, tuple)):
        raise TypeError("the audited C32 head requires an operation_order sequence")
    if "temp_gnn" in operation_order:
        raise ValueError("single-frame CPU metadata forbids temp_gnn operations")


@MODELS.register_module()
class Fisheye3DODGravityCenterSparse4D(Sparse4D):
    """Use gravity centres in state space without adding parameters."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(
            self.data_preprocessor,
            Fisheye3DODSingleFrameDataPreprocessor,
        ):
            _require_temporal_free_metadata_path(self.head)

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


__all__ = [
    "Fisheye3DODGravityCenterSparse4D",
    "Fisheye3DODSingleFrameDataPreprocessor",
]
