"""Small Sparse4D components retained by the current V49 detector.

These are ordinary detector utilities, not the retired ROSA sidecar/head.
Keeping them here makes the active dependency explicit and prevents the
current model from importing the old refinement implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from projects.mmdet3d_plugin.core.box3d import CNS, COS_YAW, SIN_YAW, YNS
from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import (
    SparseBox3DKeyPointsGenerator,
)
from projects.mmdet3d_plugin.models.detection3d.losses import SparseBox3DLoss
from projects.mmdet3d_plugin.models.sparse4d_head import Sparse4DHead


def _normalized_yaw_geometry(anchor: Tensor) -> Tensor:
    """Normalize the yaw vector only in the keypoint-geometry copy."""

    yaw = anchor[..., [SIN_YAW, COS_YAW]]
    finite = torch.isfinite(yaw).all(dim=-1, keepdim=True)
    safe_yaw = torch.where(finite, yaw, torch.zeros_like(yaw))
    norm = torch.linalg.vector_norm(safe_yaw, dim=-1, keepdim=True)
    fallback = torch.zeros_like(safe_yaw)
    fallback[..., 1] = 1.0
    normalized = torch.where(
        finite & norm.gt(1e-6),
        safe_yaw / norm.clamp_min(1e-6),
        fallback,
    )
    geometry_anchor = anchor.clone()
    geometry_anchor[..., SIN_YAW] = normalized[..., 0]
    geometry_anchor[..., COS_YAW] = normalized[..., 1]
    return geometry_anchor


@MODELS.register_module()
class NormalizedSparseBoxKeyPointsGenerator(SparseBox3DKeyPointsGenerator):
    """Generate gravity-centred keypoints from a unit yaw direction."""

    def __init__(self, *args, anchor_z_is_bottom: bool = False, **kwargs) -> None:
        super().__init__(
            *args,
            anchor_z_is_bottom=anchor_z_is_bottom,
            **kwargs,
        )


@MODELS.register_module()
class CorrectedSparse4DHead(Sparse4DHead):
    """Sparse4D head with the released MMCV attention residual restored."""

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        layer = self.layers[index]
        if not isinstance(layer, torch.nn.MultiheadAttention):
            raise TypeError("corrected Sparse4D requires torch MultiheadAttention")
        if not layer.batch_first:
            raise RuntimeError("corrected Sparse4D requires batch_first=True")
        if query_pos is not None or key_pos is not None:
            raise RuntimeError("non-decoupled positional attention is unsupported")
        if key is None:
            key = query
        if value is None:
            value = key
        allowed = {"attn_mask", "key_padding_mask"}
        unexpected = sorted(set(kwargs).difference(allowed))
        if unexpected:
            raise TypeError(f"unsupported attention arguments: {unexpected}")
        attention, _ = layer(
            query,
            key,
            value,
            attn_mask=kwargs.get("attn_mask"),
            key_padding_mask=kwargs.get("key_padding_mask"),
        )
        attention = F.dropout(
            attention,
            p=float(layer.dropout),
            training=self.training,
        )
        return self.fc_after(query + attention)

    def forward(self, anchor: Tensor, *args, **kwargs):
        return super().forward(
            _normalized_yaw_geometry(anchor),
            *args,
            **kwargs,
        )


def evaluator_aligned_centerness_target(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """Detached BEV-distance target used by the trained V49 box loss."""

    if (
        prediction.shape != target.shape
        or prediction.shape[-1] < 2
        or not prediction.is_floating_point()
    ):
        raise ValueError("centerness prediction/target boxes must be aligned")
    distance = torch.linalg.vector_norm(
        target[..., :2] - prediction[..., :2],
        dim=-1,
    )
    return torch.exp(-distance).detach()


def evaluator_aligned_scale_iou_error(
    prediction_log_size: Tensor,
    target_log_size: Tensor,
) -> Tensor:
    """Return the official aligned-size error, ``1 - scale IoU``."""

    if (
        prediction_log_size.shape != target_log_size.shape
        or prediction_log_size.shape[-1] != 3
        or not prediction_log_size.is_floating_point()
        or not target_log_size.is_floating_point()
    ):
        raise ValueError("scale-IoU log sizes must be aligned [..., 3] tensors")
    prediction_size = prediction_log_size.exp()
    target_size = target_log_size.exp()
    intersection = torch.minimum(prediction_size, target_size).prod(dim=-1)
    union = (
        prediction_size.prod(dim=-1)
        + target_size.prod(dim=-1)
        - intersection
    )
    return 1.0 - intersection / union.clamp_min(torch.finfo(union.dtype).eps)


@MODELS.register_module()
class EvaluatorAlignedSparseBox3DLoss(SparseBox3DLoss):
    """Sparse4D box loss with the evaluator-aligned BEV centerness target."""

    def forward(
        self,
        box: Tensor,
        box_target: Tensor,
        weight: Tensor | None = None,
        avg_factor=None,
        suffix: str = "",
        quality: Tensor | None = None,
        cls_target: Tensor | None = None,
        **kwargs,
    ):
        output = super().forward(
            box,
            box_target,
            weight=weight,
            avg_factor=avg_factor,
            suffix=suffix,
            quality=None,
            cls_target=cls_target,
            **kwargs,
        )
        if quality is None:
            return output
        centerness = quality[..., CNS]
        centerness_target = evaluator_aligned_centerness_target(box, box_target)
        output[f"loss_cns{suffix}"] = self.loss_cns(
            centerness,
            centerness_target,
            avg_factor=avg_factor,
        )
        yawness = quality[..., YNS].sigmoid()
        yawness_target = (
            F.cosine_similarity(
                box_target[..., [SIN_YAW, COS_YAW]],
                box[..., [SIN_YAW, COS_YAW]],
                dim=-1,
            )
            > 0
        ).to(yawness.dtype)
        output[f"loss_yns{suffix}"] = self.loss_yns(
            yawness,
            yawness_target,
            avg_factor=avg_factor,
        )
        return output


@MODELS.register_module()
class EvaluatorAlignedScaleIoUSparseBox3DLoss(EvaluatorAlignedSparseBox3DLoss):
    """Replace only the log-size L1 terms with aligned scale-IoU error."""

    def forward(
        self,
        box: Tensor,
        box_target: Tensor,
        weight: Tensor | None = None,
        avg_factor=None,
        suffix: str = "",
        **kwargs,
    ):
        if box.shape != box_target.shape or box.shape[-1] < 6:
            raise ValueError("scale-IoU box prediction and target must be aligned")
        if weight is None:
            non_size_weight = torch.ones_like(box)
            size_weight = torch.ones_like(box[..., 0])
        else:
            if weight.shape != box.shape:
                raise ValueError("scale-IoU regression weights must match boxes")
            non_size_weight = weight.clone()
            size_weight = weight[..., 3:6].mean(dim=-1)
        non_size_weight[..., 3:6] = 0
        output = super().forward(
            box,
            box_target,
            weight=non_size_weight,
            avg_factor=avg_factor,
            suffix=suffix,
            **kwargs,
        )
        per_box = evaluator_aligned_scale_iou_error(
            box[..., 3:6],
            box_target[..., 3:6],
        )
        weighted = per_box * size_weight.to(dtype=per_box.dtype)
        if avg_factor is None:
            scale_loss = weighted.mean()
        else:
            denominator = torch.as_tensor(
                avg_factor,
                dtype=weighted.dtype,
                device=weighted.device,
            ).clamp_min(1)
            scale_loss = weighted.sum() / denominator
        output[f"loss_box{suffix}"] = (
            output[f"loss_box{suffix}"]
            + scale_loss * float(self.loss_box.loss_weight)
        )
        return output


__all__ = [
    "CorrectedSparse4DHead",
    "EvaluatorAlignedSparseBox3DLoss",
    "EvaluatorAlignedScaleIoUSparseBox3DLoss",
    "NormalizedSparseBoxKeyPointsGenerator",
    "evaluator_aligned_centerness_target",
    "evaluator_aligned_scale_iou_error",
]
