"""Small Sparse4D components used by the ROSA detector."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from projects.mmdet3d_plugin.core.box3d import CNS, COS_YAW, SIN_YAW, YNS
from projects.mmdet3d_plugin.models.detection3d.losses import SparseBox3DLoss




def centerness_target(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """Detached BEV-distance centerness target used by the ROSA box loss."""

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




@MODELS.register_module()
class ROSABoxLoss(SparseBox3DLoss):
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
        cns_target = centerness_target(box, box_target)
        output[f"loss_cns{suffix}"] = self.loss_cns(
            centerness,
            cns_target,
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




__all__ = [
    "ROSABoxLoss",
    "centerness_target",
]
