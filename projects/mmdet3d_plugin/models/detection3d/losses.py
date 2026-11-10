# ------------------------------------------------------------------------
# Ported from Sparse4D (HorizonRobotics) to mmcv 2.x / mmdet3d 1.4
# Changes: LOSSES -> mmdet3d MODELS
# ------------------------------------------------------------------------
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SparseBox3DLoss(nn.Module):
    """L1 box loss; the centerness/yawness losses are built for subclasses."""

    def __init__(self, loss_box, loss_centerness=None, loss_yawness=None):
        super().__init__()
        self.loss_box = MODELS.build(loss_box)
        self.loss_cns = MODELS.build(loss_centerness)
        self.loss_yns = MODELS.build(loss_yawness)

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        box_loss = self.loss_box(box, box_target, weight=weight, avg_factor=avg_factor)
        return {f"loss_box{suffix}": box_loss}
