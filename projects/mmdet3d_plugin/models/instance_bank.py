# ------------------------------------------------------------------------
# Ported from Sparse4D (HorizonRobotics) to mmcv 2.x / mmdet3d 1.4.
# Single-frame only: the temporal cache of the original is not used.
# ------------------------------------------------------------------------
import numpy as np
import torch
from torch import nn
from mmdet3d.registry import MODELS

__all__ = ["InstanceBank"]


@MODELS.register_module()
class InstanceBank(nn.Module):
    """Learnable anchors (k-means initialized) and query features."""

    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_grad=True,
        feat_grad=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def get(self, batch_size):
        instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1))
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        return instance_feature, anchor
