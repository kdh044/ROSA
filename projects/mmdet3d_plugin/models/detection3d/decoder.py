# ------------------------------------------------------------------------
# Ported from Sparse4D (HorizonRobotics) to mmcv 2.x / mmdet3d 1.4
# Changes: BBOX_CODERS → mmdet3d TASK_UTILS
# ------------------------------------------------------------------------
import torch

from mmdet3d.registry import TASK_UTILS

from projects.mmdet3d_plugin.core.box3d import CNS, COS_YAW, H, L, SIN_YAW, VX, W, X, Y, Z


@TASK_UTILS.register_module()
class SparseBox3DDecoder(object):
    """Top-k over (query, class) scores, re-ranked by predicted centerness."""

    def __init__(self, num_output: int = 300, sorted: bool = True):
        super().__init__()
        self.num_output = num_output
        self.sorted = sorted

    def decode_box(self, box):
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        return torch.cat(
            [
                box[:, [X, Y, Z]],
                box[:, [W, L, H]].exp(),
                yaw[:, None],
                box[:, VX:],
            ],
            dim=-1,
        )

    def decode(self, cls_scores, box_preds, instance_id=None, quality=None, output_idx=-1):
        cls_scores = cls_scores[output_idx].sigmoid()
        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        cls_ids = indices % num_cls
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            cls_ids = torch.gather(cls_ids, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            query_indices = indices[i] // num_cls
            box = self.decode_box(box_preds[i, query_indices])
            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": cls_scores[i].cpu(),
                    "labels_3d": cls_ids[i].cpu(),
                    "query_indices": query_indices.cpu(),
                }
            )
            if quality is not None:
                output[-1]["cls_scores"] = cls_scores_origin[i].cpu()
        return output
