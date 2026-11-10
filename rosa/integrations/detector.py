"""Multi-view detector: image backbone + FPN + ROSA head."""

from __future__ import annotations

import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from mmdet3d.registry import MODELS
from mmdet3d.structures import LiDARInstance3DBoxes


@MODELS.register_module()
class ROSADetector(BaseModel):
    """Sparse4D-style single-frame detector.

    Ground-truth boxes are given to the head with their gravity centre as the
    box position, and predictions are returned as gravity-centred
    ``LiDARInstance3DBoxes``.
    """

    def __init__(
        self,
        img_backbone: dict,
        head: dict,
        img_neck: dict | None = None,
        data_preprocessor: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None
        self.head = MODELS.build(head)

    def extract_feat(self, img):
        batch, cameras = img.shape[:2]
        feature_maps = self.img_backbone(img.flatten(end_dim=1))
        if self.img_neck is not None:
            feature_maps = self.img_neck(feature_maps)
        return [
            torch.reshape(feat, (batch, cameras) + feat.shape[1:])
            for feat in feature_maps
        ]

    @staticmethod
    def _build_metas(inputs, data_samples):
        """Flatten the collated inputs and data samples into the head's dict."""

        metas = dict(inputs)
        img_metas, gt_bboxes_3d, gt_labels_3d = [], [], []
        for sample in data_samples:
            img_metas.append(dict(sample.metainfo))
            gt = getattr(sample, "gt_instances_3d", None)
            if gt is not None and "bboxes_3d" in gt:
                boxes = gt.bboxes_3d
                tensor = boxes.tensor.clone()
                tensor[..., :3] = boxes.gravity_center.to(tensor)
                gt_bboxes_3d.append(tensor)
            if gt is not None and "labels_3d" in gt:
                gt_labels_3d.append(gt.labels_3d)
        metas["img_metas"] = img_metas
        if gt_bboxes_3d:
            metas["gt_bboxes_3d"] = gt_bboxes_3d
        if gt_labels_3d:
            metas["gt_labels_3d"] = gt_labels_3d
        return metas

    def forward(self, inputs, data_samples=None, mode="tensor", **kwargs):
        metas = self._build_metas(inputs, data_samples)
        feature_maps = self.extract_feat(metas["img"])
        model_outs = self.head(feature_maps, metas)
        if mode == "loss":
            return self.head.loss(model_outs, metas)
        if mode == "predict":
            for sample, result in zip(data_samples, self.head.post_process(model_outs)):
                pred = InstanceData()
                boxes = LiDARInstance3DBoxes(
                    result["boxes_3d"],
                    box_dim=result["boxes_3d"].shape[-1],
                    origin=(0.5, 0.5, 0.5),
                )
                pred.bboxes_3d = boxes
                pred.boxes_3d = boxes
                for key, value in result.items():
                    if key != "boxes_3d":
                        setattr(pred, key, value)
                sample.pred_instances_3d = pred
            return data_samples
        return model_outs


__all__ = ["ROSADetector"]
