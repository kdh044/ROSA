# ------------------------------------------------------------------------
# Ported from Sparse4D (HorizonRobotics) to mmcv 2.x / mmdet3d 1.4
# Changes:
#   DETECTORS / build_backbone / build_head / build_neck → mmdet3d MODELS
#   BaseDetector → mmengine.model.BaseModel with forward(inputs, data_samples, mode)
#   force_fp32 / auto_fp16 dropped (fp16 handled at runner level via OptimWrapper)
# ------------------------------------------------------------------------
from inspect import signature

import torch

from mmengine.model import BaseModel
from mmdet3d.registry import MODELS

from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except Exception:
    DAF_VALID = False

__all__ = ["Sparse4D"]


@MODELS.register_module()
class Sparse4D(BaseModel):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        data_preprocessor=None,
    ):
        super(Sparse4D, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor
        )
        if pretrained is not None:
            img_backbone["pretrained"] = pretrained
        self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        else:
            self.img_neck = None
        self.head = MODELS.build(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = MODELS.build(depth_branch)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @staticmethod
    def _build_metas(inputs, data_samples=None):
        """Assemble the per-frame metas dict the Sparse4D head expects.

        ``inputs`` carries the collated per-cam tensors. ``data_samples`` is
        the list of mmengine Det3DDataSample (one per batch sample) holding
        GT / metainfo. We flatten them into the dict shape the head reads.
        """
        import numpy as np

        metas = dict(inputs)  # shallow copy; keep tensors as-is
        if data_samples is None:
            metas.setdefault("img_metas", [{}])
            return metas

        bs = len(data_samples)
        img_metas = []
        gt_bboxes_3d = []
        gt_labels_3d = []
        for i, ds in enumerate(data_samples):
            mi = dict(ds.metainfo) if ds.metainfo is not None else {}
            T_global = inputs["T_global"][i].detach().cpu().numpy()
            T_global_inv = inputs["T_global_inv"][i].detach().cpu().numpy()
            mi["T_global"] = T_global
            mi["T_global_inv"] = T_global_inv
            mi["timestamp"] = float(inputs["timestamp"][i].item()) \
                if inputs["timestamp"].dim() > 0 \
                else float(inputs["timestamp"].item())
            gt3d = getattr(ds, "gt_instances_3d", None)
            if gt3d is not None and "instance_inds" in gt3d:
                mi["instance_id"] = (
                    gt3d.instance_inds.detach().cpu().numpy().astype(np.int64)
                )
            img_metas.append(mi)
            if gt3d is not None and "bboxes_3d" in gt3d:
                bb = gt3d.bboxes_3d
                gt_bboxes_3d.append(
                    bb.tensor if hasattr(bb, "tensor") else bb
                )
            if gt3d is not None and "labels_3d" in gt3d:
                gt_labels_3d.append(gt3d.labels_3d)
        metas["img_metas"] = img_metas
        if gt_bboxes_3d:
            metas["gt_bboxes_3d"] = gt_bboxes_3d
        if gt_labels_3d:
            metas["gt_labels_3d"] = gt_labels_3d
        return metas

    def forward(self, inputs, data_samples=None, mode="tensor", **kwargs):
        """mmengine BaseModel dispatch.

        ``inputs`` (after data_preprocessor): collated per-cam tensors
        (``img``, ``projection_mat``, ``image_wh``, ``T_global``,
        ``T_global_inv``, ``timestamp``, ``focal``, ``lidar2cam``,
        ``ocam_intrinsic``, ``ocam_pol``, ``gt_depth`` (optional)).
        ``data_samples``: list[Det3DDataSample] carrying GT.
        """
        metas = self._build_metas(inputs, data_samples)
        if mode == "loss":
            return self.forward_train(metas)
        elif mode == "predict":
            return self.forward_test(metas, data_samples=data_samples)
        elif mode == "tensor":
            feature_maps = self.extract_feat(metas["img"], metas=metas)
            return self.head(feature_maps, metas)
        else:
            raise NotImplementedError(f"unknown mode={mode}")

    def forward_train(self, data):
        img = data["img"]
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, data, data_samples=None):
        img = data["img"]
        if isinstance(img, list):
            results = self.aug_test(data)
        else:
            results = self.simple_test(data)
        if data_samples is None:
            return results
        # Wire predictions back into mmengine Det3DDataSample list.
        from mmengine.structures import InstanceData
        for ds, res in zip(data_samples, results):
            pred = InstanceData()
            for k, v in res["img_bbox"].items():
                setattr(pred, k, v)
                if k == "boxes_3d":
                    setattr(pred, "bboxes_3d", v)
            ds.pred_instances_3d = pred
        return data_samples

    def simple_test(self, data):
        feature_maps = self.extract_feat(data["img"], metas=data)
        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs)
        return [{"img_bbox": result} for result in results]

    def aug_test(self, data):
        # fake test-time augmentation
        unwrapped = {}
        for key, value in data.items():
            if isinstance(value, list):
                unwrapped[key] = value[0]
            else:
                unwrapped[key] = value
        unwrapped["img"] = data["img"][0]
        return self.simple_test(unwrapped)
