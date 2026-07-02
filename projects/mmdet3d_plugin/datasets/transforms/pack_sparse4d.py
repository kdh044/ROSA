# ------------------------------------------------------------------------
# PackSparse4DInputs — pack one RoboSense fisheye frame into the dict
# layout expected by Sparse4D's head (per-frame metas).
#
# Modeled on sphere_stream's PackStreamInputs but flattens to single-frame
# (Sparse4D handles temporal via its InstanceBank).
# ------------------------------------------------------------------------
from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Det3DDataSample


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x))
    return torch.as_tensor(x)


@TRANSFORMS.register_module()
class PackSparse4DInputs(BaseTransform):
    """Pack one frame into ({inputs}, Det3DDataSample) for Sparse4D.

    `inputs` keys (consumed by Sparse4D.forward):
        img             (N_cam, C, H, W)         float32
        projection_mat  (N_cam, 4, 4)            float32  (lidar2img)
        image_wh        (N_cam, 2)               float32  (W, H per cam)
        T_global        (4, 4)                   float32  (ego2global)
        T_global_inv    (4, 4)                   float32
        timestamp       scalar                   float64
        focal           (N_cam,)                 float32  (equivalent focal)
        lidar2cam       (N_cam, 4, 4)            float32  (for OCam projection)
        ocam_intrinsic  (N_cam, 3, 3)            float32  (cam2img)
        ocam_pol        (N_cam, P)               float32  (fisheye_params)
        ocam_lens_fov_deg   scalar               float    (default 205)
        kb_intrinsic    (N_cam, 3, 3)            float32  (cam2img)
        kb_distortion   (N_cam, 4)               float32  (KB k1..k4)
        kb_lens_fov_deg scalar                   float    (default ocam FOV)
        gt_depth        list[Tensor] (optional)            (per-scale, N_cam, h, w)
        gt_bboxes_3d    Tensor (N_gt, D)          float32  (training only)
        gt_labels_3d    Tensor (N_gt,)            long
        instance_id     ndarray (N_gt,)           int64    (training only)

    `data_samples` carries Det3DDataSample with gt_instances_3d for the
    mmengine evaluation hooks.
    """

    META_KEYS = (
        "cam_names", "cam_model", "img_shape", "ori_shape", "pad_shape",
        "box_mode_3d", "box_type_3d", "sample_idx", "token", "prev", "next",
        "scene_token", "frame_idx", "timestamp",
    )

    def __init__(
        self,
        ocam_lens_fov_deg: float = 205.0,
        kb_lens_fov_deg: Optional[float] = None,
        equivalent_focal: Optional[float] = None,
    ):
        """
        Args:
            ocam_lens_fov_deg: full lens FOV in degrees, used both as the
                cone mask in projection.py and to compute equivalent focal
                when ``equivalent_focal`` is None.
            equivalent_focal: focal length (px) used by DenseDepthNet's
                focal-equal rescale. If None, derived from image width
                and FOV: f_eq = W / 2 / tan(FOV / 4).
        """
        self.ocam_lens_fov_deg = float(ocam_lens_fov_deg)
        self.kb_lens_fov_deg = (
            float(kb_lens_fov_deg)
            if kb_lens_fov_deg is not None
            else float(ocam_lens_fov_deg)
        )
        self.equivalent_focal = equivalent_focal

    @staticmethod
    def _pack_img(imgs) -> torch.Tensor:
        if isinstance(imgs, np.ndarray):
            arr = imgs
        else:
            arr = np.stack(imgs, axis=0)
        if arr.ndim == 4 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    def _equivalent_focal(self, image_w: int) -> float:
        if self.equivalent_focal is not None:
            return float(self.equivalent_focal)
        # f_eq picked so the equivalent pinhole horizontal FOV equals
        # the fisheye's nominal FOV. Standard f = W/2 / tan(HFoV/2)
        # — we use HFoV/2 (i.e. FOV/4 since lens FOV is full angle ×2
        # of half-angle): a rough but stable choice.
        return image_w / 2.0 / math.tan(math.radians(self.ocam_lens_fov_deg / 4.0))

    def transform(self, results: dict) -> dict:
        inputs: dict = {}

        # ---- image ----
        if "img" in results:
            inputs["img"] = self._pack_img(results["img"])
        # ori/img_shape is list[N_cam] of (H, W, C)
        shapes = results.get("img_shape", [])
        if not shapes and "img" in inputs:
            H, W = inputs["img"].shape[-2:]
            shapes = [(H, W, 3)] * inputs["img"].shape[0]
        wh = np.asarray([[s[1], s[0]] for s in shapes], dtype=np.float32)
        inputs["image_wh"] = _to_tensor(wh)

        # ---- camera matrices ----
        if "lidar2img" in results:
            inputs["projection_mat"] = _to_tensor(results["lidar2img"]).float()
        if "lidar2cam" in results:
            inputs["lidar2cam"] = _to_tensor(results["lidar2cam"]).float()
        if "cam2img" in results:
            cam2img = _to_tensor(results["cam2img"]).float()
            inputs["ocam_intrinsic"] = cam2img
            inputs["kb_intrinsic"] = cam2img
        if "fisheye_params" in results:
            fisheye_params = _to_tensor(results["fisheye_params"]).float()
            inputs["ocam_pol"] = fisheye_params
            inputs["kb_distortion"] = fisheye_params[..., :4]

        # ---- ego pose ----
        if "ego2global" in results:
            T = _to_tensor(results["ego2global"]).float()
            inputs["T_global"] = T
            inputs["T_global_inv"] = torch.inverse(T)

        # ---- timestamp ----
        ts = float(results.get("timestamp", 0)) / 1e6  # μs → s
        inputs["timestamp"] = torch.tensor(ts, dtype=torch.float64)

        # ---- equivalent focal for DenseDepthNet ----
        if "image_wh" in inputs and inputs["image_wh"].numel() > 0:
            W_first = float(inputs["image_wh"][0, 0])
            f_eq = self._equivalent_focal(int(W_first))
            num_cams = inputs["image_wh"].shape[0]
            inputs["focal"] = torch.full((num_cams,), f_eq, dtype=torch.float32)

        inputs["ocam_lens_fov_deg"] = float(self.ocam_lens_fov_deg)
        inputs["kb_lens_fov_deg"] = float(self.kb_lens_fov_deg)

        # ---- depth GT (already prepared by MultiScaleDepthMapGenerator) ----
        if "gt_depth" in results:
            inputs["gt_depth"] = [
                _to_tensor(d).float() for d in results["gt_depth"]
            ]

        # ---- GT (training only) — kept ONLY in data_samples so the
        # default collator doesn't try to stack variable-size tensors.
        # Sparse4D._build_metas pulls them out per-sample in model.forward.
        ann = results.get("ann_info") or {}

        # ---- data sample (mmengine convention) ----
        ds = Det3DDataSample()
        gt3d = InstanceData()
        if "gt_bboxes_3d" in ann:
            gt3d.bboxes_3d = ann["gt_bboxes_3d"]
        if "gt_labels_3d" in ann:
            gt3d.labels_3d = _to_tensor(ann["gt_labels_3d"]).long()
        if "instance_inds" in ann:
            gt3d.instance_inds = _to_tensor(ann["instance_inds"]).long()
        ds.gt_instances_3d = gt3d
        if "eval_ann_info" in results:
            ds.eval_ann_info = results["eval_ann_info"]
        meta = {k: results[k] for k in self.META_KEYS if k in results}
        ds.set_metainfo(meta)

        return {"inputs": inputs, "data_samples": ds}
