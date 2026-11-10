# ------------------------------------------------------------------------
# Pack one multi-camera frame into the (inputs, Det3DDataSample) layout
# read by the ROSA detector.
# ------------------------------------------------------------------------
from __future__ import annotations

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
    """Pack one frame into ``{"inputs", "data_samples"}``.

    ``inputs``:
        img             (N_cam, C, H, W)  float32
        image_wh        (N_cam, 2)        float32  (W, H per camera)
        lidar2cam       (N_cam, 4, 4)     float32
        ocam_intrinsic  (N_cam, 3, 3)     float32  (OCam affine, Fisheye3DOD)
        ocam_pol        (N_cam, P)        float32  (OCam polynomial, Fisheye3DOD)
        ocam_lens_fov_deg  float
        camera_model_id, camera_intrinsic, mei_distortion, image_affine
                                          (KITTI-360 pinhole/MEI rig)

    Ground truth stays in the ``Det3DDataSample`` so that the default
    collate function does not stack variable-size tensors.
    """

    META_KEYS = (
        "cam_names", "cam_model", "img_shape", "ori_shape", "pad_shape",
        "box_mode_3d", "box_type_3d", "sample_idx", "token", "prev", "next",
        "scene_token", "frame_idx", "timestamp",
    )

    def __init__(self, ocam_lens_fov_deg: float = 205.0):
        self.ocam_lens_fov_deg = float(ocam_lens_fov_deg)

    @staticmethod
    def _pack_img(imgs) -> torch.Tensor:
        arr = imgs if isinstance(imgs, np.ndarray) else np.stack(imgs, axis=0)
        if arr.ndim == 4 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (0, 3, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    def transform(self, results: dict) -> dict:
        inputs: dict = {}
        if "img" in results:
            inputs["img"] = self._pack_img(results["img"])
        shapes = results.get("img_shape", [])
        if not shapes and "img" in inputs:
            H, W = inputs["img"].shape[-2:]
            shapes = [(H, W, 3)] * inputs["img"].shape[0]
        inputs["image_wh"] = _to_tensor(
            np.asarray([[s[1], s[0]] for s in shapes], dtype=np.float32)
        )
        if "lidar2cam" in results:
            inputs["lidar2cam"] = _to_tensor(results["lidar2cam"]).float()
        if "cam2img" in results:
            inputs["ocam_intrinsic"] = _to_tensor(results["cam2img"]).float()
        if "fisheye_params" in results:
            inputs["ocam_pol"] = _to_tensor(results["fisheye_params"]).float()
        for name, dtype in (
            ("camera_model_id", torch.long),
            ("camera_intrinsic", torch.float32),
            ("mei_distortion", torch.float32),
            ("image_affine", torch.float32),
        ):
            if name in results:
                inputs[name] = _to_tensor(results[name]).to(dtype=dtype)
        inputs["ocam_lens_fov_deg"] = self.ocam_lens_fov_deg

        ann = results.get("ann_info") or {}
        sample = Det3DDataSample()
        gt = InstanceData()
        if "gt_bboxes_3d" in ann:
            gt.bboxes_3d = ann["gt_bboxes_3d"]
        if "gt_labels_3d" in ann:
            gt.labels_3d = _to_tensor(ann["gt_labels_3d"]).long()
        sample.gt_instances_3d = gt
        if "eval_ann_info" in results:
            sample.eval_ann_info = results["eval_ann_info"]
        sample.set_metainfo({k: results[k] for k in self.META_KEYS if k in results})
        return {"inputs": inputs, "data_samples": sample}
