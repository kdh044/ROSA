"""Pack ``LoadKITTI360MixedCameraImages`` output for DAPETRMixedCameraHead.

ROSA's own ``PackSparse4DInputs`` flattens every camera into one ``img``
tensor plus batched top-level fields (``lidar2cam``, ``projection_mat``, ...),
which is what ``Sparse4DDataPreprocessor``/``Fisheye3DODGravityCenterSparse4D``
expect.  DAPETR's stack (``FisheyeDet3DDataPreprocessor`` /
``DAPETRFisheyeHead.native_position_embedding``) instead expects the images
under one named key (``inputs[input_key]``) and the per-camera extrinsic
nested in that same key's slot of each sample's ``metainfo``
(``data_sample.metainfo[input_key]['cam2lidar']``).  Those two detector
stacks cannot share a pack step, so this is a second, DAPETR-specific one --
reading the *same* ``LoadKITTI360MixedCameraImages`` output the ROSA arm
reads, so both arms see identical per-camera image and calibration data and
differ only in how the model consumes it.
"""

from __future__ import annotations

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmengine.structures import InstanceData


@TRANSFORMS.register_module()
class KITTI360DAPETRPackInputs(BaseTransform):
    """Pack images/calibration/GT into ``(inputs, Det3DDataSample)``.

    ``inputs[input_key]``: float32 tensor ``[C, 3, H, W]``, RGB, HWC->CHW.
    ``data_sample.metainfo[input_key]``: dict with ``cam2lidar`` ``[C,4,4]``
    (float32 numpy, matching what ``DAPETRFisheyeHead.native_position_embedding``
    reads) plus the same bookkeeping fields ``Fisheye3DODPackDetInputs`` keeps
    (``lidar2cam``, ``cam2img``, ``lidar2img``) for parity/debugging, unused by
    the head itself.
    """

    def __init__(self, input_key: str = "cam_fisheye") -> None:
        self.input_key = input_key

    def transform(self, results: dict) -> dict:
        images = results["img"]
        stacked = np.stack(images, axis=0).astype(np.float32)  # [C,H,W,3]
        image_tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()

        camera_meta = {
            "cam2lidar": np.asarray(results["cam2ego"], dtype=np.float32),
            "lidar2cam": np.asarray(results["lidar2cam"], dtype=np.float32),
            "cam2img": np.asarray(results["cam2img"], dtype=np.float32),
            "lidar2img": np.asarray(results["lidar2img"], dtype=np.float32),
        }

        # The forward-projection ports (RepDETR3D, SRCN3D) project their own
        # key points instead of lifting rays, so they need the calibration
        # ``mixed_projection`` validates -- per-camera lens model, intrinsics,
        # MEI coefficients, and the affine that maps native pixels to the
        # cropped-and-resized network input, augmentation included.  These are
        # carried at the top level of ``metainfo`` rather than inside
        # ``input_key`` because that nested slot is DAPETR's own contract with
        # its head; adding keys beside it leaves that contract untouched.
        projection_meta = {
            name: np.asarray(results[name])
            for name in (
                "camera_model_id",
                "camera_intrinsic",
                "mei_distortion",
                "image_affine",
                "lidar2cam",
            )
            if name in results
        }

        data_sample = Det3DDataSample()
        data_sample.set_metainfo(
            {
                self.input_key: camera_meta,
                **projection_meta,
                "token": results.get("token", results.get("sample_idx")),
                "sample_idx": results.get("sample_idx", results.get("token")),
                "box_type_3d": results.get("box_type_3d", LiDARInstance3DBoxes),
                "box_mode_3d": results.get("box_mode_3d"),
                "img_shape": results.get("img_shape"),
                "ori_shape": results.get("ori_shape"),
                "pad_shape": results.get("pad_shape"),
            }
        )

        # Training packs ``results['ann_info']``, val/test packs
        # ``results['eval_ann_info']`` -- matching
        # ``KITTI360MixedCameraDataset.parse_data_info`` and the contract
        # ``KITTI360ObjectRangeFilter`` already relies on upstream in this
        # same pipeline.
        annotation = results.get("ann_info") or results.get("eval_ann_info")
        if annotation is not None:
            gt_instances_3d = InstanceData()
            gt_instances_3d.bboxes_3d = annotation["gt_bboxes_3d"]
            gt_instances_3d.labels_3d = torch.as_tensor(
                annotation["gt_labels_3d"], dtype=torch.long
            )
            data_sample.gt_instances_3d = gt_instances_3d
            data_sample.eval_ann_info = {
                "gt_bboxes_3d": annotation["gt_bboxes_3d"],
                "gt_labels_3d": annotation["gt_labels_3d"],
            }

        return {
            "inputs": {self.input_key: image_tensor},
            "data_samples": data_sample,
        }
