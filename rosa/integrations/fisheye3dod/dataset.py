"""Released-schema Fisheye3DOD dataset and native image transforms."""

from __future__ import annotations

import hashlib
import os
import os.path as osp
from pathlib import Path
import sys
from typing import Optional

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS, TRANSFORMS
from mmdet3d.structures import LiDARInstance3DBoxes

from .calibration import convert_lidar2cam, convert_ocamcamera
from .constants import (
    CAMERAS,
    CAMERA_IMAGE_SIZE,
    CLASSES,
    DEFAULT_OFFICIAL_REPO,
    OFFICIAL_CLASS_RANGES,
    OFFICIAL_POINT_CLOUD_RANGE,
)

# Converter pickles may refer to ``utils.fisheye3dod_calib``.  Make the
# released repository root available before Det3DDataset unpickles an info.
_official_root = Path(
    os.environ.get("FISHEYE3DOD_OFFICIAL_ROOT", DEFAULT_OFFICIAL_REPO)
).resolve()
if str(_official_root) not in sys.path:
    sys.path.insert(0, str(_official_root))


def _stable_instance_id(scene: str, raw_id: object) -> int:
    payload = f"{scene}:{raw_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


@DATASETS.register_module()
class Fisheye3DODSparse4DDataset(Det3DDataset):
    """Read released ``fisheye3dod_infos_*.pkl`` files without rewriting."""

    METAINFO = {
        "classes": (
            "Car",
            "Van",
            "Truck",
            "Bus",
            "Pedestrian",
            "Cyclist",
        ),
        "version": "2hz",
    }

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        pipeline: list,
        test_mode: bool = False,
        metainfo: Optional[dict] = None,
        with_velocity: bool = True,
        **kwargs,
    ) -> None:
        self.with_velocity = bool(with_velocity)
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            metainfo=metainfo,
            modality=dict(use_camera=True, use_lidar=False),
            box_type_3d="LiDAR",
            **kwargs,
        )

    def parse_anno_info(self, info: dict) -> dict:
        annotations = info.get("annos", {})
        boxes = np.asarray(
            annotations.get("gt_boxes", np.zeros((0, 7))),
            dtype=np.float32,
        ).reshape(-1, 7)
        names = np.asarray(annotations.get("gt_names", ()))
        category = self.metainfo.get(
            "categories",
            {name: index for index, name in enumerate(self.METAINFO["classes"])},
        )
        labels = np.asarray(
            [category.get(str(name), -1) for name in names],
            dtype=np.int64,
        )
        valid = labels >= 0
        boxes, labels = boxes[valid], labels[valid]
        if self.with_velocity:
            boxes = np.concatenate(
                (
                    boxes,
                    np.zeros((len(boxes), 2), dtype=np.float32),
                ),
                axis=1,
            )
        scene = info.get("metainfo", {}).get("scene_name", "")
        raw_ids = np.asarray(annotations.get("gt_ids", np.arange(len(valid))))
        raw_ids = (
            raw_ids[valid] if len(raw_ids) == len(valid) else np.arange(len(boxes))
        )
        instance_inds = np.asarray(
            [_stable_instance_id(scene, raw_id) for raw_id in raw_ids],
            dtype=np.int64,
        )
        return {
            "gt_bboxes_3d": LiDARInstance3DBoxes(
                boxes,
                box_dim=boxes.shape[-1] if boxes.ndim == 2 else 9,
                origin=(0.5, 0.5, 0.5),
            ).convert_to(self.box_mode_3d),
            "gt_labels_3d": labels,
            "instance_inds": instance_inds,
        }

    def parse_data_info(self, info: dict) -> dict:
        meta = info.get("metainfo", {})
        scene = str(meta.get("scene_name", ""))
        frame_id = str(meta.get("frame_id", "0"))
        for cameras in info["cam_info"].values():
            for camera_info in cameras.values():
                path = camera_info.get("cam_path")
                if path is not None and not osp.isabs(str(path)):
                    camera_info["cam_path"] = osp.join(self.data_root, str(path))
        info["sample_idx"] = info["token"]
        info["scene_token"] = scene
        info["frame_idx"] = int(frame_id)
        info["prev"] = str(meta.get("prev_id", ""))
        info["next"] = str(meta.get("next_id", ""))
        info["timestamp"] = int(frame_id) * 100_000
        info["ego2global"] = np.eye(4, dtype=np.float32)
        if not self.test_mode:
            info["ann_info"] = self.parse_anno_info(info)
        if self.test_mode and self.load_eval_anns:
            info["eval_ann_info"] = self.parse_anno_info(info)
        return info


@TRANSFORMS.register_module()
class OfficialFisheye3DODRangeFilter(BaseTransform):
    """Released BEV range filter with instance IDs kept aligned."""

    def transform(self, results: dict) -> dict:
        annotation = results.get("ann_info")
        if not annotation:
            return results
        boxes = annotation["gt_bboxes_3d"]
        labels = np.asarray(annotation["gt_labels_3d"], dtype=np.int64)
        keep_tensor = boxes.in_range_bev(
            boxes.tensor.new_tensor(
                (
                    OFFICIAL_POINT_CLOUD_RANGE[0],
                    OFFICIAL_POINT_CLOUD_RANGE[1],
                    OFFICIAL_POINT_CLOUD_RANGE[3],
                    OFFICIAL_POINT_CLOUD_RANGE[4],
                )
            )
        )
        keep = keep_tensor.detach().cpu().numpy().astype(bool)
        annotation["gt_bboxes_3d"] = boxes[keep_tensor]
        annotation["gt_labels_3d"] = labels[keep]
        if "instance_inds" in annotation:
            annotation["instance_inds"] = np.asarray(annotation["instance_inds"])[keep]
        return results


@TRANSFORMS.register_module()
class Fisheye3DODEvaluatorRangeFilter(BaseTransform):
    """Apply the evaluator's exact class-conditional radial GT range."""

    class_ranges = tuple(float(OFFICIAL_CLASS_RANGES[name]) for name in CLASSES)

    def transform(self, results: dict) -> dict:
        annotation = results.get("ann_info")
        if not annotation:
            return results
        boxes = annotation["gt_bboxes_3d"]
        labels = np.asarray(annotation["gt_labels_3d"], dtype=np.int64)
        if len(boxes) != len(labels):
            raise ValueError("Fisheye3DOD boxes and labels are misaligned")
        if len(labels) == 0:
            return results
        if labels.min() < 0 or labels.max() >= len(self.class_ranges):
            raise ValueError("Fisheye3DOD evaluator-range label is invalid")
        label_tensor = boxes.tensor.new_tensor(labels, dtype=torch.long)
        limits = boxes.tensor.new_tensor(self.class_ranges)[label_tensor]
        radius = torch.linalg.vector_norm(boxes.tensor[:, :2], dim=-1)
        # The released evaluator uses strict `ego_dist < class_range`.
        keep_tensor = radius < limits
        keep = keep_tensor.detach().cpu().numpy().astype(bool)
        annotation["gt_bboxes_3d"] = boxes[keep_tensor]
        annotation["gt_labels_3d"] = labels[keep]
        if "instance_inds" in annotation:
            annotation["instance_inds"] = np.asarray(
                annotation["instance_inds"]
            )[keep]
        return results


@TRANSFORMS.register_module()
class LoadOfficialFisheye3DODForSparse4D(BaseTransform):
    """Load four native 800×800 OCam images in the fixed released order."""

    _EMIT_CAM2EGO = True

    def __init__(
        self,
        cam_names: tuple[str, ...] = CAMERAS,
        to_float32: bool = True,
        color_type: str = "color",
    ) -> None:
        self.cam_names = tuple(cam_names)
        self.to_float32 = bool(to_float32)
        self.color_type = color_type
        if self.cam_names != CAMERAS:
            raise ValueError(f"paired protocol fixes camera order to {CAMERAS}")

    def transform(self, results: dict) -> dict:
        camera_infos = results["cam_info"]["cam_fisheye"]
        images, lidar2cam, cam2img = [], [], []
        polynomials, shapes, paths, fovs = [], [], [], []
        for camera in self.cam_names:
            info = camera_infos[camera]
            path = str(info["cam_path"])
            image = mmcv.imread(path, flag=self.color_type)
            if image is None:
                raise FileNotFoundError(path)
            if image.shape[:2] != (
                CAMERA_IMAGE_SIZE[1],
                CAMERA_IMAGE_SIZE[0],
            ):
                raise ValueError(
                    f"{path}: expected native 800x800, " f"got {image.shape[:2]}"
                )
            if self.to_float32:
                image = image.astype(np.float32)
            calibration = convert_ocamcamera(info["cam2img"])
            images.append(image)
            shapes.append(image.shape)
            paths.append(path)
            lidar2cam.append(convert_lidar2cam(info["lidar2cam"]))
            cam2img.append(calibration.intrinsic)
            polynomials.append(calibration.polynomial)
            fovs.append(calibration.lens_fov_deg)
        if len(set(fovs)) != 1:
            raise ValueError(f"camera FOV mismatch: {fovs}")
        lengths = {len(value) for value in polynomials}
        if len(lengths) != 1:
            raise ValueError(f"OCam inverse polynomial lengths differ: {lengths}")
        count = len(self.cam_names)
        results.update(
            img=images,
            filename=paths,
            cam_names=list(self.cam_names),
            cam_model=["FISHEYE_OCAMCALIB"] * count,
            cam2img=np.stack(cam2img).astype(np.float32),
            lidar2cam=np.stack(lidar2cam).astype(np.float32),
            fisheye_params=np.stack(polynomials).astype(np.float32),
            lidar2img=np.tile(np.eye(4, dtype=np.float32), (count, 1, 1)),
            img_shape=shapes,
            ori_shape=shapes,
            pad_shape=shapes,
            ocam_lens_fov_deg=float(fovs[0]),
        )
        if self._EMIT_CAM2EGO:
            results["cam2ego"] = np.linalg.inv(results["lidar2cam"]).astype(
                np.float32
            )
        return results


@TRANSFORMS.register_module()
class LoadOfficialFisheye3DODLeanForSparse4D(
    LoadOfficialFisheye3DODForSparse4D
):
    """Exact native loader without the unused camera-pose inverse.

    This variant is only for pipelines that do not consume ``cam2ego``.  Keep
    the default loader for the optional ego D4/C4 augmentation branches.
    """

    _EMIT_CAM2EGO = False


__all__ = [
    "Fisheye3DODEvaluatorRangeFilter",
    "Fisheye3DODSparse4DDataset",
    "LoadOfficialFisheye3DODLeanForSparse4D",
    "LoadOfficialFisheye3DODForSparse4D",
    "OfficialFisheye3DODRangeFilter",
]
