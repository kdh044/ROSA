"""KITTI-360 data path matching the ICRA 2026 DAPETR protocol."""

from __future__ import annotations

import hashlib
from os import path as osp
from pathlib import Path
import sys
from typing import Optional

import mmcv
import numpy as np
from PIL import Image
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS, TRANSFORMS
from mmdet3d.structures import LiDARInstance3DBoxes

from .constants import (
    CAMERAS,
    CAMERA_MODEL_IDS,
    CLASSES,
    FISHEYE_CROPS,
    IMAGE_SIZE,
    NATIVE_IMAGE_SIZES,
    POINT_CLOUD_RANGE,
)


def _stable_int64(value: object) -> int:
    digest = hashlib.sha256(str(value).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


@DATASETS.register_module()
class KITTI360MixedCameraDataset(Det3DDataset):
    """Read the ROSA KITTI-360 info files generated from FishBEVOD tables."""

    METAINFO = {
        "classes": CLASSES,
        "version": "vkitti360-trainval",
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

    def parse_ann_info(self, info: dict) -> dict:
        if "instances" not in info and "annos" in info:
            annotations = info["annos"]
            boxes = np.asarray(
                annotations.get("gt_boxes", np.zeros((0, 9))), dtype=np.float32
            ).reshape(-1, 9)
            names = np.asarray(annotations.get("gt_names", ()))
            categories = {
                name: index for index, name in enumerate(self.metainfo["classes"])
            }
            aliases = {"trafficSign": "traffic_sign"}
            labels = np.asarray(
                [
                    categories.get(aliases.get(str(name), str(name)), -1)
                    for name in names
                ],
                dtype=np.int64,
            )
            valid = labels >= 0
            raw_ids = np.asarray(
                annotations.get("gt_ids", np.arange(len(names))), dtype=np.int64
            )
            return {
                "gt_bboxes_3d": LiDARInstance3DBoxes(
                    boxes[valid],
                    box_dim=9,
                    origin=(0.5, 0.5, 0.5),
                ).convert_to(self.box_mode_3d),
                "gt_labels_3d": labels[valid],
                "instance_inds": raw_ids[valid],
            }
        instances = info.get("instances", ())
        boxes = np.asarray(
            [item["bbox_3d"] for item in instances], dtype=np.float32
        ).reshape(-1, 7)
        labels = np.asarray(
            [item["bbox_label_3d"] for item in instances], dtype=np.int64
        )
        valid = labels >= 0
        boxes, labels = boxes[valid], labels[valid]
        if self.with_velocity:
            velocities = np.asarray(
                [item.get("velocity", (0.0, 0.0)) for item in instances],
                dtype=np.float32,
            ).reshape(-1, 2)[valid]
            velocities[~np.isfinite(velocities)] = 0.0
            boxes = np.concatenate((boxes, velocities), axis=-1)
        ids = [
            _stable_int64(item.get("instance_token", index))
            for index, item in enumerate(instances)
        ]
        instance_inds = np.asarray(ids, dtype=np.int64)[valid]
        return {
            "gt_bboxes_3d": LiDARInstance3DBoxes(
                boxes,
                box_dim=boxes.shape[-1],
                origin=(0.5, 0.5, 0.5),
            ).convert_to(self.box_mode_3d),
            "gt_labels_3d": labels,
            "instance_inds": instance_inds,
        }

    def parse_data_info(self, info: dict) -> dict:
        for image in info.get("images", {}).values():
            path = image["img_path"]
            if not osp.isabs(path):
                image["img_path"] = osp.join(self.data_root, path)
        info["sample_idx"] = info.get("sample_idx", info["token"])
        info.setdefault("scene_token", info.get("sequence", ""))
        info.setdefault("frame_idx", info.get("frame_id", 0))
        info.setdefault("prev", "")
        info.setdefault("next", "")
        info.setdefault("timestamp", int(info["frame_idx"]) * 100_000)
        info.setdefault("ego2global", np.eye(4, dtype=np.float32))
        if not self.test_mode:
            info["ann_info"] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info["eval_ann_info"] = self.parse_ann_info(info)
        return info


def _native_to_network_affine(
    camera_index: int,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    affine = np.eye(3, dtype=np.float32)
    if camera_index < 2:
        return affine, None
    crop = FISHEYE_CROPS[camera_index - 2]
    x0, y0, x1, y1 = crop
    target_w, target_h = IMAGE_SIZE
    scale_x = target_w / float(x1 - x0)
    scale_y = target_h / float(y1 - y0)
    affine = np.asarray(
        (
            (scale_x, 0.0, -scale_x * x0),
            (0.0, scale_y, -scale_y * y0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    return affine, crop


@TRANSFORMS.register_module()
class LoadKITTI360MixedCameraImages(BaseTransform):
    """Load two pinhole and two native-MEI views using DAPETR's crops."""

    def __init__(
        self,
        cameras: tuple[str, ...] = CAMERAS,
        data_root: str | None = None,
        to_float32: bool = True,
        color_type: str = "color",
    ) -> None:
        self.cameras = tuple(cameras)
        self.to_float32 = bool(to_float32)
        self.color_type = color_type
        if self.cameras != CAMERAS:
            raise ValueError(f"DAPETR protocol fixes camera order to {CAMERAS}")
        self.data_root = Path(data_root).resolve() if data_root is not None else None
        self._legacy_calibration = None
        if self.data_root is not None:
            helper_root = str(self.data_root.parent)
            if helper_root not in sys.path:
                sys.path.insert(0, helper_root)
            from kitti360_mixed_camera import (
                load_cam_to_pose,
                load_mei,
                load_perspective,
            )

            calibration_root = self.data_root / "calibration"
            cam_to_pose = load_cam_to_pose(
                calibration_root / "calib_cam_to_pose.txt"
            )
            records = {}
            for index, (key, camera) in enumerate(
                zip(CAMERAS, ("image_00", "image_01", "image_02", "image_03"))
            ):
                if index < 2:
                    calibration = load_perspective(
                        calibration_root / "perspective.txt", index
                    )
                    transform = calibration.rectification @ np.linalg.inv(
                        cam_to_pose[camera]
                    )
                    intrinsic = calibration.projection[:, :3]
                    coefficients = np.zeros(5, dtype=np.float32)
                    folder = "data_rect"
                else:
                    calibration = load_mei(calibration_root / f"{camera}.yaml")
                    transform = np.linalg.inv(cam_to_pose[camera])
                    intrinsic = np.asarray(
                        (
                            (calibration.gamma1, 0.0, calibration.u0),
                            (0.0, calibration.gamma2, calibration.v0),
                            (0.0, 0.0, 1.0),
                        )
                    )
                    coefficients = np.asarray(
                        (
                            calibration.k1,
                            calibration.k2,
                            calibration.p1,
                            calibration.p2,
                            calibration.xi,
                        )
                    )
                    folder = "data_rgb"
                records[key] = (camera, folder, transform, intrinsic, coefficients)
            self._legacy_calibration = records

    def transform(self, results: dict) -> dict:
        if "images" not in results:
            if self._legacy_calibration is None:
                raise KeyError("legacy KITTI-360 info requires data_root in loader")
            sequence = str(results["sequence"])
            frame = int(results["frame_id"])
            results["images"] = {
                key: {
                    "img_path": str(
                        self.data_root
                        / "data_2d_raw"
                        / sequence
                        / camera
                        / folder
                        / f"{frame:010d}.png"
                    ),
                    "lidar2cam": transform,
                    "camera_intrinsic": intrinsic,
                    "distortion_coefficients": coefficients,
                }
                for key, (
                    camera,
                    folder,
                    transform,
                    intrinsic,
                    coefficients,
                ) in self._legacy_calibration.items()
            }
        images = []
        paths = []
        shapes = []
        lidar2cam = []
        intrinsics = []
        distortion = []
        affines = []
        cam2ego = []
        for index, camera in enumerate(self.cameras):
            record = results["images"][camera]
            path = str(record["img_path"])
            image = mmcv.imread(path, flag=self.color_type)
            if image is None:
                raise FileNotFoundError(path)
            expected_w, expected_h = NATIVE_IMAGE_SIZES[index]
            if image.shape[:2] != (expected_h, expected_w):
                raise ValueError(
                    f"{path}: expected {expected_w}x{expected_h}, "
                    f"got {image.shape[1]}x{image.shape[0]}"
                )
            affine, crop = _native_to_network_affine(index)
            if crop is not None:
                x0, y0, x1, y1 = crop
                image = image[y0:y1, x0:x1]
                image = mmcv.imresize(image, IMAGE_SIZE)
            if self.to_float32:
                image = image.astype(np.float32)
            transform = np.asarray(record["lidar2cam"], dtype=np.float32)
            images.append(image)
            paths.append(path)
            shapes.append(image.shape)
            lidar2cam.append(transform)
            intrinsics.append(
                np.asarray(record["camera_intrinsic"], dtype=np.float32)
            )
            distortion.append(
                np.asarray(
                    record.get("distortion_coefficients", (0, 0, 0, 0, 0)),
                    dtype=np.float32,
                )
            )
            affines.append(affine)
            cam2ego.append(np.linalg.inv(transform).astype(np.float32))
        count = len(self.cameras)
        results.update(
            img=images,
            filename=paths,
            cam_names=list(self.cameras),
            cam_model=["PINHOLE", "PINHOLE", "FISHEYE_MEI", "FISHEYE_MEI"],
            camera_model_id=np.asarray(CAMERA_MODEL_IDS, dtype=np.int64),
            camera_intrinsic=np.stack(intrinsics),
            mei_distortion=np.stack(distortion),
            image_affine=np.stack(affines),
            lidar2cam=np.stack(lidar2cam),
            cam2ego=np.stack(cam2ego),
            cam2img=np.stack(intrinsics),
            lidar2img=np.tile(np.eye(4, dtype=np.float32), (count, 1, 1)),
            img_shape=shapes,
            ori_shape=shapes,
            pad_shape=shapes,
        )
        return results


@TRANSFORMS.register_module()
class ConvertLegacyKITTI360MEIOrder(BaseTransform):
    """Convert the existing camera-only infos to the public MEI convention.

    The earlier local KITTI-360 loader emits ``[xi, k1, k2, p1, p2]`` while
    the common mixed-camera projector deliberately exposes FishBEVOD's
    calibration order ``[k1, k2, p1, p2, xi]``.
    """

    def transform(self, results: dict) -> dict:
        distortion = np.asarray(results["mei_distortion"], dtype=np.float32)
        if distortion.shape[-1] != 5:
            raise ValueError("mei_distortion must end in five coefficients")
        results["mei_distortion"] = distortion[..., (1, 2, 3, 4, 0)]
        return results


@TRANSFORMS.register_module()
class DAPETRImageAugment(BaseTransform):
    """Apply the shared 1408x376 DAPETR resize/crop/flip augmentation."""

    def __init__(
        self,
        training: bool = True,
        resize_lim: tuple[float, float] = (0.94, 1.25),
        rand_flip: bool = True,
    ) -> None:
        self.training = bool(training)
        self.resize_lim = tuple(float(value) for value in resize_lim)
        self.rand_flip = bool(rand_flip)

    def _sample(self) -> tuple[float, tuple[int, int, int, int], bool]:
        width, height = IMAGE_SIZE
        resize = (
            float(np.random.uniform(*self.resize_lim)) if self.training else 1.0
        )
        new_w, new_h = int(width * resize), int(height * resize)
        if self.training:
            crop_x = int(np.random.uniform(0, max(0, new_w - width)))
            flip = self.rand_flip and bool(np.random.randint(0, 2))
        else:
            crop_x = max(0, new_w - width) // 2
            flip = False
        crop_y = new_h - height
        return resize, (crop_x, crop_y, crop_x + width, crop_y + height), flip

    def transform(self, results: dict) -> dict:
        resize, crop, flip = self._sample()
        width, height = IMAGE_SIZE
        new_size = (int(width * resize), int(height * resize))
        augmented = []
        for array in results["img"]:
            image = Image.fromarray(np.asarray(array, dtype=np.uint8))
            image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
            image = image.crop(crop)
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            output = np.asarray(image)
            if array.dtype == np.float32:
                output = output.astype(np.float32)
            augmented.append(output)

        crop_x, crop_y = crop[:2]
        augment_affine = np.asarray(
            (
                (resize, 0.0, -crop_x),
                (0.0, resize, -crop_y),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        if flip:
            flip_affine = np.asarray(
                ((-1.0, 0.0, width), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float32,
            )
            augment_affine = flip_affine @ augment_affine
        results["image_affine"] = (
            augment_affine[None] @ np.asarray(results["image_affine"])
        ).astype(np.float32)
        results["img"] = augmented
        shapes = [image.shape for image in augmented]
        results["img_shape"] = shapes
        results["pad_shape"] = shapes
        return results


@TRANSFORMS.register_module()
class DAPETRGlobalRotScale(BaseTransform):
    """Jointly augment ego boxes and the ego-to-camera projection."""

    def __init__(
        self,
        rot_range: tuple[float, float] = (-0.3925, 0.3925),
        scale_ratio_range: tuple[float, float] = (0.95, 1.05),
        training: bool = True,
    ) -> None:
        self.rot_range = tuple(float(value) for value in rot_range)
        self.scale_ratio_range = tuple(
            float(value) for value in scale_ratio_range
        )
        self.training = bool(training)

    def transform(self, results: dict) -> dict:
        if not self.training:
            return results
        angle = float(np.random.uniform(*self.rot_range))
        scale = float(np.random.uniform(*self.scale_ratio_range))
        cosine, sine = np.cos(angle), np.sin(angle)
        augmentation = np.asarray(
            (
                (scale * cosine, -scale * sine, 0.0, 0.0),
                (scale * sine, scale * cosine, 0.0, 0.0),
                (0.0, 0.0, scale, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        inverse = np.linalg.inv(augmentation).astype(np.float32)
        results["lidar2cam"] = (
            np.asarray(results["lidar2cam"], dtype=np.float32) @ inverse[None]
        )
        results["cam2ego"] = np.linalg.inv(results["lidar2cam"]).astype(
            np.float32
        )
        annotation = results.get("ann_info")
        if annotation and "gt_bboxes_3d" in annotation:
            boxes = annotation["gt_bboxes_3d"]
            boxes.rotate(torch.tensor(angle, dtype=boxes.tensor.dtype))
            boxes.scale(scale)
            if "gt_bboxes_3d" in results:
                results["gt_bboxes_3d"] = boxes
        return results


@TRANSFORMS.register_module()
class KITTI360ObjectRangeFilter(BaseTransform):
    """Filter GT with the PETR/DAPETR BEV range and preserve instance IDs."""

    def transform(self, results: dict) -> dict:
        annotation = results.get("ann_info")
        if not annotation:
            return results
        boxes = annotation["gt_bboxes_3d"]
        labels = np.asarray(annotation["gt_labels_3d"], dtype=np.int64)
        keep_tensor = boxes.in_range_bev(
            boxes.tensor.new_tensor(
                (
                    POINT_CLOUD_RANGE[0],
                    POINT_CLOUD_RANGE[1],
                    POINT_CLOUD_RANGE[3],
                    POINT_CLOUD_RANGE[4],
                )
            )
        )
        keep = keep_tensor.detach().cpu().numpy().astype(bool)
        annotation["gt_bboxes_3d"] = boxes[keep_tensor]
        annotation["gt_labels_3d"] = labels[keep]
        if "instance_inds" in annotation:
            annotation["instance_inds"] = np.asarray(
                annotation["instance_inds"]
            )[keep]
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = annotation["gt_bboxes_3d"]
        if "gt_labels_3d" in results:
            results["gt_labels_3d"] = annotation["gt_labels_3d"]
        return results


__all__ = [
    "DAPETRGlobalRotScale",
    "DAPETRImageAugment",
    "KITTI360MixedCameraDataset",
    "KITTI360ObjectRangeFilter",
    "LoadKITTI360MixedCameraImages",
]
