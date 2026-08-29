"""AAE-free KITTI-360 nuScenes-style metric used by the ICRA DAPETR paper."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import mmengine
import numpy as np
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from nuscenes.eval.common.data_classes import EvalBox, EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from pyquaternion import Quaternion

from .constants import CLASSES


CLASS_RANGE = {
    "car": 50.0,
    "truck": 50.0,
    "trailer": 50.0,
    "bus": 50.0,
    "bicycle": 40.0,
    "motorcycle": 40.0,
    "pedestrian": 40.0,
    "pole": 30.0,
    "object": 50.0,
    "traffic_sign": 40.0,
}
DISTANCE_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


class KITTI360DetectionBox(EvalBox):
    """Detection box without nuScenes' hard-coded category assertion."""

    def __init__(
        self,
        *,
        sample_token: str,
        translation,
        size,
        rotation,
        velocity,
        ego_translation,
        num_pts: int,
        detection_name: str,
        detection_score: float,
        attribute_name: str = "",
    ) -> None:
        super().__init__(
            sample_token=sample_token,
            translation=translation,
            size=size,
            rotation=rotation,
            velocity=velocity,
            ego_translation=ego_translation,
            num_pts=num_pts,
        )
        if not detection_name:
            raise ValueError("KITTI-360 detection name cannot be empty")
        self.detection_name = detection_name
        self.detection_score = float(detection_score)
        self.attribute_name = attribute_name

    def serialize(self) -> dict:
        return {
            "sample_token": self.sample_token,
            "translation": list(self.translation),
            "size": list(self.size),
            "rotation": list(self.rotation),
            "velocity": list(self.velocity),
            "ego_translation": list(self.ego_translation),
            "num_pts": self.num_pts,
            "detection_name": self.detection_name,
            "detection_score": self.detection_score,
            "attribute_name": self.attribute_name,
        }

    @classmethod
    def deserialize(cls, content: dict):
        return cls(**content)


def _make_box(
    sample_token: str,
    tensor,
    name: str,
    score: float,
    num_pts: int,
) -> KITTI360DetectionBox:
    values = np.asarray(tensor, dtype=np.float64)
    center = values[:3].tolist()
    length, width, height = values[3:6]
    yaw = float(values[6])
    velocity = values[7:9].tolist() if len(values) >= 9 else [0.0, 0.0]
    return KITTI360DetectionBox(
        sample_token=str(sample_token),
        translation=center,
        size=[float(width), float(length), float(height)],
        rotation=Quaternion(axis=[0, 0, 1], angle=yaw).elements.tolist(),
        velocity=velocity,
        ego_translation=center,
        num_pts=int(num_pts),
        detection_name=name,
        detection_score=float(score),
    )


def _filter(
    boxes: EvalBoxes, ground_truth: bool, class_range: dict[str, float]
) -> EvalBoxes:
    output = EvalBoxes()
    for token in boxes.sample_tokens:
        keep = []
        for box in boxes[token]:
            if box.ego_dist >= class_range[box.detection_name]:
                continue
            if ground_truth and box.num_pts <= 0:
                continue
            keep.append(box)
        output.add_boxes(token, keep)
    return output


@METRICS.register_module()
class KITTI360DAPETRMetric(BaseMetric):
    """nuScenes AP/TP metrics with AAE removed and NDS rebalanced.

    NDS = (5*mAP + ATE_score + ASE_score + AOE_score + AVE_score) / 9.
    """

    default_prefix = "kitti360_bev"

    def __init__(
        self,
        ann_file: str,
        classes: Sequence[str] = CLASSES,
        class_range: dict[str, float] | None = None,
        collect_device: str = "cpu",
        prefix: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.classes = tuple(classes)
        self.class_range = dict(class_range or CLASS_RANGE)
        missing = set(self.classes) - set(self.class_range)
        if missing:
            raise ValueError(f"class_range missing KITTI-360 classes: {sorted(missing)}")

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            prediction = getattr(data_sample, "pred_instances_3d", None)
            if prediction is None and isinstance(data_sample, dict):
                prediction = data_sample.get("pred_instances_3d")
            if prediction is None:
                raise KeyError("prediction has no pred_instances_3d")
            token = (
                data_sample.metainfo.get("token", data_sample.metainfo.get("sample_idx"))
                if hasattr(data_sample, "metainfo")
                else data_sample.get("token", data_sample.get("sample_idx"))
            )
            # ``pred_instances_3d`` normally reaches here as an ``InstanceData``
            # (attribute access), but under some launch paths (observed with
            # 2-GPU DDP here) it arrives as the plain dict the detector's
            # ``results["img_bbox"]`` was built from -- same underlying
            # content, different container.  Accept both rather than assume
            # one, matching the normalisation already used for the
            # Fisheye3DOD arms throughout this project.
            if isinstance(prediction, dict):
                boxes = prediction.get("bboxes_3d", prediction.get("boxes_3d"))
                labels = prediction.get("labels_3d")
                scores = prediction.get("scores_3d")
            else:
                boxes = getattr(prediction, "bboxes_3d", None)
                if boxes is None:
                    boxes = prediction.boxes_3d
                labels = prediction.labels_3d
                scores = prediction.scores_3d
            if boxes is None:
                raise KeyError("prediction has neither bboxes_3d nor boxes_3d")
            boxes = getattr(boxes, "tensor", boxes)
            boxes = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
            labels = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
            scores = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            self.results.append(
                {
                    "token": str(token),
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores,
                }
            )

    def compute_metrics(self, results: list[dict]) -> dict[str, float]:
        data = mmengine.load(self.ann_file)
        infos = {str(info["token"]): info for info in data["data_list"]}
        predictions = {str(result["token"]): result for result in results}
        gt_boxes, pred_boxes = EvalBoxes(), EvalBoxes()
        for token, info in infos.items():
            gt = []
            if "instances" in info:
                for instance in info.get("instances", ()):
                    label = int(instance["bbox_label_3d"])
                    if label < 0:
                        continue
                    values = list(instance["bbox_3d"]) + list(
                        instance.get("velocity", (0.0, 0.0))
                    )
                    gt.append(
                        _make_box(
                            token,
                            values,
                            self.classes[label],
                            -1.0,
                            int(instance.get("num_lidar_pts", 1)),
                        )
                    )
            else:
                annotations = info.get("annos", {})
                categories = {name: index for index, name in enumerate(self.classes)}
                aliases = {"trafficSign": "traffic_sign"}
                for values, name in zip(
                    annotations.get("gt_boxes", ()),
                    annotations.get("gt_names", ()),
                ):
                    label = categories.get(aliases.get(str(name), str(name)), -1)
                    if 0 <= label < len(self.classes):
                        gt.append(
                            _make_box(
                                token,
                                values,
                                self.classes[label],
                                -1.0,
                                1,
                            )
                        )
            gt_boxes.add_boxes(token, gt)
            result = predictions.get(token)
            pred = []
            if result is not None:
                order = np.argsort(result["scores"])[::-1][:300]
                for index in order:
                    label = int(result["labels"][index])
                    if 0 <= label < len(self.classes):
                        pred.append(
                            _make_box(
                                token,
                                result["boxes"][index],
                                self.classes[label],
                                float(result["scores"][index]),
                                -1,
                            )
                        )
            pred_boxes.add_boxes(token, pred)
        gt_boxes = _filter(gt_boxes, ground_truth=True, class_range=self.class_range)
        pred_boxes = _filter(
            pred_boxes, ground_truth=False, class_range=self.class_range
        )

        aps = defaultdict(dict)
        errors = defaultdict(dict)
        for name in self.classes:
            metric_data = {}
            for threshold in DISTANCE_THRESHOLDS:
                data_at_threshold = accumulate(
                    gt_boxes,
                    pred_boxes,
                    name,
                    center_distance,
                    threshold,
                )
                metric_data[threshold] = data_at_threshold
                aps[name][threshold] = calc_ap(data_at_threshold, 0.1, 0.1)
            matched = metric_data[2.0]
            for metric_name in (
                "trans_err",
                "scale_err",
                "orient_err",
                "vel_err",
            ):
                errors[name][metric_name] = calc_tp(matched, 0.1, metric_name)

        class_ap = {
            name: float(np.mean(tuple(aps[name].values()))) for name in CLASSES
        }
        mean_ap = float(np.mean(tuple(class_ap.values())))
        mean_errors = {
            metric_name: float(
                np.nanmean([errors[name][metric_name] for name in self.classes])
            )
            for metric_name in (
                "trans_err",
                "scale_err",
                "orient_err",
                "vel_err",
            )
        }
        tp_scores = {
            name: max(0.0, 1.0 - value) for name, value in mean_errors.items()
        }
        nds = (5.0 * mean_ap + sum(tp_scores.values())) / 9.0
        metrics = {
            "mAP": mean_ap,
            "NDS": nds,
            "mATE": mean_errors["trans_err"],
            "mASE": mean_errors["scale_err"],
            "mAOE": mean_errors["orient_err"],
            "mAVE": mean_errors["vel_err"],
        }
        metrics.update({f"AP/{name}": value for name, value in class_ap.items()})
        return metrics


__all__ = ["KITTI360DAPETRMetric", "KITTI360DetectionBox"]
