#!/usr/bin/env python3
"""Analyze ranked native-D5 proposals from a trusted MMEngine dump.

The native B0 decoder emits all 300 queries ordered by
``sigmoid(max_class_logit) * sigmoid(centerness_logit)``.  This tool performs
only post-hoc CPU analysis; it never runs a model or exposes evaluation GT to
production inference.

``cls_scores`` from the legacy host decoder is intentionally forbidden here:
that diagnostic is not gathered after centerness reordering and is therefore
not query-aligned.  The aligned fields used by this tool are boxes, joint
scores, frozen labels, and original query indices.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

CLASSES = ("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist")
CLASS_RANGES_M = np.asarray((48.0, 48.0, 48.0, 48.0, 36.0, 42.0))
OFFICIAL_THRESHOLDS_M = (0.5, 1.0, 2.0, 4.0)
DEFAULT_KS = (50, 100, 150, 300)
QUERY_COUNT = 300
EXPECTED_VAL_SAMPLES = 3_906
EXPECTED_OFFICIAL_GT = 22_496

# Counts after the exact strict class-radial filter in corrected-v2 training.
TRAIN_OFFICIAL_CLASS_COUNTS = np.asarray(
    (27_433, 5_168, 3_952, 1_051, 8_627, 5_639), dtype=np.int64
)

POLICIES = ("global", "equal", "train_prior", "dedup")
MATCHING_MODES = ("class_agnostic", "class_aware", "official_class_aware")
FORBIDDEN_PREDICTION_FIELDS = frozenset({"cls_scores"})
SAFE_PREDICTION_FIELDS = frozenset(
    {
        "bboxes_3d",
        "boxes_3d",
        "scores_3d",
        "labels_3d",
        "query_indices",
        "instance_ids",
    }
)
DISTANCE_BINS = (
    (0.0, 10.0, "[0,10)"),
    (10.0, 20.0, "[10,20)"),
    (20.0, 30.0, "[20,30)"),
    (30.0, 40.0, "[30,40)"),
    (40.0, 48.0, "[40,48)"),
)


class DumpContractError(ValueError):
    """Raised when a dump cannot prove the native-D5 analysis contract."""


def _has_field(container: Any, name: str) -> bool:
    if isinstance(container, Mapping):
        return name in container
    try:
        return name in container
    except (TypeError, AttributeError):
        return hasattr(container, name)


def _field(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        if name not in container:
            raise DumpContractError(f"missing field {name!r}")
        return container[name]
    if not _has_field(container, name):
        raise DumpContractError(f"missing field {name!r}")
    return getattr(container, name)


def safe_prediction_field(prediction: Any, name: str) -> Any:
    """Read only fields proven aligned after native centerness sorting."""

    if name in FORBIDDEN_PREDICTION_FIELDS:
        raise DumpContractError(
            f"prediction field {name!r} is forbidden: the legacy decoder "
            "does not align it after centerness sorting"
        )
    if name not in SAFE_PREDICTION_FIELDS:
        raise DumpContractError(f"prediction field {name!r} is not allowlisted")
    return _field(prediction, name)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _centers_xy(boxes: Any) -> np.ndarray:
    if hasattr(boxes, "gravity_center"):
        centers = _as_numpy(boxes.gravity_center)
    elif hasattr(boxes, "tensor"):
        centers = _as_numpy(boxes.tensor)
    else:
        centers = _as_numpy(boxes)
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise DumpContractError(
            f"boxes must expose an [N,D>=2] center tensor, got {centers.shape}"
        )
    centers = np.asarray(centers[:, :2], dtype=np.float64)
    if not np.isfinite(centers).all():
        raise DumpContractError("box XY centers contain non-finite values")
    return centers


def _prediction_boxes(prediction: Any) -> Any:
    if _has_field(prediction, "bboxes_3d"):
        return safe_prediction_field(prediction, "bboxes_3d")
    return safe_prediction_field(prediction, "boxes_3d")


def _sample_token(sample: Any) -> str:
    if _has_field(sample, "token"):
        token = _field(sample, "token")
    elif _has_field(sample, "sample_idx"):
        token = _field(sample, "sample_idx")
    elif hasattr(sample, "metainfo"):
        metadata = sample.metainfo
        token = metadata.get("token", metadata.get("sample_idx"))
    else:
        token = None
    if token is None or not str(token):
        raise DumpContractError("sample is missing a non-empty token")
    return str(token)


def _distance_bin(radius_m: float) -> str:
    for lower, upper, label in DISTANCE_BINS:
        if lower <= radius_m < upper:
            return label
    raise DumpContractError(f"official-range GT has invalid radius {radius_m}")


def _hamilton_quota(k: int, counts: np.ndarray) -> np.ndarray:
    raw = float(k) * counts.astype(np.float64) / float(counts.sum())
    quota = np.floor(raw).astype(np.int64)
    remaining = int(k - quota.sum())
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(raw[index] - quota[index]), index),
    )
    for index in order[:remaining]:
        quota[index] += 1
    if int(quota.sum()) != int(k):
        raise RuntimeError("Hamilton quota did not preserve K")
    return quota


def equal_quota_floor(k: int) -> np.ndarray:
    """Equal-mAP lower quotas; any remainder is filled by global score."""

    return np.full(len(CLASSES), int(k) // len(CLASSES), dtype=np.int64)


def train_prior_quota(k: int) -> np.ndarray:
    return _hamilton_quota(int(k), TRAIN_OFFICIAL_CLASS_COUNTS)


def _quota_selection(labels: np.ndarray, quota: np.ndarray, k: int) -> np.ndarray:
    chosen = np.zeros(labels.shape[0], dtype=bool)
    for class_index, class_quota in enumerate(quota.tolist()):
        positions = np.flatnonzero(labels == class_index)[:class_quota]
        chosen[positions] = True
    selected = np.flatnonzero(chosen).tolist()
    if len(selected) < k:
        for position in range(labels.shape[0]):
            if not chosen[position]:
                chosen[position] = True
                selected.append(position)
                if len(selected) == k:
                    break
    selected_array = np.flatnonzero(chosen)
    if selected_array.size != k:
        raise DumpContractError(f"quota policy selected {selected_array.size}, not {k}")
    return selected_array


def _deduplicated_selection(
    labels: np.ndarray,
    centers_xy: np.ndarray,
    k: int,
    radius_m: float,
) -> np.ndarray:
    """Match the host's deterministic same-class local-maximum policy."""

    if radius_m < 0 or not math.isfinite(radius_m):
        raise ValueError("dedup radius must be finite and non-negative")
    keep: list[int] = []
    squared_radius = radius_m * radius_m
    for position in range(labels.shape[0]):
        if position == 0 or radius_m == 0:
            keep.append(position)
            continue
        higher = np.arange(position)
        same_class = labels[higher] == labels[position]
        delta = centers_xy[higher] - centers_xy[position]
        close = np.einsum("ij,ij->i", delta, delta) <= squared_radius
        if not bool(np.any(same_class & close)):
            keep.append(position)
    selected = keep[:k]
    selected_set = set(selected)
    if len(selected) < k:
        for position in range(labels.shape[0]):
            if position not in selected_set:
                selected.append(position)
                selected_set.add(position)
                if len(selected) == k:
                    break
    result = np.asarray(sorted(selected), dtype=np.int64)
    if result.size != k:
        raise DumpContractError(f"dedup policy selected {result.size}, not {k}")
    return result


def select_rank_positions(
    policy: str,
    labels: np.ndarray,
    centers_xy: np.ndarray,
    k: int,
    *,
    dedup_radius_m: float,
) -> np.ndarray:
    """Select positions from arrays already sorted by native joint score."""

    if policy not in POLICIES:
        raise ValueError(f"unknown selection policy {policy!r}")
    if not 0 < int(k) <= labels.shape[0]:
        raise ValueError("K must be in (0, number of queries]")
    if policy == "global":
        return np.arange(k, dtype=np.int64)
    if policy == "equal":
        return _quota_selection(labels, equal_quota_floor(k), k)
    if policy == "train_prior":
        return _quota_selection(labels, train_prior_quota(k), k)
    return _deduplicated_selection(labels, centers_xy, k, dedup_radius_m)


@dataclass
class SampleData:
    token: str
    scores: np.ndarray
    labels: np.ndarray
    query_indices: np.ndarray
    centers_xy: np.ndarray
    gt_centers_xy: np.ndarray
    gt_labels: np.ndarray
    gt_indices: np.ndarray
    cls_scores_present: bool


def extract_sample(sample: Any, ks: Sequence[int]) -> SampleData:
    prediction = _field(sample, "pred_instances_3d")
    evaluation = _field(sample, "eval_ann_info")
    token = _sample_token(sample)

    scores = np.asarray(
        _as_numpy(safe_prediction_field(prediction, "scores_3d")),
        dtype=np.float64,
    ).reshape(-1)
    labels = np.asarray(
        _as_numpy(safe_prediction_field(prediction, "labels_3d")),
        dtype=np.int64,
    ).reshape(-1)
    query_indices = np.asarray(
        _as_numpy(safe_prediction_field(prediction, "query_indices")),
        dtype=np.int64,
    ).reshape(-1)
    centers_xy = _centers_xy(_prediction_boxes(prediction))

    expected_shape = (QUERY_COUNT,)
    if any(array.shape != expected_shape for array in (scores, labels, query_indices)):
        raise DumpContractError(f"{token}: native D5 must emit exactly 300 queries")
    if centers_xy.shape != (QUERY_COUNT, 2):
        raise DumpContractError(f"{token}: D5 box/query axes are misaligned")
    if not np.isfinite(scores).all():
        raise DumpContractError(f"{token}: joint scores contain non-finite values")
    if np.any(scores[:-1] < scores[1:]):
        raise DumpContractError(f"{token}: joint scores are not monotonic descending")
    if not np.array_equal(np.sort(query_indices), np.arange(QUERY_COUNT)):
        raise DumpContractError(f"{token}: query_indices are not a 0..299 permutation")
    if np.any((labels < 0) | (labels >= len(CLASSES))):
        raise DumpContractError(f"{token}: predicted class label is invalid")
    for k in ks:
        if k < QUERY_COUNT and scores[k - 1] == scores[k]:
            raise DumpContractError(
                f"{token}: score tie crosses K={k}; ranked dump cannot prove "
                "the direct GPU top-k membership"
            )

    gt_boxes = _field(evaluation, "gt_bboxes_3d")
    gt_labels_all = np.asarray(
        _as_numpy(_field(evaluation, "gt_labels_3d")), dtype=np.int64
    ).reshape(-1)
    gt_centers_all = _centers_xy(gt_boxes)
    if gt_centers_all.shape[0] != gt_labels_all.shape[0]:
        raise DumpContractError(f"{token}: evaluation GT boxes/labels are misaligned")
    if np.any((gt_labels_all < 0) | (gt_labels_all >= len(CLASSES))):
        raise DumpContractError(f"{token}: evaluation GT class label is invalid")
    gt_radius = np.linalg.norm(gt_centers_all, axis=1)
    keep = gt_radius < CLASS_RANGES_M[gt_labels_all]

    return SampleData(
        token=token,
        scores=scores,
        labels=labels,
        query_indices=query_indices,
        centers_xy=centers_xy,
        gt_centers_xy=gt_centers_all[keep],
        gt_labels=gt_labels_all[keep],
        gt_indices=np.flatnonzero(keep).astype(np.int64),
        cls_scores_present=_has_field(prediction, "cls_scores"),
    )


@dataclass
class Aggregate:
    gt_count: int = 0
    finite_error_count: int = 0
    error_sum_m: float = 0.0
    matched: dict[float, int] = field(
        default_factory=lambda: {threshold: 0 for threshold in OFFICIAL_THRESHOLDS_M}
    )

    def add(self, error_m: float) -> None:
        self.gt_count += 1
        if math.isfinite(error_m):
            self.finite_error_count += 1
            self.error_sum_m += float(error_m)
        for threshold in OFFICIAL_THRESHOLDS_M:
            # Match the evaluator: equality is not a true positive.
            self.matched[threshold] += int(error_m < threshold)


def _nearest(
    selected_centers: np.ndarray,
    selected_query_indices: np.ndarray,
    gt_center: np.ndarray,
    eligible: np.ndarray,
) -> tuple[float, int]:
    positions = np.flatnonzero(eligible)
    if positions.size == 0:
        return math.inf, -1
    distances = np.linalg.norm(selected_centers[positions] - gt_center, axis=1)
    local = int(np.argmin(distances))
    position = int(positions[local])
    return float(distances[local]), int(selected_query_indices[position])


def _aggregate_keys(
    policy: str,
    k: int,
    matching: str,
    class_name: str,
    distance_bin: str,
) -> Iterable[tuple[str, int, str, str, str, str]]:
    yield (policy, k, matching, "__all__", "all", "all")
    yield (policy, k, matching, class_name, "all", "all")
    yield (policy, k, matching, "__all__", "distance", distance_bin)
    yield (policy, k, matching, class_name, "distance", distance_bin)


PER_GT_FIELDS = (
    "token",
    "gt_index",
    "class_name",
    "distance_m",
    "distance_bin",
    "policy",
    "k",
    "nearest_any_error_m",
    "nearest_any_query_index",
    "nearest_class_error_m",
    "nearest_class_query_index",
    "nearest_official_class_error_m",
    "nearest_official_class_query_index",
    *tuple(
        f"{mode}_lt_{str(threshold).replace('.', 'p')}m"
        for mode in ("any", "class", "official_class")
        for threshold in OFFICIAL_THRESHOLDS_M
    ),
)


def _csv_float(value: float) -> str | float:
    return "" if not math.isfinite(value) else value


def analyze_samples(
    samples: Sequence[Any],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    dedup_radius_m: float = 0.5,
    per_gt_writer: csv.DictWriter | None = None,
    expected_samples: int | None = EXPECTED_VAL_SAMPLES,
    expected_official_gt: int | None = EXPECTED_OFFICIAL_GT,
) -> dict[str, Any]:
    raw_ks = tuple(int(k) for k in ks)
    if len(set(raw_ks)) != len(raw_ks):
        raise ValueError("K values must be unique")
    ks = tuple(sorted(raw_ks))
    if not ks or ks[0] <= 0 or ks[-1] > QUERY_COUNT:
        raise ValueError("K values must be unique integers in [1,300]")
    if not samples:
        raise DumpContractError("dump contains no data samples")
    if expected_samples is not None and len(samples) != expected_samples:
        raise DumpContractError(
            f"expected {expected_samples} validation samples, got {len(samples)}"
        )

    aggregates: dict[tuple[str, int, str, str, str, str], Aggregate] = {}
    selection_counts: dict[tuple[str, int, int], int] = {}
    zero_class_frames: dict[tuple[str, int, int], int] = {}
    seen_tokens: set[str] = set()
    official_gt_count = 0
    cls_scores_present_samples = 0

    for raw_sample in samples:
        sample = extract_sample(raw_sample, ks)
        if sample.token in seen_tokens:
            raise DumpContractError(f"duplicate sample token {sample.token!r}")
        seen_tokens.add(sample.token)
        official_gt_count += int(sample.gt_labels.size)
        cls_scores_present_samples += int(sample.cls_scores_present)

        for policy in POLICIES:
            for k in ks:
                positions = select_rank_positions(
                    policy,
                    sample.labels,
                    sample.centers_xy,
                    k,
                    dedup_radius_m=dedup_radius_m,
                )
                selected_labels = sample.labels[positions]
                selected_centers = sample.centers_xy[positions]
                selected_queries = sample.query_indices[positions]
                for class_index in range(len(CLASSES)):
                    count = int(np.sum(selected_labels == class_index))
                    key = (policy, k, class_index)
                    selection_counts[key] = selection_counts.get(key, 0) + count
                    zero_class_frames[key] = zero_class_frames.get(key, 0) + int(
                        count == 0
                    )

                selected_radius = np.linalg.norm(selected_centers, axis=1)
                within_prediction_range = (
                    selected_radius < CLASS_RANGES_M[selected_labels]
                )
                for local_gt_index, (gt_center, gt_label, raw_gt_index) in enumerate(
                    zip(
                        sample.gt_centers_xy,
                        sample.gt_labels,
                        sample.gt_indices,
                    )
                ):
                    del local_gt_index
                    gt_label = int(gt_label)
                    class_name = CLASSES[gt_label]
                    radius_m = float(np.linalg.norm(gt_center))
                    distance_bin = _distance_bin(radius_m)
                    any_error, any_query = _nearest(
                        selected_centers,
                        selected_queries,
                        gt_center,
                        np.ones(k, dtype=bool),
                    )
                    same_class = selected_labels == gt_label
                    class_error, class_query = _nearest(
                        selected_centers,
                        selected_queries,
                        gt_center,
                        same_class,
                    )
                    official_error, official_query = _nearest(
                        selected_centers,
                        selected_queries,
                        gt_center,
                        same_class & within_prediction_range,
                    )
                    errors = {
                        "class_agnostic": any_error,
                        "class_aware": class_error,
                        "official_class_aware": official_error,
                    }
                    for matching, error_m in errors.items():
                        for key in _aggregate_keys(
                            policy, k, matching, class_name, distance_bin
                        ):
                            aggregates.setdefault(key, Aggregate()).add(error_m)

                    if per_gt_writer is not None:
                        row: dict[str, Any] = {
                            "token": sample.token,
                            "gt_index": int(raw_gt_index),
                            "class_name": class_name,
                            "distance_m": radius_m,
                            "distance_bin": distance_bin,
                            "policy": policy,
                            "k": k,
                            "nearest_any_error_m": _csv_float(any_error),
                            "nearest_any_query_index": any_query,
                            "nearest_class_error_m": _csv_float(class_error),
                            "nearest_class_query_index": class_query,
                            "nearest_official_class_error_m": _csv_float(
                                official_error
                            ),
                            "nearest_official_class_query_index": official_query,
                        }
                        for prefix, error_m in (
                            ("any", any_error),
                            ("class", class_error),
                            ("official_class", official_error),
                        ):
                            for threshold in OFFICIAL_THRESHOLDS_M:
                                name = (
                                    f"{prefix}_lt_"
                                    f"{str(threshold).replace('.', 'p')}m"
                                )
                                row[name] = int(error_m < threshold)
                        per_gt_writer.writerow(row)

    if expected_official_gt is not None and official_gt_count != expected_official_gt:
        raise DumpContractError(
            f"expected {expected_official_gt} official-range GT, "
            f"got {official_gt_count}"
        )

    class_order = {"__all__": -1, **{name: i for i, name in enumerate(CLASSES)}}
    condition_order = {"all": 0, "distance": 1}
    policy_order = {name: i for i, name in enumerate(POLICIES)}
    matching_order = {name: i for i, name in enumerate(MATCHING_MODES)}
    sorted_stats = sorted(
        aggregates.items(),
        key=lambda item: (
            policy_order[item[0][0]],
            item[0][1],
            matching_order[item[0][2]],
            class_order[item[0][3]],
            condition_order[item[0][4]],
            item[0][5],
        ),
    )
    recall_rows: list[dict[str, Any]] = []
    for (policy, k, matching, class_name, condition, bin_name), stats in sorted_stats:
        mean_error = (
            stats.error_sum_m / stats.finite_error_count
            if stats.finite_error_count
            else None
        )
        for threshold in OFFICIAL_THRESHOLDS_M:
            matched = stats.matched[threshold]
            recall_rows.append(
                {
                    "policy": policy,
                    "k": k,
                    "matching": matching,
                    "class_name": class_name,
                    "condition": condition,
                    "bin": bin_name,
                    "threshold_m": threshold,
                    "gt_count": stats.gt_count,
                    "finite_error_count": stats.finite_error_count,
                    "mean_center_error_m": mean_error,
                    "matched_count": matched,
                    "recall": matched / stats.gt_count if stats.gt_count else None,
                }
            )

    selection_rows = []
    for policy in POLICIES:
        for k in ks:
            for class_index, class_name in enumerate(CLASSES):
                key = (policy, k, class_index)
                total = selection_counts.get(key, 0)
                selection_rows.append(
                    {
                        "policy": policy,
                        "k": k,
                        "class_name": class_name,
                        "selected_queries": total,
                        "mean_queries_per_frame": total / len(samples),
                        "zero_query_frames": zero_class_frames.get(key, 0),
                    }
                )

    return {
        "schema_version": 1,
        "protocol": "Fisheye3DOD Main126 corrected-v2 native D5",
        "score_contract": "sigmoid(max_class_logit) * sigmoid(CNS)",
        "query_count": QUERY_COUNT,
        "ks": list(ks),
        "thresholds_m": list(OFFICIAL_THRESHOLDS_M),
        "class_ranges_m": dict(zip(CLASSES, CLASS_RANGES_M.tolist())),
        "samples": len(samples),
        "official_range_gt": official_gt_count,
        "validation": {
            "query_indices_permutation": True,
            "joint_scores_monotonic_descending": True,
            "requested_k_boundary_ties": 0,
            "gt_class_range_comparison": "strict_less_than",
            "coverage_distance_comparison": "strict_less_than",
        },
        "forbidden_prediction_fields": sorted(FORBIDDEN_PREDICTION_FIELDS),
        "ignored_misaligned_cls_scores_samples": cls_scores_present_samples,
        "policies": {
            "global": "joint-score prefix",
            "equal": "floor(K/6) per frozen class, then global-score fill",
            "train_prior": "Hamilton quota from corrected train official-range GT",
            "dedup": ("same-class higher-score BEV local maxima, then global fill"),
            "dedup_radius_m": dedup_radius_m,
            "equal_floor_quotas": {str(k): equal_quota_floor(k).tolist() for k in ks},
            "train_prior_quotas": {str(k): train_prior_quota(k).tolist() for k in ks},
        },
        "selection": selection_rows,
        "recall": recall_rows,
    }


RECALL_FIELDS = (
    "policy",
    "k",
    "matching",
    "class_name",
    "condition",
    "bin",
    "threshold_m",
    "gt_count",
    "finite_error_count",
    "mean_center_error_m",
    "matched_count",
    "recall",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trusted_dump(path: Path) -> Sequence[Any]:
    # Pickle can execute code. This command is intentionally documented and
    # named for trusted, locally produced MMEngine DumpResults artifacts only.
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if isinstance(value, Mapping) and "results" in value:
        value = value["results"]
    if not isinstance(value, (list, tuple)):
        raise DumpContractError("dump must contain a list of data samples")
    return value


def _ensure_outputs_available(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing artifacts: {joined}")


def run(
    dump_path: Path,
    output_dir: Path,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    dedup_radius_m: float = 0.5,
    allow_partial: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    dump_path = dump_path.resolve()
    output_dir = output_dir.resolve()
    if not dump_path.is_file():
        raise FileNotFoundError(dump_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    recall_path = output_dir / "recall.csv"
    per_gt_path = output_dir / "per_gt.csv"
    outputs = (summary_path, recall_path, per_gt_path)
    _ensure_outputs_available(outputs, overwrite)

    samples = _load_trusted_dump(dump_path)
    per_gt_temp = output_dir / ".per_gt.csv.tmp"
    with per_gt_temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PER_GT_FIELDS)
        writer.writeheader()
        summary = analyze_samples(
            samples,
            ks=ks,
            dedup_radius_m=dedup_radius_m,
            per_gt_writer=writer,
            expected_samples=None if allow_partial else EXPECTED_VAL_SAMPLES,
            expected_official_gt=None if allow_partial else EXPECTED_OFFICIAL_GT,
        )

    summary["dump"] = str(dump_path)
    summary["dump_sha256"] = _sha256(dump_path)
    summary["artifacts"] = {
        "summary": str(summary_path),
        "recall": str(recall_path),
        "per_gt": str(per_gt_path),
    }

    recall_temp = output_dir / ".recall.csv.tmp"
    with recall_temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECALL_FIELDS)
        writer.writeheader()
        writer.writerows(summary["recall"])

    summary_temp = output_dir / ".summary.json.tmp"
    summary_temp.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(per_gt_temp, per_gt_path)
    os.replace(recall_temp, recall_path)
    os.replace(summary_temp, summary_path)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dump",
        type=Path,
        help="Trusted local .pkl produced by tools/run.py evaluate --dump.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--dedup-radius-m", type=float, default=0.5)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Skip the 3906-frame/22496-GT Main126 completeness checks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing summary.json, recall.csv, and per_gt.csv.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run(
        args.dump,
        args.output_dir,
        ks=args.ks,
        dedup_radius_m=args.dedup_radius_m,
        allow_partial=args.allow_partial,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "samples": summary["samples"],
                "official_range_gt": summary["official_range_gt"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
