from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "analyze_topk_proposals",
    ROOT / "tools" / "analyze_topk_proposals.py",
)
assert _SPEC is not None and _SPEC.loader is not None
analyzer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analyzer
_SPEC.loader.exec_module(analyzer)


def _sample() -> dict:
    scores = torch.linspace(1.0, 0.001, analyzer.QUERY_COUNT)
    labels = torch.arange(analyzer.QUERY_COUNT) % len(analyzer.CLASSES)
    query_indices = (torch.arange(analyzer.QUERY_COUNT) * 37) % analyzer.QUERY_COUNT
    boxes = torch.zeros(analyzer.QUERY_COUNT, 3)
    boxes[:, 0] = torch.arange(analyzer.QUERY_COUNT) + 100.0
    boxes[:, 1] = 100.0

    # Official strict-threshold probes.
    labels[0] = 0
    boxes[0, :2] = torch.tensor((0.5, 0.0))
    labels[1] = 1
    boxes[1, :2] = torch.tensor((10.25, 0.0))

    prediction = {
        "bboxes_3d": boxes,
        "scores_3d": scores,
        "labels_3d": labels,
        "query_indices": query_indices,
        "instance_ids": torch.arange(analyzer.QUERY_COUNT),
        # Deliberately nonsensical: the analyzer must never consume this
        # known-misaligned legacy diagnostic.
        "cls_scores": torch.full((analyzer.QUERY_COUNT,), -1234.0),
    }
    evaluation = {
        "gt_bboxes_3d": torch.tensor(
            (
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
                (36.0, 0.0, 0.0),
            )
        ),
        "gt_labels_3d": torch.tensor((0, 1, 4)),
    }
    return {
        "token": "synthetic-frame",
        "pred_instances_3d": prediction,
        "eval_ann_info": evaluation,
    }


def _copy_sample() -> dict:
    sample = _sample()
    return {
        "token": sample["token"],
        "pred_instances_3d": {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in sample["pred_instances_3d"].items()
        },
        "eval_ann_info": {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in sample["eval_ann_info"].items()
        },
    }


def test_cls_scores_is_fail_closed_forbidden():
    prediction = _sample()["pred_instances_3d"]
    with pytest.raises(analyzer.DumpContractError, match="forbidden"):
        analyzer.safe_prediction_field(prediction, "cls_scores")


@pytest.mark.parametrize("violation", ("permutation", "monotonic", "tie"))
def test_native_d5_dump_contract_rejects_unsafe_rankings(violation: str):
    sample = _copy_sample()
    prediction = sample["pred_instances_3d"]
    if violation == "permutation":
        prediction["query_indices"][0] = prediction["query_indices"][1]
        message = "permutation"
    elif violation == "monotonic":
        prediction["scores_3d"][10] = prediction["scores_3d"][9] + 1.0
        message = "monotonic"
    else:
        prediction["scores_3d"][49] = prediction["scores_3d"][50]
        message = "tie crosses K=50"
    with pytest.raises(analyzer.DumpContractError, match=message):
        analyzer.extract_sample(sample, analyzer.DEFAULT_KS)


def test_equal_and_train_prior_policies_use_predeclared_quotas():
    # Give every class enough candidates to satisfy both quota policies.
    labels = np.concatenate(
        (
            np.zeros(100, dtype=np.int64),
            np.ones(40, dtype=np.int64),
            np.full(40, 2, dtype=np.int64),
            np.full(40, 3, dtype=np.int64),
            np.full(40, 4, dtype=np.int64),
            np.full(40, 5, dtype=np.int64),
        )
    )
    centers = np.stack((np.arange(300), np.zeros(300)), axis=-1).astype(float)

    equal = analyzer.select_rank_positions(
        "equal", labels, centers, 100, dedup_radius_m=0.5
    )
    equal_counts = np.bincount(labels[equal], minlength=6)
    assert equal.size == 100
    assert np.all(equal_counts >= 16)

    prior = analyzer.select_rank_positions(
        "train_prior", labels, centers, 100, dedup_radius_m=0.5
    )
    assert prior.size == 100
    np.testing.assert_array_equal(
        np.bincount(labels[prior], minlength=6),
        analyzer.train_prior_quota(100),
    )
    np.testing.assert_array_equal(
        analyzer.train_prior_quota(100),
        np.asarray((53, 10, 7, 2, 17, 11)),
    )


def test_dedup_uses_higher_score_same_class_bev_local_maxima():
    labels = np.zeros(analyzer.QUERY_COUNT, dtype=np.int64)
    centers = np.stack(
        (np.arange(analyzer.QUERY_COUNT, dtype=float), np.zeros(300)), axis=-1
    )
    centers[0] = (0.0, 0.0)
    centers[1] = (0.1, 0.0)
    centers[2] = (1.0, 0.0)

    selected = analyzer.select_rank_positions(
        "dedup", labels, centers, 2, dedup_radius_m=0.5
    )
    np.testing.assert_array_equal(selected, np.asarray((0, 2)))


def test_end_to_end_outputs_strict_coverage_and_filters_class_range(tmp_path: Path):
    dump = tmp_path / "native_d5.pkl"
    with dump.open("wb") as stream:
        pickle.dump([_sample()], stream)
    output_dir = tmp_path / "analysis"

    summary = analyzer.run(
        dump,
        output_dir,
        allow_partial=True,
    )

    assert summary["samples"] == 1
    # Pedestrian at exactly 36m is outside the evaluator's strict range.
    assert summary["official_range_gt"] == 2
    assert summary["ignored_misaligned_cls_scores_samples"] == 1
    assert summary["forbidden_prediction_fields"] == ["cls_scores"]
    for name in ("summary.json", "recall.csv", "per_gt.csv"):
        assert (output_dir / name).is_file()

    saved = json.loads((output_dir / "summary.json").read_text())
    assert saved["official_range_gt"] == 2
    with (output_dir / "recall.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    def recall(threshold: str) -> dict:
        return next(
            row
            for row in rows
            if row["policy"] == "global"
            and row["k"] == "50"
            and row["matching"] == "class_aware"
            and row["class_name"] == "Car"
            and row["condition"] == "all"
            and row["threshold_m"] == threshold
        )

    # Center distance exactly 0.5 is not a match at 0.5, but is at 1.0.
    assert recall("0.5")["matched_count"] == "0"
    assert recall("1.0")["matched_count"] == "1"

    with (output_dir / "per_gt.csv").open(newline="") as stream:
        per_gt = list(csv.DictReader(stream))
    assert len(per_gt) == 2 * len(analyzer.POLICIES) * len(analyzer.DEFAULT_KS)
    car = next(
        row
        for row in per_gt
        if row["policy"] == "global" and row["k"] == "50" and row["class_name"] == "Car"
    )
    assert float(car["nearest_class_error_m"]) == pytest.approx(0.5)
    assert car["class_lt_0p5m"] == "0"
    assert car["class_lt_1p0m"] == "1"


def test_cli_help_is_available_without_loading_a_dump():
    result = subprocess.run(
        [sys.executable, "tools/analyze_topk_proposals.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
    assert "--allow-partial" in result.stdout
