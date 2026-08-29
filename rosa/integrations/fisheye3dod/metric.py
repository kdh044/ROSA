"""Registry bridge to the exact released Fisheye3DOD evaluator."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

from mmdet3d.registry import METRICS

from .constants import (
    DEFAULT_OFFICIAL_REPO,
    PAIRED_QUERY_BUDGET,
    RELEASED_DECLARED_MAX_BOXES,
)

OFFICIAL_ROOT = Path(
    os.environ.get("FISHEYE3DOD_OFFICIAL_ROOT", DEFAULT_OFFICIAL_REPO)
).resolve()
REFERENCE_ROOT = OFFICIAL_ROOT.parent
if not (OFFICIAL_ROOT / "dataset" / "fisheye3dod_metric.py").is_file():
    raise FileNotFoundError(
        "released Fisheye3DOD metric was not found under "
        f"{OFFICIAL_ROOT}; set FISHEYE3DOD_OFFICIAL_ROOT"
    )
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

_released_metric_module = importlib.import_module(
    f"{OFFICIAL_ROOT.name}.dataset.fisheye3dod_metric"
)
_ReleasedFisheye3DODMetric = _released_metric_module.Fisheye3DODMetric


@METRICS.register_module()
class Fisheye3DODReleasedMetricBridge(_ReleasedFisheye3DODMetric):
    """Preserve the released evaluator's untruncated 300-query behavior."""

    released_declared_max_boxes = RELEASED_DECLARED_MAX_BOXES
    paired_query_budget = PAIRED_QUERY_BUDGET
    applies_manual_prediction_truncation = False


__all__ = ["Fisheye3DODReleasedMetricBridge"]
