"""Registers the official Fisheye3DOD evaluator for MMEngine configs."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

from mmdet3d.registry import METRICS

from .constants import DEFAULT_OFFICIAL_REPO

OFFICIAL_ROOT = Path(
    os.environ.get("FISHEYE3DOD_OFFICIAL_ROOT", DEFAULT_OFFICIAL_REPO)
).resolve()
if str(OFFICIAL_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT.parent))
_official = importlib.import_module(f"{OFFICIAL_ROOT.name}.dataset.fisheye3dod_metric")


@METRICS.register_module()
class Fisheye3DODOfficialMetric(_official.Fisheye3DODMetric):
    """The official Fisheye3DOD metric, unchanged; all 300 predictions are kept."""


__all__ = ["Fisheye3DODOfficialMetric"]
