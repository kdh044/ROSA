from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "assets" / "fisheye3dod_main126_train_kmeans300_seed3407.npy"
ANCHOR_SHA256 = "2a08ac6a28b6eb519e59e9d9c16bae009184785f2b8745cd857576a28496b812"


def test_release_anchor_checksum_and_shape():
    assert hashlib.sha256(ANCHOR.read_bytes()).hexdigest() == ANCHOR_SHA256
    anchors = np.load(ANCHOR)
    assert anchors.shape == (300, 11)
    assert anchors.dtype == np.float32


def test_release_config_uses_only_clean_integration_imports():
    source = (ROOT / "configs" / "rosa_r18_main126.py").read_text()
    assert "/home/" not in source
    assert "final_surface_rosa_rfl" not in source
    assert "fisheye3dod_official" not in source
    for module in (
        "rosa.integrations.sparse4d",
        "rosa.integrations.optim",
        "rosa.integrations.fisheye3dod.dataset",
        "rosa.integrations.fisheye3dod.model",
        "rosa.integrations.fisheye3dod.metric",
        "rosa.integrations.fisheye_box_sparse4d",
    ):
        assert module in source
    assert "type='FisheyeSparseBoxHead'" in source
    assert "camera_rfl_before_rosa=True" in source
    assert "num_temp_instances=0" in source
    assert "sparse4d_size_head_layers=(" in source
    assert all(f"            {stage}," in source for stage in range(6))
    assert "V49" not in source
    assert "C0" not in source
