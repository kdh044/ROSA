"""Fail-closed contracts for the tolerance-gated affine OCam port."""

from copy import deepcopy
from pathlib import Path

from mmengine import Config
import pytest
import torch
from mmdet3d.registry import MODELS

from rosa.integrations.sparse4d import (
    ROSAFullFOVDeformableFeatureAggregation,
    ROSAMetricGatedAffine3x3FullFOVDeformableFeatureAggregation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
PARENT_CONFIG = (
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace.py"
)
CANDIDATE_CONFIG = (
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "affine3x3_metric_gate.py"
)


def _projection_fixture():
    points = torch.tensor(
        [[[[1.0, 0.5, -2.0], [0.25, -0.5, -1.5]]]],
        dtype=torch.float64,
    )
    lidar2cam = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4)
    intrinsic = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3)
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    polynomial = torch.zeros(1, 1, 5, dtype=torch.float64)
    polynomial[..., 1] = -150.0
    metas = {
        "lidar2cam": lidar2cam,
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "ocam_lens_fov_deg": 220.0,
    }
    image_wh = torch.full((1, 1, 2), 800.0, dtype=torch.float64)
    return points, metas, image_wh


def _uninitialized(module_type):
    module = module_type.__new__(module_type)
    torch.nn.Module.__init__(module)
    return module


def test_affine_metric_gate_is_registered_stateless_and_eval_only():
    assert MODELS.get(
        "ROSAMetricGatedAffine3x3FullFOVDeformableFeatureAggregation"
    ) is ROSAMetricGatedAffine3x3FullFOVDeformableFeatureAggregation
    candidate = _uninitialized(
        ROSAMetricGatedAffine3x3FullFOVDeformableFeatureAggregation
    )
    assert tuple(candidate.parameters()) == ()
    assert tuple(candidate.buffers()) == ()
    points, metas, image_wh = _projection_fixture()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        candidate.project_points(
            points,
            torch.empty(0),
            image_wh=image_wh,
            metas=metas,
        )

    parent = _uninitialized(ROSAFullFOVDeformableFeatureAggregation).eval()
    candidate.eval()
    expected = parent.project_points(
        points,
        torch.empty(0),
        image_wh=image_wh,
        metas=metas,
    )
    actual = candidate.project_points(
        points,
        torch.empty(0),
        image_wh=image_wh,
        metas=metas,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_affine_metric_gate_config_changes_only_deformable_type():
    parent = Config.fromfile(
        CONFIG_DIR / PARENT_CONFIG,
        import_custom_modules=False,
    ).to_dict()
    candidate = Config.fromfile(
        CONFIG_DIR / CANDIDATE_CONFIG,
        import_custom_modules=False,
    ).to_dict()
    parent_type = parent["model"]["head"]["deformable_model"]["type"]
    candidate_type = candidate["model"]["head"]["deformable_model"]["type"]
    assert parent_type == "ROSAFullFOVDeformableFeatureAggregation"
    assert candidate_type == (
        "ROSAMetricGatedAffine3x3FullFOVDeformableFeatureAggregation"
    )
    normalized = deepcopy(candidate)
    normalized["model"]["head"]["deformable_model"]["type"] = parent_type
    for view in (parent, normalized):
        view.pop("work_dir")
        view.pop("experiment_contract")
    assert normalized == parent
    contract = candidate["experiment_contract"]
    assert contract["training"] == "fail closed"
    assert contract["learned_parameters_added"] == 0
    assert contract["buffers_added"] == 0
    assert contract["checkpoint_keys_added"] == 0
    assert contract["user_numeric_hyperparameters_added"] == 0


def test_unknown_ocam_camera_transform_fails_closed():
    points, metas, image_wh = _projection_fixture()
    from rosa.geometry import ocam_project_points

    with pytest.raises(ValueError, match="unknown OCam camera transform"):
        ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform="approximate",
        )
