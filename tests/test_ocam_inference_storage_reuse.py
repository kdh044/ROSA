"""Contracts for no-grad OCam projection storage reuse."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

from mmengine import Config
from mmdet3d.registry import MODELS
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rosa.geometry import (
    _horner,
    _horner_inference_storage_reuse,
    ocam_project_points,
)
from rosa.integrations.sparse4d import (
    ROSAMetricGatedInferenceReuseAffine3x3FullFOVDeformableFeatureAggregation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
PARENT_CONFIG = (
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "affine3x3_metric_gate.py"
)
CANDIDATE_CONFIG = (
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace_"
    "affine3x3_inference_reuse_metric_gate.py"
)


def _projection_fixture(dtype: torch.dtype, *, camera_conditioned: bool):
    generator = torch.Generator().manual_seed(3407)
    batch, cameras, anchors, points_per_anchor = 2, 4, 17, 13
    if camera_conditioned:
        shape = (batch, cameras, anchors, points_per_anchor, 3)
    else:
        shape = (batch, anchors, points_per_anchor, 3)
    points = torch.randn(shape, generator=generator, dtype=dtype)
    points[..., 2].sub_(2.0)
    lidar2cam = torch.eye(4, dtype=dtype).expand(batch, cameras, 4, 4).clone()
    lidar2cam[..., :3, 3] = torch.randn(
        (batch, cameras, 3),
        generator=generator,
        dtype=dtype,
    ).mul_(0.05)
    intrinsic = torch.zeros((batch, cameras, 3, 3), dtype=dtype)
    intrinsic[..., 0, 0] = 1.0
    intrinsic[..., 0, 1] = -0.000077
    intrinsic[..., 0, 2] = 398.967516
    intrinsic[..., 1, 0] = 0.999625
    intrinsic[..., 1, 1] = -0.000148
    intrinsic[..., 1, 2] = 399.029971
    intrinsic[..., 2, 2] = 1.0
    coefficients = torch.tensor(
        [
            331.189089,
            214.532114,
            16.021439,
            26.018342,
            17.311014,
            4.130653,
            2.940404,
            4.460478,
            2.256958,
            0.36679,
        ],
        dtype=dtype,
    )
    polynomial = coefficients.expand(batch, cameras, -1).clone()
    image_wh = torch.full((batch, cameras, 2), 800.0, dtype=dtype)
    metas = {
        "lidar2cam": lidar2cam,
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "ocam_lens_fov_deg": 220.0,
    }
    return points, metas, image_wh


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.float32, torch.float64, torch.bfloat16),
)
@pytest.mark.parametrize("camera_conditioned", (False, True))
@pytest.mark.parametrize("camera_transform", ("homogeneous_4x4", "affine_3x3"))
def test_inference_storage_reuse_is_raw_bit_equal_and_input_safe(
    dtype,
    camera_conditioned,
    camera_transform,
):
    points, metas, image_wh = _projection_fixture(
        dtype,
        camera_conditioned=camera_conditioned,
    )
    points_before = points.clone()
    metas_before = {
        name: value.clone()
        for name, value in metas.items()
        if name != "ocam_lens_fov_deg"
    }
    image_wh_before = image_wh.clone()
    with torch.no_grad():
        parent = ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform=camera_transform,
        )
        candidate = ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform=camera_transform,
            inference_storage_reuse=True,
        )
    assert torch.equal(candidate, parent)
    assert candidate.stride() == parent.stride()
    assert torch.equal(points, points_before)
    assert torch.equal(image_wh, image_wh_before)
    for name, before in metas_before.items():
        assert torch.equal(metas[name], before)


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.float32, torch.float64, torch.bfloat16),
)
@pytest.mark.parametrize("coefficient_count", (1, 2, 5, 10))
def test_inference_horner_reuse_is_raw_bit_equal(dtype, coefficient_count):
    generator = torch.Generator().manual_seed(7101080 + coefficient_count)
    coefficients = torch.randn(
        (2, 4, 1, 1, coefficient_count),
        generator=generator,
        dtype=dtype,
    )
    value = torch.randn(
        (2, 4, 17, 13),
        generator=generator,
        dtype=dtype,
    )
    with torch.no_grad():
        parent = _horner(coefficients, value)
        candidate = _horner_inference_storage_reuse(coefficients, value)
    assert torch.equal(candidate, parent)
    assert candidate.stride() == parent.stride()


def test_inference_storage_reuse_rejects_grad_enabled_and_invalid_mode():
    points, metas, image_wh = _projection_fixture(
        torch.float32,
        camera_conditioned=False,
    )
    with pytest.raises(RuntimeError, match="requires torch.no_grad"):
        ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform="affine_3x3",
            inference_storage_reuse=True,
        )
    with pytest.raises(TypeError, match="must be a bool"):
        ocam_project_points(
            points,
            metas,
            image_wh,
            inference_storage_reuse="yes",
        )


def test_inference_storage_reuse_preserves_invalid_sentinels():
    points, metas, image_wh = _projection_fixture(
        torch.float32,
        camera_conditioned=False,
    )
    metas["lidar2cam"][..., :3, 3].zero_()
    points[0, 0, 0] = torch.tensor((0.0, 0.0, 0.0))
    points[0, 0, 1] = torch.tensor((float("nan"), 0.0, -1.0))
    points[0, 0, 2] = torch.tensor((float("inf"), 0.0, -1.0))
    points[0, 0, 3] = torch.tensor((1.0e5, 0.0, -1.0))
    with torch.no_grad():
        parent = ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform="affine_3x3",
        )
        candidate = ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform="affine_3x3",
            inference_storage_reuse=True,
        )
    assert torch.equal(candidate, parent)
    assert torch.equal(candidate[0, :, 0, :4], torch.full((4, 4, 2), 2.0))


def test_inference_storage_reuse_preserves_mixed_dtype_promotion():
    points, metas, image_wh = _projection_fixture(
        torch.float32,
        camera_conditioned=False,
    )
    # The generic projection API permits a wider image-extent tensor.  The
    # parent promotes normalization to FP64; an FP32 in-place division would
    # otherwise silently suppress that contract.
    image_wh = image_wh.to(torch.float64)
    with torch.no_grad():
        parent = ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform="affine_3x3",
        )
        candidate = ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform="affine_3x3",
            inference_storage_reuse=True,
        )
    assert parent.dtype == torch.float64
    assert candidate.dtype == parent.dtype
    assert torch.equal(candidate, parent)
    assert candidate.stride() == parent.stride()


def test_inference_storage_reuse_preserves_polynomial_dtype_promotion():
    points, metas, image_wh = _projection_fixture(
        torch.float32,
        camera_conditioned=False,
    )
    metas["ocam_pol"] = metas["ocam_pol"].to(torch.float64)
    with torch.no_grad():
        parent = ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform="affine_3x3",
        )
        candidate = ocam_project_points(
            points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform="affine_3x3",
            inference_storage_reuse=True,
        )
    assert parent.dtype == torch.float64
    assert candidate.dtype == parent.dtype
    assert torch.equal(candidate, parent)
    assert candidate.stride() == parent.stride()


@pytest.mark.parametrize(
    ("contract", "expected_inplace_conjunctions"),
    (("full_fov", 7), ("legacy_front_only", 8)),
)
def test_inference_storage_reuse_owns_incidence_and_validity_buffers(
    contract,
    expected_inplace_conjunctions,
):
    class OperatorLog(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.operations.append(str(func))
            return func(*args, **(kwargs or {}))

    points, metas, image_wh = _projection_fixture(
        torch.float32,
        camera_conditioned=False,
    )
    parent_log = OperatorLog()
    candidate_log = OperatorLog()
    with torch.no_grad(), parent_log:
        parent = ocam_project_points(
            points,
            metas,
            image_wh,
            contract=contract,
            camera_transform="affine_3x3",
        )
    with torch.no_grad(), candidate_log:
        candidate = ocam_project_points(
            points,
            metas,
            image_wh,
            contract=contract,
            camera_transform="affine_3x3",
            inference_storage_reuse=True,
        )

    assert torch.equal(candidate, parent)
    parent_counts = Counter(parent_log.operations)
    candidate_counts = Counter(candidate_log.operations)
    assert parent_counts["aten.atan2.default"] == 2
    assert parent_counts["aten.atan2.out"] == 0
    assert parent_counts["aten.logical_and_.default"] == 0
    assert candidate_counts["aten.atan2.default"] == 1
    assert candidate_counts["aten.atan2.out"] == 1
    assert (
        candidate_counts["aten.logical_and_.default"] == expected_inplace_conjunctions
    )


def test_inference_reuse_class_is_registered_stateless_and_eval_only():
    module_type = (
        ROSAMetricGatedInferenceReuseAffine3x3FullFOVDeformableFeatureAggregation
    )
    assert MODELS.get(module_type.__name__) is module_type
    module = module_type.__new__(module_type)
    torch.nn.Module.__init__(module)
    assert tuple(module.parameters()) == ()
    assert tuple(module.buffers()) == ()
    points, metas, image_wh = _projection_fixture(
        torch.float32,
        camera_conditioned=False,
    )
    with pytest.raises(RuntimeError, match="evaluation-only"):
        module.project_points(
            points,
            torch.empty(0),
            image_wh=image_wh,
            metas=metas,
        )
    module.eval()
    with pytest.raises(RuntimeError, match="requires torch.no_grad"):
        module.project_points(
            points,
            torch.empty(0),
            image_wh=image_wh,
            metas=metas,
        )
    with torch.no_grad():
        output = module.project_points(
            points,
            torch.empty(0),
            image_wh=image_wh,
            metas=metas,
        )
    assert output.shape == (2, 4, 17, 13, 2)


def test_inference_reuse_config_changes_only_deformable_type():
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
    assert parent_type == (
        "ROSAMetricGatedAffine3x3FullFOVDeformableFeatureAggregation"
    )
    assert candidate_type == (
        "ROSAMetricGatedInferenceReuseAffine3x3FullFOV" "DeformableFeatureAggregation"
    )
    normalized = deepcopy(candidate)
    normalized["model"]["head"]["deformable_model"]["type"] = parent_type
    for view in (parent, normalized):
        view.pop("work_dir")
        view.pop("experiment_contract")
    assert normalized == parent
    contract = candidate["experiment_contract"]
    assert contract["training"] == "fail closed"
    assert contract["grad_enabled_evaluation"] == "fail closed"
    assert contract["learned_parameters_added"] == 0
    assert contract["buffers_added"] == 0
    assert contract["checkpoint_keys_added"] == 0
    assert contract["user_numeric_hyperparameters_added"] == 0
