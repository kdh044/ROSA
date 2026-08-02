from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

from mmengine import Config
from mmengine.model import BaseModel
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules
import numpy as np
import pytest
import torch
from projects.mmdet3d_plugin.data_preprocessor.sparse4d_preprocessor import (
    Sparse4DDataPreprocessor,
)

from rosa.integrations.fisheye3dod.model import (
    Fisheye3DODGravityCenterSparse4D,
    Fisheye3DODSingleFrameDataPreprocessor,
    _require_temporal_free_metadata_path,
)
from rosa.integrations.sparse4d import (
    ROSANativeBaselineHead,
    ROSAOfficialResidualNativeBaselineHead,
    ROSARefinementHead,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
BASE_CONFIG = "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py"
CPU_CONFIG = "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_cpu_metadata.py"
HOST_KEYS = ("T_global", "T_global_inv", "timestamp")


def _batch():
    return {
        "inputs": {
            "img": torch.randint(0, 255, (2, 4, 3, 8, 8), dtype=torch.uint8),
            "T_global": torch.eye(4).repeat(2, 1, 1),
            "T_global_inv": torch.eye(4).repeat(2, 1, 1),
            "timestamp": torch.tensor((0.1, 0.2), dtype=torch.float64),
            "image_wh": torch.full((2, 4, 2), 8.0),
        },
        "data_samples": [],
    }


def _preprocessor(cls):
    return cls(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True,
        pad_size_divisor=32,
    )


def _assert_exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, dict):
        assert tuple(left) == tuple(right)
        for name in left:
            _assert_exact(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_exact(left_value, right_value)
    else:
        assert left == right


def test_prediction_excludes_temporal_metadata_from_device_cast(monkeypatch):
    preprocessor = Fisheye3DODSingleFrameDataPreprocessor()
    original_cast = preprocessor.cast_data
    observed = []

    def recording_cast(data):
        if isinstance(data, dict) and isinstance(data.get("inputs"), dict):
            observed.append(set(data["inputs"]))
        return original_cast(data)

    monkeypatch.setattr(preprocessor, "cast_data", recording_cast)
    batch = _batch()
    originals = {name: batch["inputs"][name] for name in HOST_KEYS}

    output = preprocessor(batch, training=False)

    assert observed
    assert all(name not in observed[0] for name in HOST_KEYS)
    assert all(output["inputs"][name] is originals[name] for name in HOST_KEYS)
    assert output["inputs"]["img"].dtype == torch.float32


def test_training_retains_the_original_device_cast_contract(monkeypatch):
    preprocessor = Fisheye3DODSingleFrameDataPreprocessor()
    original_cast = preprocessor.cast_data
    observed = []

    def recording_cast(data):
        if isinstance(data, dict) and isinstance(data.get("inputs"), dict):
            observed.append(set(data["inputs"]))
        return original_cast(data)

    monkeypatch.setattr(preprocessor, "cast_data", recording_cast)

    preprocessor(_batch(), training=True)

    assert observed
    assert all(name in observed[0] for name in HOST_KEYS)


def test_prediction_preserves_processed_inputs_and_internal_metas_bitwise():
    baseline = _preprocessor(Sparse4DDataPreprocessor)
    candidate = _preprocessor(Fisheye3DODSingleFrameDataPreprocessor)
    baseline_batch = deepcopy(_batch())
    candidate_batch = deepcopy(baseline_batch)
    rng_before = torch.get_rng_state().clone()

    baseline_output = baseline(baseline_batch, training=False)
    baseline_rng = torch.get_rng_state().clone()
    candidate_output = candidate(candidate_batch, training=False)
    candidate_rng = torch.get_rng_state().clone()

    _assert_exact(baseline_output, candidate_output)
    assert torch.equal(rng_before, baseline_rng)
    assert torch.equal(baseline_rng, candidate_rng)
    data_samples = [
        SimpleNamespace(metainfo={"sample_idx": index}) for index in range(2)
    ]
    baseline_metas = Fisheye3DODGravityCenterSparse4D._build_metas(
        baseline_output["inputs"], data_samples
    )
    candidate_metas = Fisheye3DODGravityCenterSparse4D._build_metas(
        candidate_output["inputs"], data_samples
    )
    _assert_exact(baseline_metas, candidate_metas)


def test_prediction_list_path_fails_closed_as_unsupported_tta():
    candidate = _preprocessor(Fisheye3DODSingleFrameDataPreprocessor)

    with pytest.raises(ValueError, match="does not support TTA/list input"):
        candidate([_batch()], training=False)


class _MetadataEchoModel(BaseModel):
    def forward(self, inputs, data_samples=None, mode="tensor"):
        assert mode == "predict"
        return inputs


def test_mmengine_test_step_uses_the_prediction_only_metadata_path(monkeypatch):
    preprocessor = _preprocessor(Fisheye3DODSingleFrameDataPreprocessor)
    original_cast = preprocessor.cast_data
    observed = []

    def recording_cast(data):
        if isinstance(data, dict) and isinstance(data.get("inputs"), dict):
            observed.append(tuple(data["inputs"]))
        return original_cast(data)

    monkeypatch.setattr(preprocessor, "cast_data", recording_cast)
    model = _MetadataEchoModel(data_preprocessor=preprocessor)
    output = model.test_step(_batch())

    assert observed
    assert all(name not in observed[0] for name in HOST_KEYS)
    assert all(output[name].device.type == "cpu" for name in HOST_KEYS)


def test_non_cpu_prediction_metadata_fails_closed():
    preprocessor = Fisheye3DODSingleFrameDataPreprocessor()
    batch = _batch()
    batch["inputs"]["timestamp"] = torch.empty(2, device="meta")

    with pytest.raises(ValueError, match="timestamp must enter on CPU"):
        preprocessor(batch, training=False)


def _audited_head(
    count=0,
    operation_order=("deformable", "refine", "gnn"),
    head_type=ROSAOfficialResidualNativeBaselineHead,
):
    head = object.__new__(head_type)
    object.__setattr__(
        head,
        "instance_bank",
        SimpleNamespace(num_temp_instances=count),
    )
    object.__setattr__(head, "operation_order", operation_order)
    return head


def test_only_exact_audited_non_temporal_heads_are_accepted():
    class ExtendedC0Head(ROSANativeBaselineHead):
        pass

    class ExtendedROSAHead(ROSARefinementHead):
        pass

    for head_type in (
        ROSANativeBaselineHead,
        ROSAOfficialResidualNativeBaselineHead,
        ROSARefinementHead,
    ):
        _require_temporal_free_metadata_path(_audited_head(head_type=head_type))
    for extended_type in (ExtendedC0Head, ExtendedROSAHead):
        with pytest.raises(TypeError, match="exact audited temporal-free"):
            _require_temporal_free_metadata_path(
                _audited_head(head_type=extended_type)
            )

    with pytest.raises(ValueError, match="num_temp_instances=0"):
        _require_temporal_free_metadata_path(_audited_head(count=1))
    for invalid in (0.5, "0", True, None):
        with pytest.raises(TypeError, match="exact integer zero"):
            _require_temporal_free_metadata_path(_audited_head(count=invalid))
    with pytest.raises(TypeError, match="exact audited temporal-free"):
        _require_temporal_free_metadata_path(
            SimpleNamespace(instance_bank=SimpleNamespace(num_temp_instances=0))
        )
    missing_bank = _audited_head()
    object.__setattr__(missing_bank, "instance_bank", None)
    with pytest.raises(TypeError, match="exact integer zero"):
        _require_temporal_free_metadata_path(missing_bank)
    with pytest.raises(ValueError, match="forbids temp_gnn"):
        _require_temporal_free_metadata_path(
            _audited_head(operation_order=("deformable", "temp_gnn", "refine"))
        )


def test_runtime_config_changes_only_the_preprocessor_type():
    baseline = Config.fromfile(
        CONFIG_DIR / BASE_CONFIG,
        import_custom_modules=False,
    ).to_dict()
    candidate = Config.fromfile(
        CONFIG_DIR / CPU_CONFIG,
        import_custom_modules=False,
    ).to_dict()

    assert baseline["model"]["data_preprocessor"]["type"] == (
        "Sparse4DDataPreprocessor"
    )
    assert candidate["model"]["data_preprocessor"]["type"] == (
        "Fisheye3DODSingleFrameDataPreprocessor"
    )
    baseline_view = deepcopy(baseline)
    candidate_view = deepcopy(candidate)
    baseline_view["model"]["data_preprocessor"]["type"] = candidate_view["model"][
        "data_preprocessor"
    ]["type"]
    for view in (baseline_view, candidate_view):
        view.pop("work_dir")
        view.pop("experiment_contract")
    assert baseline_view == candidate_view


def test_runtime_model_has_identical_state_parameters_buffers_and_rng():
    register_all_modules(init_default_scope=True)

    def build(config_name):
        random.seed(3407)
        np.random.seed(3407)
        torch.manual_seed(3407)
        config = Config.fromfile(CONFIG_DIR / config_name)
        import_modules_from_strings(**config.custom_imports)
        model = MODELS.build(config.model)
        return model, torch.get_rng_state().clone()

    baseline, baseline_rng = build(BASE_CONFIG)
    candidate, candidate_rng = build(CPU_CONFIG)
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()

    assert baseline_state.keys() == candidate_state.keys()
    for name, value in baseline_state.items():
        assert torch.equal(value, candidate_state[name])
    assert torch.equal(baseline_rng, candidate_rng)
    assert sum(value.numel() for value in baseline.parameters()) == sum(
        value.numel() for value in candidate.parameters()
    )
    baseline_buffers = dict(baseline.named_buffers())
    candidate_buffers = dict(candidate.named_buffers())
    assert baseline_buffers.keys() == candidate_buffers.keys()
    for name, value in baseline_buffers.items():
        assert torch.equal(value, candidate_buffers[name])
    candidate.load_state_dict(baseline_state, strict=True)
