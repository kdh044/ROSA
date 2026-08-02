"""Strict CPU contracts for the native C0/C32 evaluation cleanup."""

from __future__ import annotations

import copy
from pathlib import Path
import pickle
import random
import sys
from types import MethodType
import weakref

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

pytest.importorskip("mmengine")
pytest.importorskip("mmdet3d")

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "external" / "detector_host"
if not HOST.is_dir():
    pytest.skip(
        "optional Sparse4D detector host is not available",
        allow_module_level=True,
    )
sys.path.insert(0, str(HOST))

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

from projects.mmdet3d_plugin.models.sparse4d import Sparse4D


@MODELS.register_module(name="RuntimeCleanupBackbonePlain", force=True)
class _RuntimeCleanupBackbonePlain(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, image):
        return [image * self.scale]


@MODELS.register_module(name="RuntimeCleanupBackboneMetas", force=True)
class _RuntimeCleanupBackboneMetas(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, image, num_cams, metas=None):
        assert num_cams == image.shape[0]
        return [image * self.scale + metas["backbone_offset"]]


@MODELS.register_module(name="RuntimeCleanupHead", force=True)
class _RuntimeCleanupHead(nn.Module):
    def forward(self, feature_maps, metas):
        del metas
        return feature_maps


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def _assert_nested_raw_equal(left, right) -> None:
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        assert _raw_equal(left, right)
    elif hasattr(left, "tensor") and torch.is_tensor(left.tensor):
        assert hasattr(right, "tensor")
        assert _raw_equal(left.tensor, right.tensor)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_raw_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_raw_equal(left_item, right_item)
    else:
        assert left == right


def _install_cpu_daf(monkeypatch) -> None:
    from projects.mmdet3d_plugin.models import blocks

    def cpu_daf(_features, _shapes, _starts, _points, weights):
        grouped = weights.sum(dim=(2, 3, 4))
        group_dims = 256 // grouped.shape[-1]
        return grouped[..., None].expand(*grouped.shape, group_dims)

    monkeypatch.setattr(blocks, "DAF", cpu_daf)


def _build_exact_model(config_name: str):
    register_all_modules(init_default_scope=True)
    config = Config.fromfile("configs/" + config_name)
    import_modules_from_strings(**config.custom_imports)
    torch.manual_seed(3407)
    model = MODELS.build(config.model)
    model.eval()
    return model


def _head_fixture(batch: int = 1):
    cameras = 4
    feature_maps = (
        torch.zeros(batch, 1, 256),
        torch.zeros(1, 4, 2, dtype=torch.long),
        torch.zeros(1, 4, dtype=torch.long),
    )
    lidar2cam = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        batch, cameras, 1, 1
    )
    intrinsic = torch.eye(3).reshape(1, 1, 3, 3).repeat(
        batch, cameras, 1, 1
    )
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    polynomial = torch.zeros(batch, cameras, 5)
    polynomial[..., 1] = -150.0
    metas = {
        "projection_mat": lidar2cam,
        "lidar2cam": lidar2cam,
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "image_wh": torch.full((batch, cameras, 2), 800.0),
        "ocam_lens_fov_deg": 220.0,
        "timestamp": torch.zeros(batch),
        "img_metas": [{} for _ in range(batch)],
    }
    return feature_maps, metas


def _run_eval_with_liveness(head, feature_maps, metas):
    references = []
    alive_before_refine = []
    handles = []

    def capture(_module, args):
        alive_before_refine.append(
            sum(reference() is not None for reference in references)
        )
        references.append(weakref.ref(args[0]))

    for operation, layer in zip(head.operation_order, head.layers):
        if operation == "refine":
            handles.append(layer.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            output = head(feature_maps, metas)
            decoded = head.post_process(output)
    finally:
        for handle in handles:
            handle.remove()
    return output, decoded, alive_before_refine


@pytest.mark.parametrize(
    "config_name",
    (
        "rosa_r18_main126_v2_c0_cbgs_native.py",
        "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py",
    ),
)
def test_native_eval_cleanup_is_raw_exact_and_releases_d0_d4_queries(
    config_name,
    monkeypatch,
):
    _install_cpu_daf(monkeypatch)
    baseline = _build_exact_model(config_name)
    reference = copy.deepcopy(baseline)
    candidate = baseline
    reference.head._strict_native_eval_cleanup_enabled = MethodType(
        lambda _self, _temporal: False,
        reference.head,
    )
    feature_maps, metas = _head_fixture()

    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    reference_output, reference_decoded, reference_alive = (
        _run_eval_with_liveness(reference.head, feature_maps, metas)
    )
    reference_rng_after = torch.get_rng_state().clone()
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    candidate_output, candidate_decoded, candidate_alive = (
        _run_eval_with_liveness(candidate.head, feature_maps, metas)
    )
    candidate_rng_after = torch.get_rng_state().clone()

    _assert_nested_raw_equal(reference_output, candidate_output)
    _assert_nested_raw_equal(reference_decoded, candidate_decoded)
    assert torch.equal(reference_rng_after, candidate_rng_after)
    _assert_nested_raw_equal(reference.state_dict(), candidate.state_dict())
    assert len(reference.state_dict()) == len(candidate.state_dict()) == 595
    assert sum(value.numel() for value in reference.parameters()) == 29_667_109
    assert sum(value.numel() for value in candidate.parameters()) == 29_667_109
    assert sum(value.numel() for value in reference.buffers()) == 10_097
    assert sum(value.numel() for value in candidate.buffers()) == 10_097

    # The historical recurrent list holds D0..D4 query tensors at D5.  The
    # cleanup holds none; at B=1 this is 5*300*256*4 = 1,536,000 referenced
    # FP32 bytes, and scales to 6,144,000 bytes at the production B=4.
    assert reference_alive == [0, 1, 2, 3, 4, 5]
    assert candidate_alive == [0, 0, 0, 0, 0, 0]

    private_names = (
        "_last_decoder_aggregation_weights",
        "_last_decoder_aggregation_weights_by_keypoint",
        "_last_decoder_aggregation_keypoint_mass",
        "_last_decoder_ray_object_attention_pre_dropout",
        "_last_decoder_ray_object_valid_views",
        "_last_decoder_query_reliabilities",
        "_last_decoder_state_update_multipliers",
        "_last_reliability_state_strength",
        "_last_reliability_cache_strength",
    )
    for name in private_names:
        _assert_nested_raw_equal(
            getattr(reference.head, name),
            getattr(candidate.head, name),
        )
    for (operation, reference_layer), (_, candidate_layer) in zip(
        zip(reference.head.operation_order, reference.head.layers),
        zip(candidate.head.operation_order, candidate.head.layers),
    ):
        if operation != "deformable":
            continue
        assert candidate_layer._strict_native_eval_diagnostics_are_clear()
        for name in candidate_layer._STRICT_NATIVE_EVAL_CLEAR_ONLY_ATTRIBUTES:
            _assert_nested_raw_equal(
                getattr(reference_layer, name, None),
                getattr(candidate_layer, name, None),
            )
        _assert_nested_raw_equal(
            reference_layer._last_base_key_points,
            candidate_layer._last_base_key_points,
        )
        _assert_nested_raw_equal(
            reference_layer._last_sampling_key_points,
            candidate_layer._last_sampling_key_points,
        )


def test_native_eval_cleanup_fails_closed_for_training_extensions_and_stale_state(
    monkeypatch,
):
    _install_cpu_daf(monkeypatch)
    model = _build_exact_model(
        "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py"
    )
    head = model.head
    assert head._strict_native_eval_cleanup_enabled(False)
    head.train()
    assert not head._strict_native_eval_cleanup_enabled(False)
    head.eval()
    assert not head._strict_native_eval_cleanup_enabled(True)

    for name in (
        "use_decoder_feature_aux",
        "use_fisheye_information_update",
        "use_overlap_support_distillation",
        "use_ray_yaw_attention",
        "use_reliability_state_update",
        "ray_object_regression_sidecar",
    ):
        original = getattr(head, name)
        setattr(head, name, True)
        assert not head._strict_native_eval_cleanup_enabled(False)
        setattr(head, name, original)

    deformable = next(
        layer
        for operation, layer in zip(head.operation_order, head.layers)
        if operation == "deformable"
    )
    deformable._last_view_support = torch.ones(1)
    assert not deformable._strict_native_eval_diagnostics_are_clear()
    feature_maps, metas = _head_fixture()
    with torch.inference_mode():
        head(feature_maps, metas)
    assert deformable._last_view_support is None
    assert deformable._strict_native_eval_diagnostics_are_clear()

    # The fallback must retain the historical external-subclass call
    # signature: an explicit forward accepting the established spherical-RoI
    # keyword, but not the new private keyword, must still work.
    original_forward = deformable.forward

    def explicit_forward(
        _self,
        instance_feature,
        anchor,
        anchor_embed,
        maps,
        metadata,
        spherical_object_roi_query_confidence=None,
    ):
        return original_forward(
            instance_feature,
            anchor,
            anchor_embed,
            maps,
            metadata,
            spherical_object_roi_query_confidence=(
                spherical_object_roi_query_confidence
            ),
        )

    deformable.forward = MethodType(explicit_forward, deformable)
    head._diagnostic_return_all_cls = True
    with torch.inference_mode():
        head(feature_maps, metas)


def test_training_forward_gradients_and_one_step_stay_raw_exact(monkeypatch):
    _install_cpu_daf(monkeypatch)
    baseline = _build_exact_model(
        "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py"
    ).head
    reference = copy.deepcopy(baseline).train()
    candidate = baseline.train()
    for head in (reference, candidate):
        head.sampler.get_dn_anchors = MethodType(
            lambda _self, *_args, **_kwargs: None,
            head.sampler,
        )
        assert not head._strict_native_eval_cleanup_enabled(False)
    feature_maps, metas = _head_fixture()
    metas.update(gt_labels_3d=[], gt_bboxes_3d=[])
    initial_rng = torch.get_rng_state().clone()

    def train_step(head):
        torch.set_rng_state(initial_rng)
        optimizer = torch.optim.SGD(head.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        output = head(feature_maps, metas)
        loss = sum(
            value.float().square().mean()
            for name in ("classification", "prediction", "quality")
            for value in output[name]
            if value is not None
        )
        loss.backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in head.named_parameters()
        }
        optimizer.step()
        return loss.detach(), gradients, torch.get_rng_state().clone()

    reference_loss, reference_gradients, reference_rng = train_step(reference)
    candidate_loss, candidate_gradients, candidate_rng = train_step(candidate)
    assert _raw_equal(reference_loss, candidate_loss)
    assert torch.equal(reference_rng, candidate_rng)
    assert reference_gradients.keys() == candidate_gradients.keys()
    for name in reference_gradients:
        left = reference_gradients[name]
        right = candidate_gradients[name]
        assert (left is None) == (right is None)
        if left is not None:
            assert _raw_equal(left, right)
    _assert_nested_raw_equal(reference.state_dict(), candidate.state_dict())


@pytest.mark.parametrize(
    ("backbone_type", "accepts_metas", "expected_mean"),
    (
        ("RuntimeCleanupBackbonePlain", False, 2.0),
        ("RuntimeCleanupBackboneMetas", True, 5.0),
    ),
)
def test_backbone_signature_route_survives_pickle_and_cpu_ddp(
    backbone_type,
    accepts_metas,
    expected_mean,
    monkeypatch,
    tmp_path,
):
    model = Sparse4D(
        img_backbone=dict(type=backbone_type),
        head=dict(type="RuntimeCleanupHead"),
        use_grid_mask=False,
        use_deformable_func=False,
    )
    assert model._img_backbone_accepts_metas is accepts_metas
    original_keys = tuple(model.state_dict())
    payload = pickle.dumps(model)
    restored = pickle.loads(payload)
    assert restored._img_backbone_accepts_metas is accepts_metas
    assert tuple(restored.state_dict()) == original_keys

    import projects.mmdet3d_plugin.models.sparse4d as sparse4d_module

    monkeypatch.setattr(
        sparse4d_module,
        "signature",
        lambda _method: (_ for _ in ()).throw(
            AssertionError("extract_feat repeated inspect.signature")
        ),
    )
    image = torch.ones(1, 1, 1, 2, 2)
    metas = {"backbone_offset": torch.tensor(3.0)}
    output = restored.extract_feat(image, metas=metas)
    assert output[0].mean().item() == expected_mean

    if dist.is_initialized():
        pytest.skip("an external process group is already active")
    init_file = tmp_path / "gloo_init"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    try:
        wrapped = DistributedDataParallel(restored)
        ddp_output = wrapped(
            {"img": image, "backbone_offset": metas["backbone_offset"]},
            mode="tensor",
        )
        assert _raw_equal(output[0], ddp_output[0])
        assert tuple(wrapped.module.state_dict()) == original_keys
    finally:
        dist.destroy_process_group()
