from collections import Counter
import warnings

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _python_dispatch
from torch.utils._python_dispatch import TorchDispatchMode

from rosa.attention import PointwiseROSA
from rosa.config import ROSAConfig
from rosa.rfl import (
    SphericalRayFeatureLifting,
    _mask_cast_and_add_owned_delta,
)
from rosa.types import RayFeatureRelation


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


_SPECIAL_BITS = {
    torch.float16: (
        torch.uint16,
        (0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x7FFF),
    ),
    torch.bfloat16: (
        torch.uint16,
        (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7FFF),
    ),
    torch.float32: (
        torch.uint32,
        (
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7FFFFFFF,
        ),
    ),
    torch.float64: (
        torch.uint64,
        (
            0x0000000000000000,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000000001,
            0x7FFFFFFFFFFFFFFF,
        ),
    ),
}


def _inject_special_values(value: torch.Tensor) -> None:
    integer_dtype, patterns = _SPECIAL_BITS[value.dtype]
    integer_view = value.view(integer_dtype).reshape(-1)
    integer_view[: len(patterns)].copy_(torch.tensor(patterns, dtype=integer_dtype))


def _rfl_fixture(dtype: torch.dtype, *, factorized: bool = False):
    generator = torch.Generator().manual_seed(7101080)
    module = (
        SphericalRayFeatureLifting.for_compact_profile(embed_dims=8)
        if factorized
        else SphericalRayFeatureLifting(embed_dims=8, hidden_dims=4)
    ).eval()
    with torch.no_grad():
        module.appearance_proj.weight.normal_(generator=generator)
        if module.ray_proj is not None:
            module.ray_proj.weight.normal_(generator=generator)
        module.output_proj.weight.normal_(generator=generator)
    assert torch.count_nonzero(module.output_proj.weight) == (
        module.output_proj.weight.numel()
    )

    features = torch.randn(
        2,
        3,
        4,
        5,
        8,
        generator=generator,
        dtype=dtype,
    )
    _inject_special_values(features)
    ray = torch.randn(
        2,
        3,
        4,
        5,
        3,
        generator=generator,
        dtype=dtype,
    )
    ray.reshape(-1)[:3] = torch.tensor(
        (float("nan"), float("inf"), float("-inf")),
        dtype=dtype,
    )
    valid = torch.rand(2, 3, 4, 5, generator=generator) > 0.25
    valid.reshape(-1)[:6] = torch.tensor((True, False, True, False, True, False))
    return module, features, RayFeatureRelation(ray=ray, valid=valid)


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
@pytest.mark.parametrize("factorized", (False, True))
def test_owned_delta_rfl_is_raw_exact_for_nonzero_weights_and_special_values(
    dtype,
    factorized,
):
    module, features, relation = _rfl_fixture(dtype, factorized=factorized)
    features_before = features.clone()
    ray_before = relation.ray.clone()
    valid_before = relation.valid.clone()

    with torch.no_grad():
        assert module._can_reuse_owned_delta(features, relation)
        expected = features + 1.0 * module(features, relation)
        actual = module._forward_lifted_owned_delta_inference(features, relation)

    assert _raw_equal(actual, expected)
    assert actual.stride() == expected.stride()
    assert actual._base is None
    assert actual.data_ptr() != features.data_ptr()
    assert _raw_equal(features, features_before)
    assert _raw_equal(relation.ray, ray_before)
    assert torch.equal(relation.valid, valid_before)


def test_owned_delta_finish_has_one_inplace_mask_sanitize_and_add_out():
    generator = torch.Generator().manual_seed(3407)
    features = torch.randn(2, 3, 4, 5, 8, generator=generator)
    delta = torch.randn(features.shape, generator=generator)
    _inject_special_values(features)
    _inject_special_values(delta)
    valid = torch.rand(2, 3, 4, 5, generator=generator) > 0.25

    expected_delta = torch.where(
        valid[..., None],
        delta,
        torch.zeros_like(delta),
    )
    expected_delta = torch.nan_to_num(
        expected_delta,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    expected = features + 1.0 * expected_delta

    class OperationLog(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            self.operations.append(str(func))
            return func(*args, **(kwargs or {}))

    owned_delta = delta.clone()
    owned_pointer = owned_delta.data_ptr()
    operation_log = OperationLog()
    with operation_log:
        actual = _mask_cast_and_add_owned_delta(features, owned_delta, valid)

    assert _raw_equal(actual, expected)
    assert actual.data_ptr() == owned_pointer == owned_delta.data_ptr()
    assert actual._base is None
    assert Counter(operation_log.operations) == Counter(
        {
            "aten.unsqueeze.default": 1,
            "aten.bitwise_not.default": 1,
            "aten.masked_fill_.Scalar": 1,
            "aten.nan_to_num_.default": 1,
            "aten.add.out": 1,
        }
    )


def test_owned_delta_rfl_falls_back_for_noncontiguous_input_and_hooks():
    module, contiguous, relation = _rfl_fixture(torch.float32)
    features = contiguous.transpose(1, 2)
    relation = RayFeatureRelation(
        ray=relation.ray.transpose(1, 2),
        valid=relation.valid.transpose(1, 2),
    )
    with torch.no_grad():
        assert not module._can_reuse_owned_delta(features, relation)
        expected = features + 1.0 * module(features, relation)
        actual = module._forward_lifted_owned_delta_inference(features, relation)
    assert _raw_equal(actual, expected)
    assert actual.stride() == expected.stride()

    module, features, relation = _rfl_fixture(torch.float32)
    calls = []
    handle = module.register_forward_hook(lambda *_args: calls.append("rfl"))
    try:
        with torch.no_grad():
            expected = features + 1.0 * module(features, relation)
            calls.clear()
            assert not module._can_reuse_owned_delta(features, relation)
            actual = module._forward_lifted_owned_delta_inference(
                features,
                relation,
            )
    finally:
        handle.remove()
    assert calls == ["rfl"]
    assert _raw_equal(actual, expected)


def test_owned_delta_rfl_falls_back_to_public_autograd():
    module, features, relation = _rfl_fixture(torch.float32)
    features = torch.randn_like(features).requires_grad_()
    assert not module._can_reuse_owned_delta(features, relation)
    actual = module._forward_lifted_owned_delta_inference(features, relation)
    assert actual.requires_grad
    actual.nan_to_num().square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_owned_delta_rfl_falls_back_under_tensor_interception_modes():
    module, features, relation = _rfl_fixture(torch.float32)
    with torch.no_grad():
        expected = features + 1.0 * module(features, relation)

    class CountDispatch(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.where_calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            if "aten.where" in str(func):
                self.where_calls += 1
            return func(*args, **(kwargs or {}))

    dispatch = CountDispatch()
    with torch.no_grad(), dispatch:
        assert not module._can_reuse_owned_delta(features, relation)
        dispatch_output = module._forward_lifted_owned_delta_inference(
            features,
            relation,
        )
    assert dispatch.where_calls == 2
    assert _raw_equal(dispatch_output, expected)

    class CountFunction(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.where_calls = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            del types
            if func is torch.where:
                self.where_calls += 1
            return func(*args, **(kwargs or {}))

    function = CountFunction()
    with torch.no_grad(), function:
        assert not module._can_reuse_owned_delta(features, relation)
        function_output = module._forward_lifted_owned_delta_inference(
            features,
            relation,
        )
    assert function.where_calls == 2
    assert _raw_equal(function_output, expected)


@pytest.mark.parametrize(
    ("owner", "attribute"),
    (
        (torch.overrides, "_is_torch_function_mode_enabled"),
        (_python_dispatch, "_is_in_non_infra_torch_dispatch_mode"),
    ),
)
def test_owned_delta_rfl_fails_closed_for_unknown_override_mode_state(
    monkeypatch,
    owner,
    attribute,
):
    module, features, relation = _rfl_fixture(torch.float32)
    with torch.no_grad():
        with monkeypatch.context() as patch:
            patch.delattr(owner, attribute, raising=False)
            assert not module._can_reuse_owned_delta(features, relation)
        with monkeypatch.context() as patch:
            patch.setattr(owner, attribute, lambda: 1)
            assert not module._can_reuse_owned_delta(features, relation)


@pytest.mark.parametrize(
    ("owner", "attribute"),
    (
        (torch.overrides, "_is_torch_function_mode_enabled"),
        (_python_dispatch, "_is_in_non_infra_torch_dispatch_mode"),
    ),
)
def test_owned_delta_rfl_fails_closed_when_override_mode_state_raises(
    monkeypatch,
    owner,
    attribute,
):
    module, features, relation = _rfl_fixture(torch.float32)

    def unavailable():
        raise RuntimeError("override mode state unavailable")

    monkeypatch.setattr(owner, attribute, unavailable)
    with torch.no_grad():
        assert not module._can_reuse_owned_delta(features, relation)


def test_owned_delta_rfl_fails_closed_for_unknown_autocast_state(monkeypatch):
    module, features, relation = _rfl_fixture(torch.float32)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_args: 1)
    with torch.no_grad():
        assert not module._can_reuse_owned_delta(features, relation)


def test_owned_delta_rfl_falls_back_for_tracing_and_compilation(monkeypatch):
    module, features, relation = _rfl_fixture(torch.float32)
    with torch.no_grad():
        expected = features + 1.0 * module(features, relation)

        monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
        assert not module._can_reuse_owned_delta(features, relation)
        traced = module._forward_lifted_owned_delta_inference(features, relation)
        monkeypatch.undo()

        monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)
        assert not module._can_reuse_owned_delta(features, relation)
        compiled_state = module._forward_lifted_owned_delta_inference(
            features,
            relation,
        )
    assert _raw_equal(traced, expected)
    assert _raw_equal(compiled_state, expected)


def test_public_rfl_fullgraph_compile_remains_raw_exact():
    module, features, relation = _rfl_fixture(torch.float32)

    def public_rfl(feature, ray, valid):
        return module(feature, RayFeatureRelation(ray=ray, valid=valid))

    with torch.no_grad():
        expected = public_rfl(features, relation.ray, relation.valid)
        compiled = torch.compile(public_rfl, backend="eager", fullgraph=True)
        actual = compiled(features, relation.ray, relation.valid)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`torch.jit.script_method` is deprecated.*",
            category=DeprecationWarning,
        )
        torch._dynamo.reset()
    assert _raw_equal(actual, expected)
    assert actual.stride() == expected.stride()


def _pointwise_fixture():
    generator = torch.Generator().manual_seed(3407)
    config = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    module = PointwiseROSA(config).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.1, 0.1, generator=generator)
    features = torch.randn(2, 3, 4, 13, 16, generator=generator)
    geometry = torch.randn(2, 3, 4, 13, 14, generator=generator)
    valid = torch.rand(2, 3, 4, 13, generator=generator) > 0.2
    ray = torch.randn(2, 3, 4, 13, 3, generator=generator)
    query = torch.randn(2, 3, 16, generator=generator)
    return (
        module,
        features,
        geometry,
        RayFeatureRelation(ray=ray, valid=valid),
        valid,
        query,
    )


@pytest.mark.parametrize("observation_usage", ("base_support", "candidate"))
def test_pointwise_private_rfl_buffer_route_is_raw_exact(observation_usage):
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    with torch.no_grad():
        assert module._can_reuse_owned_rfl_delta(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage,
        )
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage=observation_usage,
        )
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage=observation_usage,
        )

    assert _raw_equal(actual.object_feature, expected.object_feature)
    assert torch.equal(actual.supported, expected.supported)
    if observation_usage == "candidate":
        assert _raw_equal(
            actual.cross_camera_consistency,
            expected.cross_camera_consistency,
        )
    else:
        assert actual.cross_camera_consistency is None
        assert expected.cross_camera_consistency is None


def test_pointwise_private_rfl_buffer_route_preserves_module_hooks():
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="candidate",
        )

    calls = []
    rfl_handle = module.rfl.register_forward_hook(lambda *_args: calls.append("rfl"))
    pointwise_handle = module.register_forward_hook(
        lambda *_args: calls.append("pointwise")
    )
    try:
        with torch.no_grad():
            assert not module._can_reuse_owned_rfl_delta(
                features,
                geometry,
                relation,
                valid,
                query,
                "candidate",
            )
            actual = module._forward_compact_lean_inference(
                features,
                geometry,
                relation,
                valid,
                query,
                observation_usage="candidate",
            )
    finally:
        pointwise_handle.remove()
        rfl_handle.remove()
    assert calls == ["rfl", "pointwise"]
    assert _raw_equal(actual.object_feature, expected.object_feature)
    assert _raw_equal(
        actual.cross_camera_consistency,
        expected.cross_camera_consistency,
    )
