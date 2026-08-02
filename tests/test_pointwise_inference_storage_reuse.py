from collections import Counter
import types

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _python_dispatch
from torch.utils._python_dispatch import TorchDispatchMode

from rosa.attention import (
    PointwiseROSA,
    _is_owned_plain_fp32_tensor,
    _mean_pairwise_camera_cosine_owned_inference,
    mean_pairwise_camera_cosine,
)
from rosa.config import ROSAConfig
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


def _inject_fp32_special_values(value: torch.Tensor) -> None:
    patterns = torch.tensor(
        (
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7FFFFFFF,
        ),
        dtype=torch.uint32,
    )
    value.view(torch.uint32).reshape(-1)[: patterns.numel()].copy_(patterns)


def _fixture(*, special_values: bool = False):
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
    if special_values:
        _inject_fp32_special_values(features)
        _inject_fp32_special_values(ray)
        # Exercise fully masked camera and keypoint reductions in the same call.
        valid[0, 0] = False
        valid[1, 1, :, 0] = False
    relation = RayFeatureRelation(ray=ray, valid=valid.clone())
    return module, features, geometry, relation, valid, query


def _snapshot_tensors(module, features, geometry, relation, valid, query):
    inputs = {
        "features": features.clone(),
        "geometry": geometry.clone(),
        "relation.ray": relation.ray.clone(),
        "relation.valid": relation.valid.clone(),
        "valid": valid.clone(),
        "query": query.clone(),
    }
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
    return inputs, parameters


def _assert_snapshots_unchanged(
    module,
    features,
    geometry,
    relation,
    valid,
    query,
    inputs,
    parameters,
):
    observed_inputs = {
        "features": features,
        "geometry": geometry,
        "relation.ray": relation.ray,
        "relation.valid": relation.valid,
        "valid": valid,
        "query": query,
    }
    for name, before in inputs.items():
        assert _raw_equal(observed_inputs[name], before), name
    for name, parameter in module.named_parameters():
        assert _raw_equal(parameter, parameters[name]), name


def test_owned_tensor_check_requires_independent_plain_storage():
    source = torch.randn(2, 3, 4)
    owned = source.clone()
    parameter = torch.nn.Parameter(torch.randn(4))
    assert _is_owned_plain_fp32_tensor(
        owned,
        shape=source.shape,
        device=source.device,
        protected=(source, parameter),
    )

    detached_alias = source.detach()
    assert detached_alias._base is None
    assert not _is_owned_plain_fp32_tensor(
        detached_alias,
        shape=source.shape,
        device=source.device,
        protected=(source,),
    )
    noncontiguous = source.transpose(1, 2)
    assert not _is_owned_plain_fp32_tensor(
        noncontiguous,
        shape=noncontiguous.shape,
        device=source.device,
        protected=(source,),
    )


def test_owned_camera_consistency_is_raw_exact_and_consumes_its_input():
    generator = torch.Generator().manual_seed(7101080)
    normalized = torch.randn(2, 3, 4, 13, 8, generator=generator)
    _inject_fp32_special_values(normalized)
    valid = torch.rand(2, 3, 4, 13, generator=generator) > 0.25
    valid[0, 0] = False
    expected = mean_pairwise_camera_cosine(normalized.clone(), valid)

    consumed = normalized.clone()
    pointer = consumed.data_ptr()
    actual = _mean_pairwise_camera_cosine_owned_inference(consumed, valid)

    assert _raw_equal(actual, expected)
    assert actual.stride() == expected.stride()
    assert consumed.data_ptr() == pointer
    expected_storage = normalized.clone().permute(0, 1, 3, 2, 4)
    expected_storage.mul_(
        valid.permute(0, 1, 3, 2)[..., None].to(expected_storage.dtype)
    )
    expected_storage.square_()
    assert _raw_equal(consumed, expected_storage.permute(0, 1, 3, 2, 4))


def test_owned_camera_consistency_uses_only_owned_mask_and_square_outputs():
    values = torch.randn(1, 2, 4, 13, 8)
    valid = torch.rand(1, 2, 4, 13) > 0.2

    class OperationLog(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            self.operations.append(str(func))
            return func(*args, **(kwargs or {}))

    log = OperationLog()
    with log:
        _mean_pairwise_camera_cosine_owned_inference(values, valid)
    counts = Counter(log.operations)
    assert counts["aten.mul_.Tensor"] == 1
    assert counts["aten.pow_.Scalar"] == 2
    assert counts["aten.mul.out"] == 0


@pytest.mark.parametrize("observation_usage", ("base_support", "candidate"))
@pytest.mark.parametrize("special_values", (False, True))
def test_private_pointwise_storage_route_is_raw_exact_and_immutable(
    observation_usage,
    special_values,
):
    module, features, geometry, relation, valid, query = _fixture(
        special_values=special_values
    )
    inputs, parameters = _snapshot_tensors(
        module, features, geometry, relation, valid, query
    )

    with torch.no_grad():
        assert module._can_reuse_owned_pointwise_storage(
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
    assert actual.object_feature.stride() == expected.object_feature.stride()
    assert actual.object_feature._base is None
    assert not torch._C._is_alias_of(actual.object_feature, features)
    assert not torch._C._is_alias_of(actual.object_feature, query)
    assert torch.equal(actual.supported, expected.supported)
    assert actual.supported.stride() == expected.supported.stride()
    assert actual.point_features is None
    assert actual.camera_weights is None
    if observation_usage == "candidate":
        assert _raw_equal(
            actual.cross_camera_consistency,
            expected.cross_camera_consistency,
        )
        assert (
            actual.cross_camera_consistency.stride()
            == expected.cross_camera_consistency.stride()
        )
        assert actual.cross_camera_consistency._base is None
        assert not torch._C._is_alias_of(
            actual.cross_camera_consistency,
            features,
        )
    else:
        assert actual.cross_camera_consistency is None
        assert expected.cross_camera_consistency is None
    _assert_snapshots_unchanged(
        module,
        features,
        geometry,
        relation,
        valid,
        query,
        inputs,
        parameters,
    )


@pytest.mark.parametrize(
    ("observation_usage", "expected_out_mul", "expected_mask_mul"),
    (("base_support", 4, 0), ("candidate", 3, 1)),
)
def test_private_pointwise_impl_reuses_only_the_requested_outputs(
    observation_usage,
    expected_out_mul,
    expected_mask_mul,
):
    module, features, geometry, relation, valid, query = _fixture()

    class OperationLog(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            self.operations.append(str(func))
            return func(*args, **(kwargs or {}))

    log = OperationLog()
    with torch.no_grad(), log:
        module._forward_impl(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage=observation_usage,
            reuse_owned_rfl_delta=False,
            reuse_owned_pointwise_storage=True,
        )
    counts = Counter(log.operations)
    assert counts["aten.mul.out"] == expected_out_mul
    assert counts["aten.mul_.Tensor"] == expected_mask_mul
    assert counts["aten.nan_to_num_.default"] == expected_mask_mul
    assert counts["aten.pow_.Scalar"] == 2 * expected_mask_mul


def test_pointwise_storage_guard_rejects_dtype_layout_alias_and_module_drift():
    module, features, geometry, relation, valid, query = _fixture()

    def can_reuse(
        feature=features,
        descriptor=geometry,
        compact_relation=relation,
        mask=valid,
        object_query=query,
    ):
        with torch.no_grad():
            return module._can_reuse_owned_pointwise_storage(
                feature,
                descriptor,
                compact_relation,
                mask,
                object_query,
                "candidate",
            )

    assert can_reuse()
    assert not can_reuse(feature=features.double())
    query_storage = torch.empty(query.shape[:-1] + (query.shape[-1] * 2,))
    noncontiguous_query = query_storage[..., ::2]
    noncontiguous_query.copy_(query)
    assert not noncontiguous_query.is_contiguous()
    assert not can_reuse(object_query=noncontiguous_query)
    assert not can_reuse(
        compact_relation=RayFeatureRelation(
            ray=geometry[..., :3],
            valid=relation.valid,
        )
    )
    assert not can_reuse(
        compact_relation=RayFeatureRelation(
            ray=relation.ray,
            valid=valid,
        )
    )

    module.geometry_bias[1] = torch.nn.SiLU(inplace=True)
    assert not can_reuse()


def test_pointwise_storage_guard_rejects_forward_override_and_compiled_child():
    module, features, geometry, relation, valid, query = _fixture()

    released_key_forward = module.key_proj.forward

    def forwarded(self, value):
        return released_key_forward(value)

    module.key_proj.forward = types.MethodType(forwarded, module.key_proj)
    with torch.no_grad():
        assert not module._can_reuse_owned_pointwise_storage(
            features, geometry, relation, valid, query, "candidate"
        )
    del module.key_proj.forward

    module.point_embedding._compiled_call_impl = lambda *args, **kwargs: None
    with torch.no_grad():
        assert not module._can_reuse_owned_pointwise_storage(
            features, geometry, relation, valid, query, "candidate"
        )


def test_pointwise_private_route_falls_back_with_child_and_global_hooks():
    module, features, geometry, relation, valid, query = _fixture()
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="candidate",
        )

    local_calls = []
    local_handle = module.point_embedding.register_forward_hook(
        lambda *_args: local_calls.append("embedding")
    )
    try:
        with torch.no_grad():
            assert not module._can_reuse_owned_pointwise_storage(
                features, geometry, relation, valid, query, "candidate"
            )
            local_actual = module._forward_compact_lean_inference(
                features,
                geometry,
                relation,
                valid,
                query,
                observation_usage="candidate",
            )
    finally:
        local_handle.remove()
    assert local_calls == ["embedding"]
    assert _raw_equal(local_actual.object_feature, expected.object_feature)
    assert _raw_equal(
        local_actual.cross_camera_consistency,
        expected.cross_camera_consistency,
    )

    global_calls = []
    global_handle = torch.nn.modules.module.register_module_forward_hook(
        lambda observed, _args, _output: global_calls.append(type(observed))
    )
    try:
        with torch.no_grad():
            assert not module._can_reuse_owned_pointwise_storage(
                features, geometry, relation, valid, query, "candidate"
            )
            global_actual = module._forward_compact_lean_inference(
                features,
                geometry,
                relation,
                valid,
                query,
                observation_usage="candidate",
            )
    finally:
        global_handle.remove()
    assert PointwiseROSA in global_calls
    assert _raw_equal(global_actual.object_feature, expected.object_feature)
    assert _raw_equal(
        global_actual.cross_camera_consistency,
        expected.cross_camera_consistency,
    )


def test_pointwise_private_route_falls_back_under_tensor_interception_modes():
    module, features, geometry, relation, valid, query = _fixture()
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="candidate",
        )

    class PassthroughDispatch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), PassthroughDispatch():
        assert not module._can_reuse_owned_pointwise_storage(
            features, geometry, relation, valid, query, "candidate"
        )
        dispatch_actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="candidate",
        )
    assert _raw_equal(dispatch_actual.object_feature, expected.object_feature)
    assert _raw_equal(
        dispatch_actual.cross_camera_consistency,
        expected.cross_camera_consistency,
    )

    class PassthroughFunction(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            del types
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), PassthroughFunction():
        assert not module._can_reuse_owned_pointwise_storage(
            features, geometry, relation, valid, query, "candidate"
        )
        function_actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="candidate",
        )
    assert _raw_equal(function_actual.object_feature, expected.object_feature)
    assert _raw_equal(
        function_actual.cross_camera_consistency,
        expected.cross_camera_consistency,
    )


@pytest.mark.parametrize(
    ("owner", "attribute"),
    (
        (torch.overrides, "_is_torch_function_mode_enabled"),
        (_python_dispatch, "_is_in_non_infra_torch_dispatch_mode"),
    ),
)
def test_pointwise_storage_guard_fails_closed_for_unknown_mode_state(
    monkeypatch,
    owner,
    attribute,
):
    module, features, geometry, relation, valid, query = _fixture()
    with monkeypatch.context() as patch:
        patch.delattr(owner, attribute, raising=False)
        with torch.no_grad():
            assert not module._can_reuse_owned_pointwise_storage(
                features, geometry, relation, valid, query, "candidate"
            )
    with monkeypatch.context() as patch:
        patch.setattr(owner, attribute, lambda: 1)
        with torch.no_grad():
            assert not module._can_reuse_owned_pointwise_storage(
                features, geometry, relation, valid, query, "candidate"
            )


def test_pointwise_storage_guard_fails_closed_for_autocast_trace_and_compile(
    monkeypatch,
):
    module, features, geometry, relation, valid, query = _fixture()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert not module._can_reuse_owned_pointwise_storage(
            features, geometry, relation, valid, query, "candidate"
        )

    with torch.no_grad(), monkeypatch.context() as patch:
        patch.setattr(torch.jit, "is_tracing", lambda: True)
        assert not module._can_reuse_owned_pointwise_storage(
            features, geometry, relation, valid, query, "candidate"
        )
    with torch.no_grad(), monkeypatch.context() as patch:
        patch.setattr(torch._dynamo, "is_compiling", lambda: True)
        assert not module._can_reuse_owned_pointwise_storage(
            features, geometry, relation, valid, query, "candidate"
        )


def test_public_full_and_training_paths_retain_their_contracts():
    module, features, geometry, relation, valid, query = _fixture()
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="full",
        )
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="full",
        )
    assert _raw_equal(actual.object_feature, expected.object_feature)
    assert _raw_equal(actual.point_features, expected.point_features)
    assert _raw_equal(actual.camera_weights, expected.camera_weights)
    assert _raw_equal(
        actual.cross_camera_consistency,
        expected.cross_camera_consistency,
    )

    module.train()
    train_features = features.clone().requires_grad_()
    output = module(
        train_features,
        geometry,
        relation,
        valid,
        query,
        observation_usage="candidate",
    )
    output.object_feature.nan_to_num().square().mean().backward()
    assert train_features.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())
