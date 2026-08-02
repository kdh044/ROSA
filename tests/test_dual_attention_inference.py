import math
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.benchmark import Timer

import rosa.rfl as rfl_module
from rosa.attention import PointwiseROSA
from rosa.config import ROSAConfig
from rosa.refiner import ROSARefiner
from rosa.types import RayFeatureRelation


def _pointwise_fixture(num_anchors: int = 2):
    generator = torch.Generator().manual_seed(7_101_080)
    module = PointwiseROSA(ROSAConfig.from_mapping({"profile": "fast"})).eval()
    features = torch.randn(
        1,
        num_anchors,
        4,
        13,
        256,
        generator=generator,
    )
    geometry = torch.randn(
        1,
        num_anchors,
        4,
        13,
        14,
        generator=generator,
    )
    ray = F.normalize(
        torch.randn(
            1,
            num_anchors,
            4,
            13,
            3,
            generator=generator,
        ),
        dim=-1,
    )
    valid = (
        torch.rand(
            1,
            num_anchors,
            4,
            13,
            generator=generator,
        )
        > 0.2
    )
    query = torch.randn(1, num_anchors, 256, generator=generator)
    return (
        module,
        features,
        geometry,
        RayFeatureRelation(ray=ray, valid=valid),
        valid,
        query,
    )


def _dual_guard(module, features, geometry, relation, valid, query) -> bool:
    return module._can_use_dual_camera_logits(
        features,
        geometry,
        relation,
        valid,
        query,
        "base_support",
    )


def _released_camera_logits(module, normalized, geometry, query):
    query_key = module.query_proj(query)[:, :, None, None, :]
    camera_key = module.key_proj(torch.cat((normalized, geometry), dim=-1))
    logits = (query_key * camera_key).sum(dim=-1)
    logits = logits / math.sqrt(module.cfg.hidden_dims)
    return logits + module.geometry_bias(geometry).squeeze(-1)


def _dual_camera_logits(module, normalized, geometry, query):
    query_key = module.query_proj(query)
    logits = module._dual_camera_logits_inference(
        normalized,
        geometry,
        query_key,
    )
    logits = logits / math.sqrt(module.cfg.hidden_dims)
    return logits + module.geometry_bias(geometry).squeeze(-1)


def _assert_encoding_equal(left, right) -> None:
    for name in left.__dataclass_fields__:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            assert torch.equal(left_value, right_value), name


def test_randomized_released_and_dual_camera_logits_have_bounded_fp32_error():
    maximum_absolute_error = 0.0
    maximum_scaled_error = 0.0
    maximum_l2_relative_error = 0.0
    bitwise_differences = 0

    for seed, num_anchors in ((1, 1), (2, 7), (3, 50), (4, 150)):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            module = PointwiseROSA(ROSAConfig.from_mapping({"profile": "fast"})).eval()
            generator = torch.Generator().manual_seed(seed + 100)
            normalized = torch.randn(
                1,
                num_anchors,
                4,
                13,
                256,
                generator=generator,
            )
            geometry = torch.randn(
                1,
                num_anchors,
                4,
                13,
                14,
                generator=generator,
            )
            query = torch.randn(1, num_anchors, 256, generator=generator)
            with torch.no_grad():
                released = _released_camera_logits(
                    module,
                    normalized,
                    geometry,
                    query,
                )
                dual = _dual_camera_logits(
                    module,
                    normalized,
                    geometry,
                    query,
                )

        error = (dual - released).abs()
        absolute = float(error.max())
        scaled = absolute / max(float(released.abs().max()), 1e-12)
        l2_relative = float(
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(released).clamp_min(1e-12)
        )
        maximum_absolute_error = max(maximum_absolute_error, absolute)
        maximum_scaled_error = max(maximum_scaled_error, scaled)
        maximum_l2_relative_error = max(maximum_l2_relative_error, l2_relative)
        bitwise_differences += int(not torch.equal(dual, released))

    assert bitwise_differences == 4
    assert maximum_absolute_error <= 2e-6
    assert maximum_scaled_error <= 2e-6
    assert maximum_l2_relative_error <= 2e-6


def test_dual_guard_is_released_fast_fp32_eager_only(monkeypatch):
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    with torch.no_grad():
        assert _dual_guard(module, features, geometry, relation, valid, query)

    assert not _dual_guard(module, features, geometry, relation, valid, query)

    module.train()
    with torch.no_grad():
        assert not _dual_guard(module, features, geometry, relation, valid, query)
    module.eval()

    noncontiguous = features.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert noncontiguous.shape == features.shape and not noncontiguous.is_contiguous()
    with torch.no_grad():
        assert not _dual_guard(
            module,
            noncontiguous,
            geometry,
            relation,
            valid,
            query,
        )
        assert not _dual_guard(
            module,
            features.double(),
            geometry,
            relation,
            valid,
            query,
        )

    class TensorSubclass(torch.Tensor):
        pass

    with torch.no_grad():
        assert not _dual_guard(
            module,
            features.as_subclass(TensorSubclass),
            geometry,
            relation,
            valid,
            query,
        )

    for owner, attribute in (
        (torch.jit, "is_scripting"),
        (torch.jit, "is_tracing"),
        (torch._dynamo, "is_compiling"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(owner, attribute, lambda: True)
            with torch.no_grad():
                assert not _dual_guard(
                    module,
                    features,
                    geometry,
                    relation,
                    valid,
                    query,
                )

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert not _dual_guard(module, features, geometry, relation, valid, query)

    class PassthroughFunction(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            del types
            return func(*args, **(kwargs or {}))

    class PassthroughDispatch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), PassthroughFunction():
        assert not _dual_guard(module, features, geometry, relation, valid, query)
    with torch.no_grad(), PassthroughDispatch():
        assert not _dual_guard(module, features, geometry, relation, valid, query)


def test_prepared_rfl_relation_preserves_dual_attention_output_raw_exact():
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    generator = torch.Generator().manual_seed(81_071)
    with torch.no_grad():
        module.rfl.output_proj.weight.normal_(generator=generator)
        prepared = rfl_module._prepare_ray_feature_relation(relation)
        assert _dual_guard(module, features, geometry, relation, valid, query)
        assert _dual_guard(module, features, geometry, prepared, valid, query)
        assert module.rfl._can_use_prepared_relation(features, prepared)
        expected = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            prepared,
            valid,
            query,
            observation_usage="base_support",
        )

    _assert_encoding_equal(actual, expected)


def test_private_entry_falls_back_to_public_autograd_and_module_override():
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    differentiable = features.clone().requires_grad_()
    output = module._forward_compact_lean_inference(
        differentiable,
        geometry,
        relation,
        valid,
        query,
        observation_usage="base_support",
    )
    assert output.object_feature.requires_grad
    output.object_feature.square().mean().backward()
    assert differentiable.grad is not None
    assert torch.isfinite(differentiable.grad).all()
    assert module.key_proj.weight.grad is not None

    module.zero_grad(set_to_none=True)
    calls = []
    released_forward = module.key_proj.forward

    def custom_forward(_self, value):
        calls.append(True)
        return released_forward(value)

    module.key_proj.forward = types.MethodType(custom_forward, module.key_proj)
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
        calls.clear()
        assert not _dual_guard(module, features, geometry, relation, valid, query)
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
    assert calls == [True]
    _assert_encoding_equal(actual, expected)


def test_private_entry_preserves_key_projection_hook_and_autocast_abi():
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
    handle = module.key_proj.register_forward_hook(lambda *_args: calls.append(True))
    try:
        with torch.no_grad():
            assert not _dual_guard(module, features, geometry, relation, valid, query)
            actual = module._forward_compact_lean_inference(
                features,
                geometry,
                relation,
                valid,
                query,
                observation_usage="candidate",
            )
    finally:
        handle.remove()
    assert calls == [True]
    _assert_encoding_equal(actual, expected)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected_autocast = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
        actual_autocast = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
    _assert_encoding_equal(actual_autocast, expected_autocast)


def test_dual_guard_preserves_module_call_override(monkeypatch):
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )

    calls = []
    released_call = torch.nn.Module.__call__

    def custom_call(self, *args, **kwargs):
        if self is module.key_proj:
            calls.append(True)
        return released_call(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "__call__", custom_call)
    with torch.no_grad():
        assert not _dual_guard(module, features, geometry, relation, valid, query)
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
    assert calls == [True]
    _assert_encoding_equal(actual, expected)


@pytest.mark.parametrize("helper_name", ("linear", "cat", "matmul", "einsum"))
def test_dual_guard_preserves_bypassed_and_new_public_helpers(
    helper_name,
    monkeypatch,
):
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )

    calls = []
    if helper_name == "linear":
        released = F.linear

        def hooked_linear(value, weight, bias=None):
            if weight is module.key_proj.weight:
                calls.append(True)
            return released(value, weight, bias)

        monkeypatch.setattr(F, "linear", hooked_linear)
    elif helper_name == "cat":
        released = torch.cat

        def hooked_cat(tensors, *args, **kwargs):
            values = tuple(tensors)
            if (
                len(values) == 2
                and values[0].shape[-1] == module.cfg.embed_dims
                and values[1].shape[-1] == module.geometry_dims
            ):
                calls.append(True)
            return released(values, *args, **kwargs)

        monkeypatch.setattr(torch, "cat", hooked_cat)
    else:

        def must_not_run(*_args, **_kwargs):
            raise AssertionError(f"overridden torch.{helper_name} executed")

        monkeypatch.setattr(torch, helper_name, must_not_run)

    with torch.no_grad():
        assert not _dual_guard(module, features, geometry, relation, valid, query)
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
    if helper_name in ("linear", "cat"):
        assert calls == [True]
    _assert_encoding_equal(actual, expected)


def test_dual_attention_composes_with_normalize_storage_fallback(monkeypatch):
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    relation = RayFeatureRelation(ray=relation.ray, valid=relation.valid.clone())
    with torch.no_grad():
        assert _dual_guard(module, features, geometry, relation, valid, query)
        assert module._can_reuse_owned_pointwise_storage(
            features,
            geometry,
            relation,
            valid,
            query,
            "candidate",
        )
        expected = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="candidate",
        )

    released_normalize = F.normalize
    observed_out = []

    def hooked_normalize(*args, **kwargs):
        observed_out.append(kwargs.get("out"))
        return released_normalize(*args, **kwargs)

    monkeypatch.setattr(F, "normalize", hooked_normalize)
    with torch.no_grad():
        # The normalization-buffer optimization must fail closed while the
        # independent dual-attention optimization remains admissible.
        assert not module._can_reuse_owned_pointwise_storage(
            features,
            geometry,
            relation,
            valid,
            query,
            "candidate",
        )
        assert _dual_guard(module, features, geometry, relation, valid, query)
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="candidate",
        )

    assert observed_out
    assert all(value is None for value in observed_out)
    _assert_encoding_equal(actual, expected)


def test_private_entry_does_not_invoke_overridden_pointwise_reuse_guard():
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("overridden Pointwise reuse guard executed")

    module._can_reuse_owned_rfl_delta = types.MethodType(must_not_run, module)
    with torch.no_grad():
        assert not _dual_guard(module, features, geometry, relation, valid, query)
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
    _assert_encoding_equal(actual, expected)


def test_private_entry_seals_rfl_reuse_guard_before_invoking_it():
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("overridden RFL reuse guard executed")

    module.rfl._can_reuse_owned_delta = types.MethodType(must_not_run, module.rfl)
    with torch.no_grad():
        assert not _dual_guard(module, features, geometry, relation, valid, query)
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
    _assert_encoding_equal(actual, expected)


def test_private_entry_seals_plain_eager_predicate_and_preserves_autograd(
    monkeypatch,
):
    module, features, geometry, relation, valid, query = _pointwise_fixture()

    def must_not_run():
        raise AssertionError("overridden plain-eager predicate executed")

    monkeypatch.setattr(
        rfl_module,
        "_plain_eager_no_grad_execution",
        must_not_run,
    )
    expected = module(
        features,
        geometry,
        relation,
        valid,
        query,
        observation_usage="base_support",
    )
    assert not _dual_guard(module, features, geometry, relation, valid, query)
    actual = module._forward_compact_lean_inference(
        features,
        geometry,
        relation,
        valid,
        query,
        observation_usage="base_support",
    )
    _assert_encoding_equal(actual, expected)
    assert actual.object_feature.requires_grad
    actual.object_feature.square().mean().backward()
    assert module.key_proj.weight.grad is not None
    assert torch.isfinite(module.key_proj.weight.grad).all()


def test_private_entry_rejects_overridden_dual_methods(monkeypatch):
    module, features, geometry, relation, valid, query = _pointwise_fixture()
    with torch.no_grad():
        expected = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("overridden dual kernel executed")

    monkeypatch.setattr(
        PointwiseROSA,
        "_can_use_dual_camera_logits",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        PointwiseROSA,
        "_dual_camera_logits_inference",
        must_not_run,
    )
    with torch.no_grad():
        actual = module._forward_compact_lean_inference(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )
    _assert_encoding_equal(actual, expected)


def _refiner_fixture():
    config = ROSAConfig.from_mapping({"profile": "fast"})
    stage = ROSARefiner(config, "ocam_fisheye").eval()
    generator = torch.Generator().manual_seed(7_101_080)
    with torch.no_grad():
        for parameter in stage.parameters():
            parameter.uniform_(-0.05, 0.05, generator=generator)

    batch, queries, cameras, dims = 1, 2, 4, config.embed_dims
    anchor = torch.zeros(batch, queries, 11)
    anchor[..., 0] = torch.tensor((1.0, 2.0))
    anchor[..., 2] = -5.0
    anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5)).log()
    anchor[..., 7] = 1.0
    query = torch.randn(batch, queries, dims, generator=generator)
    fpn = [
        torch.randn(batch, cameras, dims, size, size, generator=generator)
        for size in (16, 8, 4, 2)
    ]
    lidar2cam = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, cameras, 1, 1)
    intrinsic = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, cameras, 1, 1)
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    polynomial = torch.zeros(batch, cameras, 5)
    polynomial[..., 1] = -150.0
    metadata = {
        "lidar2cam": lidar2cam,
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "image_wh": torch.full((batch, cameras, 2), 800.0),
        "ocam_lens_fov_deg": 205.0,
    }

    def keypoint_generator(box, _query):
        scale = box.new_tensor(
            (
                (0, 0, 0),
                (0.45, 0, 0),
                (-0.45, 0, 0),
                (0, 0.45, 0),
                (0, -0.45, 0),
                (0, 0, 0.45),
                (0, 0, -0.45),
                (0.2, 0.2, 0.2),
                (-0.2, 0.2, 0.2),
                (0.2, -0.2, 0.2),
                (-0.2, -0.2, 0.2),
                (0.2, 0.2, -0.2),
                (-0.2, -0.2, -0.2),
            )
        )
        return (
            box[..., :3, None].transpose(-1, -2)
            + scale[None, None] * box[..., 3:6].exp()[..., None, :]
        )

    return stage, anchor, query, fpn, metadata, keypoint_generator


def test_dual_attention_end_to_end_final_anchor_error_is_bounded(monkeypatch):
    stage, anchor, query, fpn, metadata, generator = _refiner_fixture()
    with monkeypatch.context() as patch:
        patch.setattr(
            PointwiseROSA,
            "_can_use_dual_camera_logits",
            lambda *_args, **_kwargs: False,
        )
        with torch.no_grad():
            released = stage(
                anchor,
                query,
                fpn,
                metadata,
                generator,
                materialize_diagnostics=False,
                return_final_anchor_only=True,
            )

    calls = 0
    dual_code = PointwiseROSA._dual_camera_logits_inference.__code__

    def count_dual(frame, event, _argument):
        nonlocal calls
        if event == "call" and frame.f_code is dual_code:
            calls += 1

    previous_profile = sys.getprofile()
    sys.setprofile(count_dual)
    try:
        with torch.no_grad():
            dual = stage(
                anchor,
                query,
                fpn,
                metadata,
                generator,
                materialize_diagnostics=False,
                return_final_anchor_only=True,
            )
    finally:
        sys.setprofile(previous_profile)

    error = (dual - released).abs()
    maximum_absolute_error = float(error.max())
    maximum_scaled_error = maximum_absolute_error / max(
        float(released.abs().max()),
        1e-12,
    )
    l2_relative_error = float(
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(released).clamp_min(1e-12)
    )
    assert calls == 2
    assert torch.isfinite(dual).all()
    assert maximum_absolute_error <= 1e-5
    assert maximum_scaled_error <= 2e-6
    assert l2_relative_error <= 2e-6


def test_dual_camera_kernel_reduces_cpu_work_and_runtime():
    module, normalized, geometry, _relation, _valid, query = _pointwise_fixture(50)
    with torch.no_grad():
        query_key = module.query_proj(query)

    @torch.no_grad()
    def released_kernel():
        expanded_query = query_key[:, :, None, None, :]
        camera_key = module.key_proj(torch.cat((normalized, geometry), dim=-1))
        return (expanded_query * camera_key).sum(dim=-1)

    @torch.no_grad()
    def dual_kernel():
        return module._dual_camera_logits_inference(
            normalized,
            geometry,
            query_key,
        )

    released_kernel()
    dual_kernel()
    released_time = Timer(
        stmt="released_kernel()",
        globals={"released_kernel": released_kernel},
        num_threads=1,
    ).blocked_autorange(min_run_time=0.2)
    dual_time = Timer(
        stmt="dual_kernel()",
        globals={"dual_kernel": dual_kernel},
        num_threads=1,
    ).blocked_autorange(min_run_time=0.2)

    batch, anchors, cameras, points = normalized.shape[:4]
    observations = batch * anchors * cameras * points
    queries = batch * anchors
    dims = module.cfg.embed_dims
    geometry_dims = module.geometry_dims
    hidden = module.cfg.hidden_dims
    common_query_macs = queries * dims * hidden
    released_macs = (
        common_query_macs
        + observations * (dims + geometry_dims) * hidden
        + observations * hidden
    )
    dual_macs = (
        common_query_macs
        + queries * hidden * (dims + geometry_dims)
        + observations * (dims + geometry_dims)
    )
    assert released_macs == 45_913_600
    assert dual_macs == 2_385_200
    assert released_macs / dual_macs > 19.0
    assert observations * hidden == 166_400
    assert queries * (dims + geometry_dims) == 13_500
    assert released_time.median > 2.0 * dual_time.median
