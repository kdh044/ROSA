import types

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from rosa.observation import (
    _can_use_prepared_multiscale,
    sample_multiscale_features,
)
from rosa.sampling import (
    DeterministicBilinearSampler,
    FusedBilinearSampler,
)


def _released_multiscale(fpn, uv, query, sampler, level_head):
    sample_uv = uv.unsqueeze(4)
    sampled = [sampler(feature, sample_uv).squeeze(4) for feature in fpn]
    levels = torch.stack(sampled, dim=2)
    weights = torch.softmax(level_head(query).float(), dim=-1).to(levels.dtype)
    return (levels * weights[:, :, :, None, None, None]).sum(dim=2)


def _raw_equal(left, right):
    integer_dtype = torch.int32 if left.dtype == torch.float32 else torch.int64
    return torch.equal(left.view(integer_dtype), right.view(integer_dtype))


def _prepared_fixture(
    *,
    queries=5,
    channels=8,
    dtype=torch.float32,
):
    generator = torch.Generator().manual_seed(3407)
    fpn = tuple(
        torch.randn(
            1,
            4,
            channels,
            size,
            size,
            generator=generator,
            dtype=dtype,
        )
        for size in (8, 4, 2, 1)
    )
    uv = torch.rand(
        1,
        queries,
        4,
        13,
        2,
        generator=generator,
        dtype=dtype,
    )
    query = torch.randn(
        1,
        queries,
        channels,
        generator=generator,
        dtype=dtype,
    )
    level_head = torch.nn.Linear(channels, 4, dtype=dtype)
    sampler = FusedBilinearSampler().eval()
    return fpn, uv, query, level_head, sampler


def test_fused_sampler_matches_the_release_four_gather_sampler():
    torch.manual_seed(11)
    release_feature = torch.randn(1, 2, 3, 5, 7, requires_grad=True)
    fused_feature = release_feature.detach().clone().requires_grad_(True)
    uv = torch.rand(1, 4, 2, 1, 3, 2)
    uv[0, 0, 0, 0, 0] = torch.tensor((0.0, 1.0))
    uv[0, 1, 1, 0, 1] = torch.tensor((-0.2, 1.2))
    uv[0, 2, 0, 0, 2] = torch.tensor((float("nan"), 0.5))

    release = DeterministicBilinearSampler()(release_feature, uv)
    fused = FusedBilinearSampler()(fused_feature, uv)

    torch.testing.assert_close(fused, release, atol=2e-6, rtol=2e-6)
    release.square().sum().backward()
    fused.square().sum().backward()
    torch.testing.assert_close(
        fused_feature.grad,
        release_feature.grad,
        atol=3e-6,
        rtol=3e-6,
    )


def test_fused_sampler_falls_back_safely_for_mixed_precision_inputs():
    feature = torch.randn(1, 2, 3, 5, 7, dtype=torch.float16)
    uv = torch.rand(1, 4, 2, 1, 3, 2, dtype=torch.float32)
    expected = DeterministicBilinearSampler()(feature, uv)
    actual = FusedBilinearSampler()(feature, uv)
    assert torch.equal(actual, expected)


def test_fused_sampler_uses_deterministic_path_when_inputs_need_gradients():
    feature = torch.randn(1, 2, 3, 5, 7, requires_grad=True)
    uv = torch.rand(1, 4, 2, 1, 3, 2)
    sampler = FusedBilinearSampler()
    expected = sampler.fallback(feature, uv)
    actual = sampler(feature, uv)
    assert torch.equal(actual, expected)


def test_eval_multiscale_storage_reuse_is_raw_exact():
    generator = torch.Generator().manual_seed(3407)
    fpn = [
        torch.randn(1, 2, 3, size, size, generator=generator)
        for size in (8, 4, 2, 1)
    ]
    uv = torch.rand(1, 5, 2, 3, 2, generator=generator)
    query = torch.randn(1, 5, 3, generator=generator)
    level_head = torch.nn.Linear(3, 4)
    sampler = DeterministicBilinearSampler()

    with torch.no_grad():
        expected = _released_multiscale(fpn, uv, query, sampler, level_head)
        actual = sample_multiscale_features(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )

    assert torch.equal(actual, expected)


def test_eval_multiscale_storage_reuse_supports_fullgraph_compile():
    generator = torch.Generator().manual_seed(7101080)
    fpn = tuple(
        torch.randn(1, 2, 8, size, size, generator=generator)
        for size in (8, 4, 2, 1)
    )
    uv = torch.rand(1, 3, 2, 13, 2, generator=generator)
    query = torch.randn(1, 3, 8, generator=generator)
    sampler = FusedBilinearSampler()
    level_head = torch.nn.Linear(8, 4)

    def fuse(query_value, uv_value, *features):
        return sample_multiscale_features(
            features,
            uv_value,
            query_value,
            sampler,
            level_head,
        )

    with torch.no_grad():
        expected = fuse(query, uv, *fpn)
        compiled = torch.compile(fuse, backend="eager", fullgraph=True)
        actual = compiled(query, uv, *fpn)

    assert torch.equal(actual, expected)


def test_training_multiscale_path_preserves_output_and_gradients():
    generator = torch.Generator().manual_seed(7101080)
    parent_fpn = [
        torch.randn(
            1,
            2,
            3,
            size,
            size,
            generator=generator,
            requires_grad=True,
        )
        for size in (8, 4, 2, 1)
    ]
    candidate_fpn = [
        value.detach().clone().requires_grad_() for value in parent_fpn
    ]
    uv = torch.rand(1, 5, 2, 3, 2, generator=generator)
    parent_query = torch.randn(
        1,
        5,
        3,
        generator=generator,
        requires_grad=True,
    )
    candidate_query = parent_query.detach().clone().requires_grad_()
    parent_head = torch.nn.Linear(3, 4)
    candidate_head = torch.nn.Linear(3, 4)
    candidate_head.load_state_dict(parent_head.state_dict())
    sampler = DeterministicBilinearSampler()

    expected = _released_multiscale(
        parent_fpn,
        uv,
        parent_query,
        sampler,
        parent_head,
    )
    actual = sample_multiscale_features(
        candidate_fpn,
        uv,
        candidate_query,
        sampler,
        candidate_head,
    )
    assert torch.equal(actual, expected)

    expected.sum().backward()
    actual.sum().backward()
    for candidate, parent in zip(candidate_fpn, parent_fpn):
        assert torch.equal(candidate.grad, parent.grad)
    assert torch.equal(candidate_query.grad, parent_query.grad)
    for candidate, parent in zip(candidate_head.parameters(), parent_head.parameters()):
        assert torch.equal(candidate.grad, parent.grad)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_prepared_grid_and_deferred_layout_are_raw_exact(dtype):
    fpn, uv, query, level_head, sampler = _prepared_fixture(dtype=dtype)
    uv[0, 0, 0, 0] = torch.tensor((float("nan"), 0.5), dtype=dtype)
    uv[0, 1, 1, 1] = torch.tensor((float("inf"), float("-inf")), dtype=dtype)

    with torch.no_grad():
        assert _can_use_prepared_multiscale(fpn, uv, sampler)
        expected = _released_multiscale(fpn, uv, query, sampler, level_head)
        actual = sample_multiscale_features(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
        prepared = sampler._sample_multiscale_prepared(fpn, uv.unsqueeze(4))
        prepared_stack = torch.stack(
            tuple(value.squeeze(4) for value in prepared),
            dim=2,
        )
        released_stack = torch.stack(
            tuple(sampler(feature, uv.unsqueeze(4)).squeeze(4) for feature in fpn),
            dim=2,
        )

    assert all(not value.is_contiguous() for value in prepared)
    assert prepared_stack.stride() == released_stack.stride()
    assert _raw_equal(prepared_stack, released_stack)
    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)


def test_prepared_grid_is_raw_exact_at_fast_candidate_shape():
    fpn, uv, query, level_head, sampler = _prepared_fixture(
        queries=150,
        channels=256,
    )

    with torch.no_grad():
        expected = _released_multiscale(fpn, uv, query, sampler, level_head)
        actual = sample_multiscale_features(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )

    assert actual.shape == (1, 150, 4, 13, 256)
    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)


def test_prepared_grid_reduces_fx_grid_work_without_changing_reduction():
    fpn, uv, query, level_head, candidate_sampler = _prepared_fixture()
    parent_sampler = FusedBilinearSampler().train()

    def capture(sampler):
        graphs = []

        def backend(graph_module, _inputs):
            graphs.append(graph_module)
            return graph_module.forward

        def fuse(query_value, uv_value, *features):
            return sample_multiscale_features(
                features,
                uv_value,
                query_value,
                sampler,
                level_head,
            )

        compiled = torch.compile(fuse, backend=backend, fullgraph=True)
        try:
            with torch.no_grad():
                output = compiled(query, uv, *fpn)
        finally:
            torch._dynamo.reset()
        assert len(graphs) == 1
        return output, graphs[0]

    parent, parent_graph = capture(parent_sampler)
    candidate, candidate_graph = capture(candidate_sampler)

    def operation_counts(graph):
        nodes = tuple(graph.graph.nodes)
        return {
            "nan_to_num": sum(
                node.op == "call_function" and node.target is torch.nan_to_num
                for node in nodes
            ),
            "grid_sample": sum(
                node.op == "call_function"
                and node.target is torch.nn.functional.grid_sample
                for node in nodes
            ),
            "contiguous": sum(
                node.op == "call_method" and node.target == "contiguous"
                for node in nodes
            ),
            "stack": sum(
                node.op == "call_function" and node.target is torch.stack
                for node in nodes
            ),
            "sum": sum(
                node.op == "call_method" and node.target == "sum" for node in nodes
            ),
        }

    assert operation_counts(parent_graph) == {
        "nan_to_num": 4,
        "grid_sample": 4,
        "contiguous": 4,
        "stack": 1,
        "sum": 1,
    }
    assert operation_counts(candidate_graph) == {
        "nan_to_num": 1,
        "grid_sample": 4,
        "contiguous": 0,
        "stack": 1,
        "sum": 1,
    }
    assert parent.stride() == candidate.stride()
    assert _raw_equal(candidate, parent)


def test_prepared_grid_fails_closed_for_sampler_hooks_and_overrides():
    fpn, uv, query, level_head, sampler = _prepared_fixture()
    with torch.no_grad():
        expected = _released_multiscale(fpn, uv, query, sampler, level_head)

    calls = []
    handle = sampler.register_forward_hook(lambda *_args: calls.append("hook"))
    try:
        with torch.no_grad():
            actual = sample_multiscale_features(
                fpn,
                uv,
                query,
                sampler,
                level_head,
            )
    finally:
        handle.remove()
    assert calls == ["hook"] * 4
    assert _raw_equal(actual, expected)

    released_forward = sampler.forward

    def overridden(self, feature, sample_uv):
        calls.append("override")
        return released_forward(feature, sample_uv)

    sampler.forward = types.MethodType(overridden, sampler)
    calls.clear()
    with torch.no_grad():
        actual = sample_multiscale_features(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
    assert calls == ["override"] * 4
    assert _raw_equal(actual, expected)


def test_prepared_grid_fails_closed_for_global_hooks_and_subclasses():
    fpn, uv, query, level_head, released_sampler = _prepared_fixture()

    class SamplerSubclass(FusedBilinearSampler):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, feature, sample_uv):
            self.calls += 1
            return super().forward(feature, sample_uv)

    sampler = SamplerSubclass().eval()
    with torch.no_grad():
        expected = _released_multiscale(
            fpn,
            uv,
            query,
            released_sampler,
            level_head,
        )
        actual = sample_multiscale_features(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
    assert sampler.calls == 4
    assert _raw_equal(actual, expected)

    observed = []

    def global_hook(module, _args, _output):
        if module is released_sampler:
            observed.append(module)

    handle = torch.nn.modules.module.register_module_forward_hook(global_hook)
    try:
        with torch.no_grad():
            actual = sample_multiscale_features(
                fpn,
                uv,
                query,
                released_sampler,
                level_head,
            )
    finally:
        handle.remove()
    assert observed == [released_sampler] * 4
    assert _raw_equal(actual, expected)


def test_prepared_grid_fails_closed_for_grad_mixed_dtype_and_tensor_subclass():
    fpn, uv, _query, _level_head, sampler = _prepared_fixture()

    with torch.no_grad():
        assert _can_use_prepared_multiscale(fpn, uv, sampler)
        assert not _can_use_prepared_multiscale(
            (fpn[0].requires_grad_(), *fpn[1:]),
            uv,
            sampler,
        )
        fpn[0].requires_grad_(False)
        assert not _can_use_prepared_multiscale(
            (fpn[0].double(), *fpn[1:]),
            uv,
            sampler,
        )
        tensor_subclass = uv.as_subclass(type("TensorSubclass", (torch.Tensor,), {}))
        assert not _can_use_prepared_multiscale(fpn, tensor_subclass, sampler)

    assert not _can_use_prepared_multiscale(fpn, uv, sampler)


def test_prepared_grid_fails_closed_for_torch_override_modes():
    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    fpn, uv, _query, _level_head, sampler = _prepared_fixture()
    with torch.no_grad():
        assert _can_use_prepared_multiscale(fpn, uv, sampler)
        with FunctionMode():
            assert not _can_use_prepared_multiscale(fpn, uv, sampler)
        with DispatchMode():
            assert not _can_use_prepared_multiscale(fpn, uv, sampler)
