import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _python_dispatch
from torch.utils._python_dispatch import TorchDispatchMode

from rosa.observation import (
    _can_use_prepared_multiscale,
    _sample_multiscale_features_prepared_inference,
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


def _prepared_fixture():
    generator = torch.Generator().manual_seed(3407)
    fpn = tuple(
        torch.randn(1, 4, 8, size, size, generator=generator)
        for size in (8, 4, 2, 1)
    )
    uv = torch.rand(1, 5, 4, 13, 2, generator=generator)
    query = torch.randn(1, 5, 8, generator=generator)
    return fpn, uv, query, torch.nn.Linear(8, 4), FusedBilinearSampler().eval()


def _capture_compiled(fpn, uv, query, level_head, sampler):
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


def test_compiled_prepared_grid_is_raw_exact_and_only_reduces_grid_prep():
    fpn, uv, query, level_head, candidate_sampler = _prepared_fixture()
    uv[0, 0, 0, 0] = torch.tensor((float("nan"), 0.5))
    uv[0, 1, 1, 1] = torch.tensor((float("inf"), float("-inf")))

    with torch.no_grad():
        expected = _released_multiscale(
            fpn,
            uv,
            query,
            candidate_sampler,
            level_head,
        )
        assert not _can_use_prepared_multiscale(fpn, uv, candidate_sampler)

    parent, parent_graph = _capture_compiled(
        fpn,
        uv,
        query,
        level_head,
        FusedBilinearSampler().train(),
    )
    candidate, candidate_graph = _capture_compiled(
        fpn,
        uv,
        query,
        level_head,
        candidate_sampler,
    )

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
                node.op == "call_method" and node.target == "sum"
                for node in nodes
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
    assert candidate.stride() == parent.stride() == expected.stride()
    assert _raw_equal(candidate, parent)
    assert _raw_equal(candidate, expected)


def test_eager_multiscale_sampling_preserves_public_sampler_hooks():
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


def test_private_eager_prepared_grid_is_raw_exact_and_fail_closed_on_hooks():
    fpn, uv, query, level_head, sampler = _prepared_fixture()
    uv[0, 0, 0, 0] = torch.tensor((float("nan"), 0.5))
    uv[0, 1, 1, 1] = torch.tensor((float("inf"), float("-inf")))

    with torch.no_grad():
        expected = _released_multiscale(fpn, uv, query, sampler, level_head)
        assert not _can_use_prepared_multiscale(fpn, uv, sampler)
        assert _can_use_prepared_multiscale(
            fpn,
            uv,
            sampler,
            private_eager_inference=True,
        )
        actual = _sample_multiscale_features_prepared_inference(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)

    calls = []
    handle = sampler.register_forward_hook(lambda *_args: calls.append("hook"))
    try:
        with torch.no_grad():
            assert not _can_use_prepared_multiscale(
                fpn,
                uv,
                sampler,
                private_eager_inference=True,
            )
            hooked = _sample_multiscale_features_prepared_inference(
                fpn,
                uv,
                query,
                sampler,
                level_head,
            )
    finally:
        handle.remove()
    assert calls == ["hook"] * 4
    assert _raw_equal(hooked, expected)


def test_private_eager_prepared_grid_requires_exact_bool_opt_in():
    fpn, uv, _query, _level_head, sampler = _prepared_fixture()
    with pytest.raises(TypeError, match="private_eager_inference"):
        _can_use_prepared_multiscale(
            fpn,
            uv,
            sampler,
            private_eager_inference=1,
        )


def test_private_eager_prepared_grid_falls_back_under_dispatch_modes():
    fpn, uv, query, level_head, sampler = _prepared_fixture()

    class CountDispatch(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.nan_to_num_calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            if "nan_to_num" in str(func):
                self.nan_to_num_calls += 1
            return func(*args, **(kwargs or {}))

    dispatch = CountDispatch()
    with torch.no_grad(), dispatch:
        assert not _can_use_prepared_multiscale(
            fpn,
            uv,
            sampler,
            private_eager_inference=True,
        )
        dispatch_output = _sample_multiscale_features_prepared_inference(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
    assert dispatch.nan_to_num_calls == 4

    class CountFunction(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.nan_to_num_calls = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            del types
            if func is torch.nan_to_num:
                self.nan_to_num_calls += 1
            return func(*args, **(kwargs or {}))

    function = CountFunction()
    with torch.no_grad(), function:
        assert not _can_use_prepared_multiscale(
            fpn,
            uv,
            sampler,
            private_eager_inference=True,
        )
        function_output = _sample_multiscale_features_prepared_inference(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
    assert function.nan_to_num_calls == 4
    assert _raw_equal(dispatch_output, function_output)


def test_private_eager_prepared_grid_falls_back_for_tracing_training_and_grad(
    monkeypatch,
):
    fpn, uv, _query, _level_head, sampler = _prepared_fixture()

    with torch.no_grad():
        monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
        assert not _can_use_prepared_multiscale(
            fpn,
            uv,
            sampler,
            private_eager_inference=True,
        )
        monkeypatch.undo()

        sampler.train()
        assert not _can_use_prepared_multiscale(
            fpn,
            uv,
            sampler,
            private_eager_inference=True,
        )

    sampler.eval()
    assert not _can_use_prepared_multiscale(
        fpn,
        uv,
        sampler,
        private_eager_inference=True,
    )


def test_prepared_guard_supports_torch_without_new_dispatch_mode_flag(
    monkeypatch,
):
    fpn, uv, _query, _level_head, sampler = _prepared_fixture()
    monkeypatch.delattr(
        _python_dispatch,
        "_is_in_non_infra_torch_dispatch_mode",
        raising=False,
    )

    with torch.no_grad():
        assert not _can_use_prepared_multiscale(fpn, uv, sampler)
        assert not _can_use_prepared_multiscale(
            fpn,
            uv,
            sampler,
            private_eager_inference=True,
        )
        monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)
        assert _can_use_prepared_multiscale(fpn, uv, sampler)


def test_compiled_sampler_state_falls_back_to_public_call(monkeypatch):
    fpn, uv, query, level_head, sampler = _prepared_fixture()
    with torch.no_grad():
        expected = _released_multiscale(fpn, uv, query, sampler, level_head)

    sampler.compile(backend="eager", fullgraph=True)
    assert sampler._compiled_call_impl is not None
    with torch.no_grad():
        monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)
        assert not _can_use_prepared_multiscale(fpn, uv, sampler)
        monkeypatch.undo()
        actual = sample_multiscale_features(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
    torch._dynamo.reset()

    assert _raw_equal(actual, expected)
