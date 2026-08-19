import sys
import weakref

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _python_dispatch
from torch.utils._python_dispatch import TorchDispatchMode

from rosa.observation import (
    _can_use_prepared_multiscale,
    _can_use_streaming_multiscale,
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
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _four_gather_reference(feature, uv):
    """Pre-optimization deterministic sampler retained as a test oracle."""

    batch, cameras, channels, height, width = feature.shape
    queries, samples, patch = uv.shape[1], uv.shape[3], uv.shape[4]
    camera_batches = batch * cameras
    sample_count = queries * samples * patch
    coordinates = torch.nan_to_num(
        uv.permute(0, 2, 1, 3, 4, 5),
        nan=-2.0,
        posinf=-2.0,
        neginf=-2.0,
    ).reshape(camera_batches, sample_count, 2)
    x = coordinates[..., 0] * width - 0.5
    y = coordinates[..., 1] * height - 0.5
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = x0 + 1, y0 + 1
    wx1 = (x - x0.to(x.dtype)).to(feature.dtype)
    wy1 = (y - y0.to(y.dtype)).to(feature.dtype)
    wx0, wy0 = 1.0 - wx1, 1.0 - wy1
    flattened = feature.reshape(camera_batches, channels, height * width)

    def gather(ix, iy):
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        linear = iy.clamp(0, height - 1) * width + ix.clamp(0, width - 1)
        values = torch.gather(
            flattened,
            2,
            linear[:, None].expand(-1, channels, -1),
        )
        return values * valid[:, None].to(values.dtype)

    output = (
        gather(x0, y0) * (wx0 * wy0)[:, None]
        + gather(x1, y0) * (wx1 * wy0)[:, None]
        + gather(x0, y1) * (wx0 * wy1)[:, None]
        + gather(x1, y1) * (wx1 * wy1)[:, None]
    )
    output = output.reshape(
        batch, cameras, channels, queries, samples, patch
    )
    return output.permute(0, 3, 1, 4, 5, 2).contiguous()


def _prepared_fixture():
    generator = torch.Generator().manual_seed(3407)
    fpn = tuple(
        torch.randn(1, 4, 8, size, size, generator=generator) for size in (8, 4, 2, 1)
    )
    uv = torch.rand(1, 5, 4, 13, 2, generator=generator)
    query = torch.randn(1, 5, 8, generator=generator)
    return fpn, uv, query, torch.nn.Linear(8, 4), FusedBilinearSampler().eval()


def _streaming_fixture(
    dtype=torch.float32,
    *,
    noncontiguous=False,
    tied_weights=False,
):
    generator = torch.Generator().manual_seed(91_817)
    fpn = [
        torch.randn(1, 4, 8, size, size, generator=generator).to(dtype)
        for size in (8, 6, 4, 2)
    ]
    uv = torch.rand(1, 7, 4, 13, 2, generator=generator).to(dtype)
    uv[0, 6, 0, :4].copy_(
        torch.tensor(
            (
                (float("nan"), float("inf")),
                (float("-inf"), -0.0),
                (0.0, 1.0),
                (-1.0, 2.0),
            ),
            dtype=dtype,
        )
    )
    fpn[0][0, 0, 0, 0, 0] = float("nan")
    fpn[1][0, 1, 1, 0, 0] = float("inf")
    fpn[2][0, 2, 2, 0, 0] = float("-inf")
    fpn[3][0, 3, 3, 0, 0] = -0.0
    uv[0, 0, 0, 0] = 0.5 / 8.0
    uv[0, 1, 1, 1] = 0.5 / 6.0
    uv[0, 2, 2, 2] = 0.5 / 4.0
    uv[0, 3, 3, 3] = 0.5 / 2.0
    if noncontiguous:
        fpn = [feature.transpose(-1, -2) for feature in fpn]
        uv_storage = torch.empty(*uv.shape[:-1], 4, dtype=dtype)
        uv_storage[..., ::2].copy_(uv)
        uv = uv_storage[..., ::2]
        assert not uv.is_contiguous()
        assert all(not feature.is_contiguous() for feature in fpn)
    query = torch.randn(1, 7, 8, generator=generator)
    level_head = torch.nn.Linear(8, 4).eval()
    if tied_weights:
        with torch.no_grad():
            level_head.weight.zero_()
            level_head.bias.zero_()
    return fpn, uv, query, level_head, FusedBilinearSampler().eval()


def _stacked_prepared_weighted(fpn, uv, query, level_head, sampler):
    levels = sampler._sample_multiscale_prepared_stacked(fpn, uv.unsqueeze(4))
    weights = torch.softmax(level_head(query).float(), dim=-1).to(levels.dtype)
    levels.mul_(weights[:, :, :, None, None, None])
    return levels.sum(dim=2)


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


def test_one_gather_sampler_matches_four_gather_output_and_gradients():
    generator = torch.Generator().manual_seed(81_105)
    reference_feature = torch.randn(
        2, 3, 5, 7, 9, generator=generator, requires_grad=True
    )
    candidate_feature = reference_feature.detach().clone().requires_grad_()
    reference_uv = (
        torch.rand(2, 11, 3, 2, 4, 2, generator=generator) * 1.4 - 0.2
    ).requires_grad_()
    candidate_uv = reference_uv.detach().clone().requires_grad_()

    expected = _four_gather_reference(reference_feature, reference_uv)
    actual = DeterministicBilinearSampler()(candidate_feature, candidate_uv)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(
        candidate_feature.grad,
        reference_feature.grad,
        atol=2e-7,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        candidate_uv.grad,
        reference_uv.grad,
        atol=2e-6,
        rtol=2e-6,
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
        torch.randn(1, 2, 3, size, size, generator=generator) for size in (8, 4, 2, 1)
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
        torch.randn(1, 2, 8, size, size, generator=generator) for size in (8, 4, 2, 1)
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
    candidate_fpn = [value.detach().clone().requires_grad_() for value in parent_fpn]
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


def test_private_eager_fills_the_final_level_buffer_directly():
    fpn, uv, query, level_head, sampler = _prepared_fixture()
    codes = {
        "prepare": type(sampler)._prepare_grid.__code__,
        "prepared_tuple": type(sampler)._sample_multiscale_prepared.__code__,
        "prepared_stacked": (
            type(sampler)._sample_multiscale_prepared_stacked.__code__
        ),
    }
    calls = {name: 0 for name in codes}

    def profile(frame, event, _arg):
        if event == "call":
            for name, code in codes.items():
                if frame.f_code is code:
                    calls[name] += 1
        return profile

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        with torch.no_grad():
            actual = _sample_multiscale_features_prepared_inference(
                fpn,
                uv,
                query,
                sampler,
                level_head,
            )
    finally:
        sys.setprofile(previous_profile)

    with torch.no_grad():
        expected = _released_multiscale(fpn, uv, query, sampler, level_head)
    assert calls == {
        "prepare": 1,
        "prepared_tuple": 0,
        "prepared_stacked": 1,
    }
    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)


def test_direct_level_buffer_matches_stack_for_special_values():
    fpn, uv, _query, _level_head, sampler = _prepared_fixture()
    mutable_fpn = [feature.clone() for feature in fpn]
    mutable_fpn[0][0, 0, 0, 0, 0] = -0.0
    mutable_fpn[1][0, 1, 1, 0, 0] = float("nan")
    mutable_fpn[2][0, 2, 2, 0, 0] = float("inf")
    mutable_fpn[3][0, 3, 3, 0, 0] = float("-inf")
    sample_uv = uv.unsqueeze(4)

    with torch.no_grad():
        expected = torch.stack(
            [
                value.squeeze(4)
                for value in sampler._sample_multiscale_prepared(
                    mutable_fpn,
                    sample_uv,
                )
            ],
            dim=2,
        )
        actual = sampler._sample_multiscale_prepared_stacked(
            mutable_fpn,
            sample_uv,
        )

    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)


def test_direct_level_buffer_matches_stack_under_cpu_autocast():
    fpn, uv, _query, _level_head, sampler = _prepared_fixture()
    fpn = tuple(feature.to(torch.bfloat16) for feature in fpn)
    sample_uv = uv.to(torch.bfloat16).unsqueeze(4)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = torch.stack(
            [
                value.squeeze(4)
                for value in sampler._sample_multiscale_prepared(
                    fpn,
                    sample_uv,
                )
            ],
            dim=2,
        )
        actual = sampler._sample_multiscale_prepared_stacked(
            fpn,
            sample_uv,
        )

    assert expected.dtype == torch.float32
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("noncontiguous", (False, True))
@pytest.mark.parametrize("tied_weights", (False, True))
def test_streaming_sampling_is_exact_or_falls_back_to_stacked(
    dtype,
    noncontiguous,
    tied_weights,
):
    fpn, uv, query, level_head, sampler = _streaming_fixture(
        dtype,
        noncontiguous=noncontiguous,
        tied_weights=tied_weights,
    )
    fpn_before = tuple(feature.clone() for feature in fpn)
    uv_before = uv.clone()
    query_before = query.clone()
    autocast = (
        torch.autocast("cpu", dtype=dtype)
        if dtype is not torch.float32
        else torch.autocast("cpu", enabled=False)
    )
    with torch.no_grad(), autocast:
        expected = _stacked_prepared_weighted(
            fpn,
            uv,
            query,
            level_head,
            sampler,
        )
        admitted = _can_use_streaming_multiscale(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
        actual = _sample_multiscale_features_prepared_inference(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )

    assert admitted is (dtype is torch.float32 and not noncontiguous)
    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)
    assert all(_raw_equal(feature, before) for feature, before in zip(fpn, fpn_before))
    assert _raw_equal(uv, uv_before)
    assert _raw_equal(query, query_before)


def test_private_eager_uses_streaming_without_the_stacked_level_buffer():
    fpn, uv, query, level_head, sampler = _streaming_fixture(tied_weights=True)
    codes = {
        "prepared_tuple": type(sampler)._sample_multiscale_prepared.__code__,
        "prepared_stacked": type(sampler)._sample_multiscale_prepared_stacked.__code__,
        "prepared_streaming": (
            type(sampler)._sample_multiscale_prepared_weighted_streaming.__code__
        ),
    }
    calls = {name: 0 for name in codes}

    def profile(frame, event, _arg):
        if event == "call":
            for name, code in codes.items():
                if frame.f_code is code:
                    calls[name] += 1
        return profile

    with torch.no_grad():
        expected = _stacked_prepared_weighted(
            fpn,
            uv,
            query,
            level_head,
            sampler,
        )
        previous_profile = sys.getprofile()
        sys.setprofile(profile)
        try:
            actual = _sample_multiscale_features_prepared_inference(
                fpn,
                uv,
                query,
                sampler,
                level_head,
            )
        finally:
            sys.setprofile(previous_profile)

    assert calls == {
        "prepared_tuple": 0,
        "prepared_stacked": 0,
        "prepared_streaming": 1,
    }
    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)


def test_streaming_helper_has_four_samples_and_no_stack_or_sum():
    fpn, uv, query, level_head, sampler = _streaming_fixture()
    weights = torch.softmax(level_head(query).float(), dim=-1)

    class OperationLog(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            self.operations.append(str(func))
            return func(*args, **(kwargs or {}))

    operation_log = OperationLog()
    with torch.no_grad(), operation_log:
        sampler._sample_multiscale_prepared_weighted_streaming(
            fpn,
            uv.unsqueeze(4),
            weights,
        )
    operations = operation_log.operations
    assert sum("grid_sampler_2d" in name for name in operations) == 4
    assert operations.count("aten.mul_.Tensor") == 4
    assert operations.count("aten.add_.Tensor") == 3
    assert operations.count("aten.copy_.default") == 1
    assert not any("aten.stack" in name for name in operations)
    assert not any("aten.sum.dim_IntList" in name for name in operations)


def test_streaming_helper_releases_each_grid_result_before_the_next(monkeypatch):
    fpn, uv, query, level_head, sampler = _streaming_fixture()
    weights = torch.softmax(level_head(query).float(), dim=-1)
    released_grid_sample = torch.nn.functional.grid_sample
    prior_results = []

    def tracked_grid_sample(*args, **kwargs):
        assert all(reference() is None for reference in prior_results)
        result = released_grid_sample(*args, **kwargs)
        prior_results.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch.nn.functional, "grid_sample", tracked_grid_sample)
    with torch.no_grad():
        sampler._sample_multiscale_prepared_weighted_streaming(
            fpn,
            uv.unsqueeze(4),
            weights,
        )
    assert len(prior_results) == 4
    assert all(reference() is None for reference in prior_results)


def test_streaming_method_override_falls_back_before_invocation(monkeypatch):
    fpn, uv, query, level_head, sampler = _streaming_fixture()
    with torch.no_grad():
        expected = _stacked_prepared_weighted(
            fpn,
            uv,
            query,
            level_head,
            sampler,
        )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("overridden streaming sampler executed")

    monkeypatch.setattr(
        sampler,
        "_sample_multiscale_prepared_weighted_streaming",
        must_not_run,
    )
    with torch.no_grad():
        assert not _can_use_streaming_multiscale(
            fpn,
            uv,
            query,
            sampler,
            level_head,
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


@pytest.mark.parametrize(
    ("owner", "helper_name"),
    (
        (torch.nn.functional, "grid_sample"),
        (torch, "softmax"),
        (torch.Tensor, "float"),
        (torch.Tensor, "to"),
        (torch.Tensor, "mul_"),
        (torch.Tensor, "sum"),
        (torch.Tensor, "new_empty"),
        (torch.Tensor, "copy_"),
    ),
)
def test_streaming_preserves_overridden_stacked_helpers(
    owner,
    helper_name,
    monkeypatch,
):
    fpn, uv, query, level_head, sampler = _streaming_fixture()
    with torch.no_grad():
        expected = _stacked_prepared_weighted(
            fpn,
            uv,
            query,
            level_head,
            sampler,
        )
    released = getattr(owner, helper_name)
    calls = []

    def observed(*args, **kwargs):
        calls.append(True)
        return released(*args, **kwargs)

    monkeypatch.setattr(owner, helper_name, observed)
    with torch.no_grad():
        assert not _can_use_streaming_multiscale(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
        actual = _sample_multiscale_features_prepared_inference(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )
    assert calls
    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)


@pytest.mark.parametrize(
    ("owner", "helper_name"),
    ((torch, "empty"), (torch.Tensor, "add_")),
)
def test_streaming_falls_back_before_overridden_new_helpers(
    owner,
    helper_name,
    monkeypatch,
):
    fpn, uv, query, level_head, sampler = _streaming_fixture()
    with torch.no_grad():
        expected = _stacked_prepared_weighted(
            fpn,
            uv,
            query,
            level_head,
            sampler,
        )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("overridden streaming-only helper executed")

    monkeypatch.setattr(owner, helper_name, must_not_run)
    with torch.no_grad():
        assert not _can_use_streaming_multiscale(
            fpn,
            uv,
            query,
            sampler,
            level_head,
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


def test_streaming_preserves_level_head_hooks():
    fpn, uv, query, level_head, sampler = _streaming_fixture()
    with torch.no_grad():
        expected = _stacked_prepared_weighted(
            fpn,
            uv,
            query,
            level_head,
            sampler,
        )
    calls = []
    handle = level_head.register_forward_hook(lambda *_args: calls.append(True))
    try:
        with torch.no_grad():
            assert not _can_use_streaming_multiscale(
                fpn,
                uv,
                query,
                sampler,
                level_head,
            )
            actual = _sample_multiscale_features_prepared_inference(
                fpn,
                uv,
                query,
                sampler,
                level_head,
            )
    finally:
        handle.remove()
    assert calls == [True]
    assert actual.stride() == expected.stride()
    assert _raw_equal(actual, expected)


def test_streaming_falls_back_under_cpu_autocast():
    fpn, uv, query, level_head, sampler = _streaming_fixture()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = _stacked_prepared_weighted(
            fpn,
            uv,
            query,
            level_head,
            sampler,
        )
        assert not _can_use_streaming_multiscale(
            fpn,
            uv,
            query,
            sampler,
            level_head,
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


def test_streaming_fails_closed_before_overridden_autocast_state(monkeypatch):
    fpn, uv, query, level_head, sampler = _streaming_fixture()
    with torch.no_grad():
        expected = _stacked_prepared_weighted(
            fpn,
            uv,
            query,
            level_head,
            sampler,
        )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("overridden autocast-state helper executed")

    monkeypatch.setattr(torch, "is_autocast_enabled", must_not_run)
    with torch.no_grad():
        assert not _can_use_streaming_multiscale(
            fpn,
            uv,
            query,
            sampler,
            level_head,
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


def test_direct_level_buffer_releases_each_grid_result_before_the_next(
    monkeypatch,
):
    fpn, uv, query, level_head, sampler = _prepared_fixture()
    original_grid_sample = torch.nn.functional.grid_sample
    prior_results = []

    def tracked_grid_sample(*args, **kwargs):
        assert all(reference() is None for reference in prior_results)
        result = original_grid_sample(*args, **kwargs)
        prior_results.append(weakref.ref(result))
        return result

    monkeypatch.setattr(
        torch.nn.functional,
        "grid_sample",
        tracked_grid_sample,
    )
    with torch.no_grad():
        _sample_multiscale_features_prepared_inference(
            fpn,
            uv,
            query,
            sampler,
            level_head,
        )

    assert len(prior_results) == 4
    assert all(reference() is None for reference in prior_results)


def test_private_eager_direct_level_method_override_falls_back(monkeypatch):
    fpn, uv, query, level_head, sampler = _prepared_fixture()
    with torch.no_grad():
        expected = _released_multiscale(fpn, uv, query, sampler, level_head)

    calls = []

    def overridden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("overridden private method must not run")

    monkeypatch.setattr(
        sampler,
        "_sample_multiscale_prepared_stacked",
        overridden,
    )
    with torch.no_grad():
        assert not _can_use_prepared_multiscale(
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

    assert calls == []
    assert _raw_equal(actual, expected)


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


@pytest.mark.parametrize(
    ("owner", "attribute"),
    (
        (torch.overrides, "_is_torch_function_mode_enabled"),
        (_python_dispatch, "_is_in_non_infra_torch_dispatch_mode"),
    ),
)
def test_prepared_guard_fails_closed_for_unknown_override_mode_state(
    monkeypatch,
    owner,
    attribute,
):
    fpn, uv, _query, _level_head, sampler = _prepared_fixture()

    with torch.no_grad():
        with monkeypatch.context() as patch:
            patch.delattr(owner, attribute, raising=False)
            assert not _can_use_prepared_multiscale(
                fpn,
                uv,
                sampler,
                private_eager_inference=True,
            )
        with monkeypatch.context() as patch:
            patch.setattr(owner, attribute, lambda: 1)
            assert not _can_use_prepared_multiscale(
                fpn,
                uv,
                sampler,
                private_eager_inference=True,
            )


@pytest.mark.parametrize(
    ("owner", "attribute"),
    (
        (torch.overrides, "_is_torch_function_mode_enabled"),
        (_python_dispatch, "_is_in_non_infra_torch_dispatch_mode"),
    ),
)
def test_prepared_guard_fails_closed_when_override_mode_state_raises(
    monkeypatch,
    owner,
    attribute,
):
    fpn, uv, _query, _level_head, sampler = _prepared_fixture()

    def unavailable():
        raise RuntimeError("override mode state unavailable")

    monkeypatch.setattr(owner, attribute, unavailable)
    with torch.no_grad():
        assert not _can_use_prepared_multiscale(
            fpn,
            uv,
            sampler,
            private_eager_inference=True,
        )


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
