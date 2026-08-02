import torch

from rosa.observation import sample_multiscale_features
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
