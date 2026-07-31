import torch

from rosa.sampling import (
    DeterministicBilinearSampler,
    FusedBilinearSampler,
)


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
