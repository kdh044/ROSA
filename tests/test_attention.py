import copy

import torch
import torch.nn.functional as F

from rosa.attention import mean_pairwise_camera_cosine
from rosa.rfl import SphericalRayFeatureLifting
from rosa.types import RayFeatureRelation


def test_linear_pairwise_camera_consistency_matches_explicit_pairs():
    torch.manual_seed(5)
    values = F.normalize(torch.randn(2, 3, 4, 5, 7), dim=-1)
    valid = torch.rand(2, 3, 4, 5) > 0.35
    efficient = mean_pairwise_camera_cosine(values, valid)

    explicit = torch.zeros_like(efficient)
    counts = torch.zeros_like(efficient)
    for left in range(values.shape[2]):
        for right in range(left + 1, values.shape[2]):
            pair_valid = valid[:, :, left] & valid[:, :, right]
            similarity = (
                values[:, :, left] * values[:, :, right]
            ).sum(dim=-1)
            explicit += similarity * pair_valid
            counts += pair_valid
    explicit = torch.where(
        counts > 0,
        explicit / counts.clamp_min(1),
        torch.zeros_like(explicit),
    )

    torch.testing.assert_close(efficient, explicit, atol=2e-6, rtol=2e-6)


def test_compact_rfl_relation_is_raw_exact_to_legacy_carrier():
    generator = torch.Generator().manual_seed(3407)
    features = torch.randn(2, 3, 4, 5, 8, generator=generator)
    ray = F.normalize(torch.randn(2, 3, 4, 5, 3, generator=generator), dim=-1)
    ray[0, 0, 0, 0] = torch.tensor((float("nan"), 0.0, 0.0))
    valid = torch.rand(2, 3, 4, 5, generator=generator) > 0.25
    legacy_bncp = torch.zeros(2, 3, 4, 5, 19)
    legacy_bncp[..., 8:11] = ray
    legacy_bncp[..., 13] = valid
    legacy = legacy_bncp.permute(0, 2, 1, 3, 4).contiguous()
    compact = RayFeatureRelation(ray=ray, valid=valid)
    parent = SphericalRayFeatureLifting(embed_dims=8, hidden_dims=4)
    candidate = copy.deepcopy(parent)
    with torch.no_grad():
        parent.output_proj.weight.normal_(generator=generator)
        candidate.load_state_dict(parent.state_dict())

    parent_features = features.clone().requires_grad_()
    candidate_features = features.clone().requires_grad_()
    expected = parent(parent_features, legacy)
    actual = candidate(candidate_features, compact)
    assert torch.equal(actual, expected)

    upstream = torch.randn(actual.shape, generator=generator)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    assert torch.equal(candidate_features.grad, parent_features.grad)
    for candidate_parameter, parent_parameter in zip(
        candidate.parameters(),
        parent.parameters(),
    ):
        assert torch.equal(candidate_parameter.grad, parent_parameter.grad)
