import copy

import torch
import torch.nn.functional as F

from rosa.attention import PointwiseROSA, mean_pairwise_camera_cosine
from rosa.config import ROSAConfig
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
            similarity = (values[:, :, left] * values[:, :, right]).sum(dim=-1)
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
    modules = (
        SphericalRayFeatureLifting(embed_dims=8, hidden_dims=4),
        SphericalRayFeatureLifting.for_compact_profile(embed_dims=8),
    )
    for parent in modules:
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


def test_compact_rfl_step_zero_identity_dtype_and_first_backward():
    generator = torch.Generator().manual_seed(3407)
    module = SphericalRayFeatureLifting.for_compact_profile(embed_dims=8)
    features = torch.randn(
        2,
        3,
        4,
        5,
        8,
        generator=generator,
        dtype=torch.float16,
    ).requires_grad_()
    ray = F.normalize(torch.randn(2, 3, 4, 5, 3, generator=generator), dim=-1)
    valid = torch.rand(2, 3, 4, 5, generator=generator) > 0.25
    relation = RayFeatureRelation(ray=ray, valid=valid)

    delta = module(features, relation)
    assert delta.shape == features.shape
    assert delta.dtype == features.dtype
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(features + delta, features)

    upstream = torch.randn(delta.shape, generator=generator)
    (delta.float() * upstream).sum().backward()
    output_grad = module.output_proj.weight.grad
    assert output_grad is not None
    assert torch.count_nonzero(output_grad) > 0
    for projection in (module.appearance_proj, module.ray_proj):
        assert projection is not None
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) == 0


def test_compact_rfl_zero_output_initialization_unlocks_all_factors_on_step_two():
    generator = torch.Generator().manual_seed(7_101_080)
    module = SphericalRayFeatureLifting.for_compact_profile(embed_dims=8)
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-2)
    features = torch.randn(2, 3, 4, 5, 8, generator=generator)
    ray = F.normalize(torch.randn(2, 3, 4, 5, 3, generator=generator), dim=-1)
    valid = torch.rand(2, 3, 4, 5, generator=generator) > 0.2
    relation = RayFeatureRelation(ray=ray, valid=valid)
    upstream = torch.randn(features.shape, generator=generator)

    first = module(features, relation)
    (first * upstream).sum().backward()
    output_gradient = module.output_proj.weight.grad
    assert output_gradient is not None
    assert torch.isfinite(output_gradient).all()
    assert torch.count_nonzero(output_gradient) > 0
    for projection in (module.appearance_proj, module.ray_proj):
        assert projection is not None
        assert torch.equal(
            projection.weight.grad,
            torch.zeros_like(projection.weight.grad),
        )
    optimizer.step()
    assert torch.count_nonzero(module.output_proj.weight) > 0

    optimizer.zero_grad(set_to_none=True)
    second = module(features, relation)
    (second * upstream).sum().backward()
    for projection in (
        module.appearance_proj,
        module.ray_proj,
        module.output_proj,
    ):
        assert projection is not None
        gradient = projection.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_compact_rfl_is_a_bias_free_appearance_ray_interaction():
    generator = torch.Generator().manual_seed(7101080)
    module = SphericalRayFeatureLifting.for_compact_profile(embed_dims=8)
    assert module.factorized
    assert module.ray_proj is not None
    assert module.appearance_proj.bias is None
    assert module.ray_proj.bias is None
    assert module.output_proj.bias is None
    with torch.no_grad():
        module.output_proj.weight.normal_(generator=generator)

    features = torch.randn(1, 2, 4, 3, 8, generator=generator)
    ray = F.normalize(torch.randn(1, 2, 4, 3, 3, generator=generator), dim=-1)
    valid = torch.ones(1, 2, 4, 3, dtype=torch.bool)
    valid[:, 0, 0, 0] = False
    relation = RayFeatureRelation(ray=ray, valid=valid)
    output = module(features, relation)

    assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))
    assert torch.count_nonzero(output[valid]) > 0
    assert torch.equal(
        module(torch.zeros_like(features), relation),
        torch.zeros_like(output),
    )

    changed_ray = RayFeatureRelation(ray=ray.roll(1, dims=-1), valid=valid)
    assert not torch.equal(module(features, changed_ray)[valid], output[valid])
    changed_features = features.roll(1, dims=-1)
    assert not torch.equal(module(changed_features, relation)[valid], output[valid])


def test_behavior_profiles_select_fixed_factorized_or_released_dense_rfl():
    compact = PointwiseROSA(
        ROSAConfig.from_mapping({"profile": "fast", "embed_dims": 16, "hidden_dims": 8})
    ).rfl
    released = PointwiseROSA(
        ROSAConfig.from_mapping(
            {"profile": "published", "embed_dims": 16, "hidden_dims": 8}
        )
    ).rfl

    assert compact.factorized
    assert compact.hidden_dims == 16
    assert compact.ray_proj is not None
    assert compact.output_proj.in_features == 16
    assert not released.factorized
    assert released.ray_proj is None
    assert released.output_proj.in_features == (
        released.num_sh_basis * released.hidden_dims
    )
    assert tuple(released.state_dict()) == (
        "appearance_proj.weight",
        "output_proj.weight",
    )


def test_compact_observation_usages_keep_required_outputs_raw_exact():
    generator = torch.Generator().manual_seed(7101080)
    config = ROSAConfig.from_mapping(
        {
            "profile": "fast",
            "embed_dims": 16,
            "hidden_dims": 8,
        }
    )
    module = PointwiseROSA(config).eval()
    features = torch.randn(2, 3, 4, 13, 16, generator=generator)
    geometry = torch.randn(2, 3, 4, 13, 14, generator=generator)
    ray = F.normalize(
        torch.randn(2, 3, 4, 13, 3, generator=generator),
        dim=-1,
    )
    valid = torch.rand(2, 3, 4, 13, generator=generator) > 0.2
    query = torch.randn(2, 3, 16, generator=generator)
    relation = RayFeatureRelation(ray=ray, valid=valid)

    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.1, 0.1, generator=generator)
        full = module(features, geometry, relation, valid, query)
        candidate = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="candidate",
        )
        base = module(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage="base_support",
        )

    assert torch.equal(candidate.object_feature, full.object_feature)
    assert torch.equal(candidate.supported, full.supported)
    assert torch.equal(
        candidate.cross_camera_consistency,
        full.cross_camera_consistency,
    )
    assert torch.equal(base.object_feature, full.object_feature)
    assert torch.equal(base.supported, full.supported)
    assert base.cross_camera_consistency is None
    for observation in (candidate, base):
        assert observation.point_features is None
        assert observation.camera_weights is None
        assert observation.camera_weight_mean is None
        assert observation.camera_geometry_summary is None
