from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rosa import ROSAConfig
from rosa.integrations.sparse4d_losses import (
    _candidate_target_distribution,
    _selected_mean,
    _yaw_selected_mean,
    evaluator_aligned_centerness_target,
    evaluator_aligned_quality_target,
    rosa_losses,
)
from rosa.integrations.sparse4d_scores import compose_quality_rank
from rosa.refiner import ROSARefiner


def test_evaluator_aligned_centerness_target_has_no_box_gradient():
    prediction = torch.tensor(
        [[[0.0, 0.0, 5.0], [3.0, 4.0, -8.0]]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    score = evaluator_aligned_centerness_target(prediction, target)
    torch.testing.assert_close(
        score,
        torch.tensor([[1.0, torch.exp(torch.tensor(-5.0))]]),
    )
    assert not score.requires_grad


def test_quality_target_matches_strict_official_ap_distance_thresholds():
    prediction = torch.zeros(1, 6, 3)
    prediction[..., 0] = torch.tensor((0.49, 0.50, 0.75, 1.50, 3.00, 0.25))
    prediction.requires_grad_()
    target = torch.zeros_like(prediction)
    matched = torch.ones(1, 6, dtype=torch.bool)
    predicted_class = torch.tensor(((0, 0, 0, 0, 0, 1),))
    target_class = torch.zeros(1, 6, dtype=torch.long)

    score = evaluator_aligned_quality_target(
        prediction,
        target,
        matched,
        predicted_class,
        target_class,
    )

    torch.testing.assert_close(
        score,
        torch.tensor(((1.0, 0.75, 0.75, 0.50, 0.25, 0.0),)),
    )
    assert not score.requires_grad


class _FixedSampler:
    gamma = 2.0

    def __init__(self, target):
        self.target = target

    def sample(self, classification, regression, *_):
        batch, queries = regression.shape[:2]
        return (
            torch.zeros(
                batch,
                queries,
                dtype=torch.long,
                device=regression.device,
            ),
            self.target,
            torch.ones_like(self.target),
        )


class _BoxLoss:
    @staticmethod
    def loss_box(prediction, target, *, weight, avg_factor):
        error = F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
        )
        return (error * weight).sum() / avg_factor


def _refiner_fixture(config):
    torch.manual_seed(7)
    refiner = ROSARefiner(config)
    batch, queries, cameras, dims = 1, 2, 4, config.embed_dims
    anchor = torch.zeros(batch, queries, 11)
    anchor[..., 0] = torch.tensor((1.0, 2.0))
    anchor[..., 2] = -5.0
    anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5)).log()
    anchor[..., 7] = 1.0
    query = torch.randn(batch, queries, dims)
    fpn = [torch.randn(batch, cameras, dims, size, size) for size in (16, 8, 4, 2)]
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
    keypoint_scale = anchor.new_tensor(
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

    def keypoint_generator(box, _query):
        return (
            box[..., :3, None].transpose(-1, -2)
            + keypoint_scale[None, None] * box[..., 3:6].exp()[..., None, :]
        )

    output = refiner(
        anchor,
        query,
        fpn,
        metadata,
        keypoint_generator,
    )
    return refiner, anchor, output


def _full_loss(config):
    refiner, anchor, output = _refiner_fixture(config)
    target = anchor[..., :10].clone()
    target[..., 0] += torch.tensor(((1.0, -0.8),))
    target[..., 1] += torch.tensor(((0.3, 0.6),))
    target[..., 3:6] += torch.tensor((0.05, -0.03, 0.02))
    yaw_delta = torch.tensor(((0.20, -0.15),))
    base_yaw = torch.atan2(anchor[..., 6], anchor[..., 7])
    target[..., 6] = torch.sin(base_yaw + yaw_delta)
    target[..., 7] = torch.cos(base_yaw + yaw_delta)

    baseline_classification = anchor.new_tensor((((2.0, -1.0), (1.5, -0.5)),))
    baseline_quality = anchor.new_zeros(1, 2, 2)
    rank = compose_quality_rank(
        baseline_classification,
        baseline_quality,
        output.quality_delta,
        output.initial_supported,
        centerness_index=0,
    )
    record = {
        "baseline_classification": baseline_classification,
        "baseline_anchor": anchor.detach(),
        "final_anchor": output.final_anchor,
        "supported": output.initial_supported,
        "iteration_base_anchors": output.iteration_base_anchors,
        "iteration_anchors": output.iteration_anchors,
        "iteration_observation_supported": (output.iteration_observation_supported),
        "iteration_tangent_delta": output.iteration_tangent_delta,
        "iteration_distribution_mean": output.iteration_distribution_mean,
        "iteration_distribution_std": output.iteration_distribution_std,
        "iteration_candidate_offsets": output.iteration_candidate_offsets,
        "iteration_candidate_supported": (output.iteration_candidate_supported),
        "iteration_candidate_probabilities": (output.iteration_candidate_probabilities),
        "iteration_radial_safe": output.iteration_radial_safe,
        "iteration_radial_proposal": output.iteration_radial_proposal,
        "iteration_radial_gate_logit": (output.iteration_radial_gate_logit),
        "iteration_yaw_raw_delta": output.iteration_yaw_raw_delta,
        "iteration_yaw_gate": output.iteration_yaw_gate,
        "iteration_yaw_delta": output.iteration_yaw_delta,
        "radial_gate": output.radial_gate,
        "radial_safe": output.radial_safe,
        "tangent_delta": output.tangent_delta,
        "posterior_entropy": output.posterior_entropy,
        "distribution_mean": output.distribution_mean,
        "distribution_std": output.distribution_std,
        "quality_delta": output.quality_delta,
        "joint_logit": rank["joint_logit"],
        "effective_jointscore_delta": rank["effective_delta"],
        "native_joint_score": rank["native_joint_score"],
        "jointscore_eligible": rank["eligible"],
    }
    head = SimpleNamespace(
        rosa_cfg=config,
        reg_weights=[2.0] * 3 + [1.0] * 7,
        sampler=_FixedSampler(target),
        loss_reg=_BoxLoss(),
        gt_cls_key="gt_cls",
        gt_reg_key="gt_reg",
    )
    losses = rosa_losses(
        head,
        record,
        {"gt_cls": None, "gt_reg": None},
    )
    objective = sum(value for name, value in losses.items() if name.startswith("loss_"))
    return refiner, losses, objective


@pytest.mark.parametrize(
    ("profile", "overrides"),
    (
        ("published", {}),
        ("balanced", {}),
        ("fast", {}),
        (
            "published",
            {
                "enable_radial": False,
                "train_disabled_branch_aux": False,
            },
        ),
        ("published", {"enable_tangent": False}),
        ("published", {"enable_shape_refinement": False}),
        ("published", {"enable_quality_score": False}),
        ("balanced", {"enable_radial": False}),
        ("balanced", {"enable_tangent": False}),
        ("balanced", {"enable_shape_refinement": False}),
    ),
)
def test_full_loss_backward_keeps_every_parameter_in_the_ddp_graph(
    profile,
    overrides,
):
    config = ROSAConfig.from_mapping(
        {
            "profile": profile,
            "embed_dims": 16,
            "hidden_dims": 8,
            **overrides,
        }
    )
    refiner, losses, objective = _full_loss(config)

    assert all(
        torch.isfinite(value).all()
        for name, value in losses.items()
        if name.startswith("loss_")
    )
    objective.backward()
    missing = [
        name
        for name, parameter in refiner.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite = [
        name
        for name, parameter in refiner.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert missing == []
    assert nonfinite == []


def test_compact_production_loss_unlocks_factorized_rfl_on_third_backward(
    monkeypatch,
):
    with torch.random.fork_rng(devices=[]):
        config = ROSAConfig.from_mapping(
            {
                "profile": "fast",
                "embed_dims": 16,
                "hidden_dims": 8,
            }
        )
        refiner, _, _ = _full_loss(config)
        # Reuse one trained module while _full_loss rebuilds the same seeded
        # inputs and the complete production supervision on every pass.
        monkeypatch.setitem(globals(), "ROSARefiner", lambda _config: refiner)
        optimizer = torch.optim.SGD(refiner.parameters(), lr=1e-3)
        readouts = (
            refiner.shared_shape_head[-1],
            refiner.shared_tangent_head[-1],
            refiner.range_distribution_head[-1],
            refiner.candidate_head[-1],
            refiner.center_gate_head[-1],
        )
        rfl = refiner.rosa.rfl
        assert rfl.ray_proj is not None
        assert all(torch.count_nonzero(head.weight) == 0 for head in readouts)
        assert torch.count_nonzero(rfl.output_proj.weight) == 0

        expected_rfl_gradients = (
            (False, False, False),
            (True, False, False),
            (True, True, True),
        )
        for backward_index, expected in enumerate(expected_rfl_gradients):
            optimizer.zero_grad(set_to_none=True)
            reused, losses, objective = _full_loss(config)
            assert reused is refiner
            assert all(
                torch.isfinite(value).all()
                for name, value in losses.items()
                if name.startswith("loss_")
            )
            objective.backward()

            for head in readouts:
                gradient = head.weight.grad
                assert gradient is not None
                assert torch.isfinite(gradient).all()
                assert torch.count_nonzero(gradient) > 0
            for parameter, should_be_nonzero in zip(
                (
                    rfl.output_proj.weight,
                    rfl.appearance_proj.weight,
                    rfl.ray_proj.weight,
                ),
                expected,
            ):
                gradient = parameter.grad
                assert gradient is not None
                assert torch.isfinite(gradient).all()
                assert bool(torch.count_nonzero(gradient)) is should_be_nonzero

            optimizer.step()
            if backward_index == 0:
                assert all(torch.count_nonzero(head.weight) > 0 for head in readouts)
                assert torch.count_nonzero(rfl.output_proj.weight) == 0
            elif backward_index == 1:
                assert torch.count_nonzero(rfl.output_proj.weight) > 0


def test_candidate_target_stays_normalized_when_best_hypothesis_is_hidden():
    distribution = _candidate_target_distribution(
        offsets=torch.tensor([[[-1.0, 0.0, 1.0]]]),
        radial_target=torch.tensor([[10.0]]),
        sigma=torch.tensor([[0.5]]),
        supported=torch.tensor([[[True, True, False]]]),
        selected=torch.tensor([[True]]),
        use_masked_softmax=True,
    )
    torch.testing.assert_close(
        distribution.sum(dim=-1),
        torch.ones(1, 1),
    )
    assert distribution[0, 0, 2] == 0
    assert distribution[0, 0, 1] > distribution[0, 0, 0]


def test_yaw_class_weighting_keeps_the_matched_count_denominator():
    per_query = torch.tensor([[1.0, 2.0, 100.0]])
    selected = torch.tensor([[True, True, False]])
    class_weight = torch.tensor([[2.0, 0.5, 99.0]])

    loss = _yaw_selected_mean(per_query, selected, class_weight)

    torch.testing.assert_close(loss, torch.tensor(1.5))


def test_disabled_yaw_class_weighting_is_exactly_the_original_reduction():
    per_query = torch.tensor([[0.7, 1.3, 2.1]], dtype=torch.float64)
    selected = torch.tensor([[True, False, True]])

    original = _selected_mean(per_query, selected)
    disabled = _yaw_selected_mean(per_query, selected, None)

    assert torch.equal(disabled, original)


def test_yaw_class_weighting_changes_only_the_three_compact_yaw_losses():
    common = {
        "profile": "balanced",
        "embed_dims": 16,
        "hidden_dims": 8,
    }
    _, unweighted, _ = _full_loss(ROSAConfig.from_mapping(common))
    _, weighted, _ = _full_loss(
        ROSAConfig.from_mapping(
            {
                **common,
                "yaw_class_loss_weights": (2.0,) * 6,
            }
        )
    )
    weighted_names = {
        "loss_cl_shape_yaw_6",
        "loss_cl_yaw_keep_6",
        "loss_cl_yaw_no_harm_6",
    }
    for name, original in unweighted.items():
        if not name.startswith("loss_"):
            continue
        if name in weighted_names:
            torch.testing.assert_close(weighted[name], 2 * original)
        else:
            assert torch.equal(weighted[name], original), name


def test_compact_quality_loss_gradients_are_score_head_only():
    config = ROSAConfig.from_mapping(
        {
            "profile": "balanced",
            "embed_dims": 16,
            "hidden_dims": 8,
            "enable_quality_score": True,
            "quality_loss_weight": 0.5,
        }
    )
    refiner, losses, _ = _full_loss(config)

    losses["loss_cl_quality_6"].backward()

    nonzero = {
        name
        for name, parameter in refiner.named_parameters()
        if parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
    }
    assert nonzero
    assert all(name.startswith("quality_head.") for name in nonzero)
