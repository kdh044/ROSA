from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rosa import ROSAConfig
from rosa.integrations.sparse4d_losses import (
    _candidate_target_distribution,
    evaluator_aligned_centerness_target,
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
