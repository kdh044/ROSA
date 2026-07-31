from copy import deepcopy
from pathlib import Path

import pytest
import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

from rosa.integrations.sparse4d_losses import (
    evaluator_aligned_centerness_target,
    evaluator_aligned_quality_target,
    validate_centerness_target_mode,
)
from rosa.integrations.sparse4d_scores import (
    compose_quality_rank,
    jointscore_quality_focal_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> Config:
    return Config.fromfile(ROOT / "configs" / name)


def test_default_centerness_target_is_bitwise_the_exp_bev_contract():
    prediction = torch.tensor(
        [[[0.25, -0.75, 9.0], [3.0, 4.0, -5.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    expected = torch.exp(
        -torch.linalg.vector_norm(target[..., :2] - prediction[..., :2], dim=-1)
    ).detach()

    observed = evaluator_aligned_centerness_target(prediction, target)

    assert torch.equal(observed, expected)
    assert not observed.requires_grad


def test_strict_centerness_target_uses_official_open_distance_boundaries():
    prediction = torch.zeros(1, 6, 3)
    prediction[..., 0] = torch.tensor((0.49, 0.50, 0.75, 1.50, 3.00, 4.00))
    target = torch.zeros_like(prediction)

    observed = evaluator_aligned_centerness_target(
        prediction,
        target,
        mode="strict_ap_fraction",
    )

    torch.testing.assert_close(
        observed,
        torch.tensor([[1.0, 0.75, 0.75, 0.50, 0.25, 0.0]]),
    )


def test_centerness_target_mode_fails_closed():
    with pytest.raises(ValueError, match="centerness_target_mode"):
        validate_centerness_target_mode("smooth_guess")


def test_c3_changes_only_the_native_regression_confidence_gate():
    c0 = _config("rosa_r18_main126_v2_c0_cbgs_native.py")
    c3 = _config("rosa_r18_main126_v2_c3_cbgs_native_no_reg_gate.py")
    expected_model = deepcopy(c0.model)
    expected_model.head.cls_threshold_to_reg = -1

    assert c3.model == expected_model
    assert c3.model.head.type == "ROSANativeBaselineHead"
    assert c3.model.head.rosa_cfg is None
    assert c3.model.head.cls_threshold_to_reg == -1
    assert c3.train_dataloader == c0.train_dataloader
    assert c3.train_cfg == c0.train_cfg
    assert c3.param_scheduler == c0.param_scheduler
    assert c3.optim_wrapper == c0.optim_wrapper
    assert c3.randomness == c0.randomness
    assert c3.experiment_contract.train_sampling == (
        "mmdet3d.CBGSDataset(raw_pre_pipeline_categories)"
    )


def test_c4_changes_only_centerness_mode_and_preserves_nested_loss_config():
    c0 = _config("rosa_r18_main126_v2_c0_cbgs_native.py")
    c4 = _config("rosa_r18_main126_v2_c4_cbgs_native_strict_centerness.py")
    expected_model = deepcopy(c0.model)
    expected_model.head.loss_reg.centerness_target_mode = "strict_ap_fraction"

    assert c4.model == expected_model
    assert c4.model.head.type == "ROSANativeBaselineHead"
    assert c4.model.head.rosa_cfg is None
    assert c4.model.head.cls_threshold_to_reg == 0.05
    assert c4.model.head.loss_reg.type == "ROSAEvaluatorAlignedSparseBox3DLoss"
    assert c4.model.head.loss_reg.loss_box == c0.model.head.loss_reg.loss_box
    assert c4.model.head.loss_reg.loss_centerness == (
        c0.model.head.loss_reg.loss_centerness
    )
    assert c4.model.head.loss_reg.loss_yawness == (c0.model.head.loss_reg.loss_yawness)
    assert c4.model.head.loss_reg.centerness_target_mode == ("strict_ap_fraction")
    assert c4.train_dataloader == c0.train_dataloader
    assert c4.train_cfg == c0.train_cfg
    assert c4.param_scheduler == c0.param_scheduler
    assert c4.optim_wrapper == c0.optim_wrapper
    assert c4.randomness == c0.randomness


def test_c4_loss_build_is_opt_in_stateless_and_invalid_modes_fail_closed():
    register_all_modules(init_default_scope=True)
    c0 = _config("rosa_r18_main126_v2_c0_cbgs_native.py")
    c4 = _config("rosa_r18_main126_v2_c4_cbgs_native_strict_centerness.py")
    import_modules_from_strings(**c4.custom_imports)

    default_loss = MODELS.build(c0.model.head.loss_reg)
    strict_loss = MODELS.build(c4.model.head.loss_reg)
    assert default_loss.centerness_target_mode == "exp_bev"
    assert strict_loss.centerness_target_mode == "strict_ap_fraction"
    assert default_loss.state_dict().keys() == strict_loss.state_dict().keys()

    invalid = deepcopy(c4.model.head.loss_reg)
    invalid.centerness_target_mode = "smooth_guess"
    with pytest.raises(ValueError, match="centerness_target_mode"):
        MODELS.build(invalid)


def test_quality_target_rejects_unmatched_and_wrong_class_queries():
    prediction = torch.zeros(1, 3, 3)
    target = torch.zeros_like(prediction)
    matched = torch.tensor([[True, False, True]])
    predicted_class = torch.tensor([[2, 2, 1]])
    target_class = torch.tensor([[2, 2, 2]])

    quality = evaluator_aligned_quality_target(
        prediction,
        target,
        matched,
        predicted_class,
        target_class,
    )

    torch.testing.assert_close(quality, torch.tensor([[1.0, 0.0, 0.0]]))


def test_quality_loss_cannot_train_native_scores_or_unsupported_delta():
    classification = torch.tensor(
        [[[2.0, -1.0], [1.0, -2.0]]],
        requires_grad=True,
    )
    centerness = torch.zeros(1, 2, 1, requires_grad=True)
    delta = torch.zeros(1, 2, 1, requires_grad=True)
    supported = torch.tensor([[True, False]])
    rank = compose_quality_rank(
        classification,
        centerness,
        delta,
        supported,
        centerness_index=0,
    )

    loss = jointscore_quality_focal_loss(
        rank["joint_logit"],
        rank["effective_delta"],
        rank["native_joint_score"],
        torch.ones(1, 2),
        rank["eligible"],
        gamma=2.0,
    ).loss
    loss.backward()

    assert classification.grad is None
    assert centerness.grad is None
    assert delta.grad is not None
    assert delta.grad[0, 0, 0] != 0
    assert delta.grad[0, 1, 0] == 0
