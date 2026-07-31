from __future__ import annotations

import importlib
import sys

import pytest
import torch

from rosa import ROSAConfig
from rosa.integrations.sparse4d_runtime import (
    NativeRefineCall,
    SingleCallCapture,
    detached_fpn,
    remap_legacy_refiner_state_dict,
    validate_native_call,
)
from rosa.integrations.sparse4d_scores import (
    compose_quality_rank,
    stable_joint_probability,
)


def test_integration_package_does_not_eagerly_import_sparse4d():
    package = importlib.import_module("rosa.integrations")
    assert package.__name__ == "rosa.integrations"
    assert "rosa.integrations.sparse4d" not in sys.modules


def test_released_config_aliases_keep_legacy_projection_default():
    config = ROSAConfig.from_mapping(
        {
            "radial_proposal_mode": "learned_log_gaussian",
            "second_step_candidate_search": False,
            "enable_shape1": True,
            "enable_shape2_feedback": True,
            "max_shape2_log_size_delta": 0.05,
            "max_shape2_yaw_delta_rad": 0.08,
        }
    )
    assert config.projection_contract == "legacy_front_only"
    assert config.enable_shape_refinement
    assert config.radial_proposal_mode == "learned_log_gaussian"


def test_native_refine_capture_forces_and_validates_scores():
    capture = SingleCallCapture(force_return_cls=True)
    args = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    kwargs = {
        "time_interval": None,
        "return_cls": False,
        "classification_feature": None,
        "regression_feature": None,
        "score_alignment_feature": None,
    }
    changed = capture(None, args, kwargs)
    assert changed is not None
    assert changed[1]["return_cls"] is True
    call = capture.take()
    validate_native_call(call)


def test_native_refine_contract_accepts_upstream_minimal_signature():
    call = NativeRefineCall(
        (torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        {"time_interval": None, "return_cls": True},
    )
    validate_native_call(call)

    extended = NativeRefineCall(
        call.args,
        {
            **call.kwargs,
            "classification_feature": None,
            "regression_feature": None,
            "score_alignment_feature": None,
        },
    )
    validate_native_call(extended)

    with pytest.raises(RuntimeError, match="signature changed"):
        validate_native_call(
            NativeRefineCall(
                call.args,
                {**call.kwargs, "unknown_sidecar": None},
            )
        )
    with pytest.raises(RuntimeError, match="signature changed"):
        validate_native_call(
            NativeRefineCall(
                call.args,
                {**call.kwargs, "classification_feature": None},
            )
        )
    with pytest.raises(RuntimeError, match="sidecars must be None"):
        validate_native_call(
            NativeRefineCall(
                extended.args,
                {**extended.kwargs, "classification_feature": torch.zeros(1)},
            )
        )


def test_fpn_detach_accepts_framework_neutral_level_tensors():
    source = [
        torch.randn(2, 4, 8, 5, 7, requires_grad=True),
        torch.randn(2, 4, 8, 3, 4, requires_grad=True),
    ]
    restored = detached_fpn(source)
    assert [value.shape for value in restored] == [value.shape for value in source]
    assert all(not value.requires_grad and value.grad_fn is None for value in restored)


def test_legacy_checkpoint_prefix_remaps_without_overwriting_modern_key():
    old_weight = torch.tensor([1.0])
    modern_weight = torch.tensor([2.0])
    state = {
        "head.canonical_sixth_block.a": old_weight,
        "head.canonical_sixth_block.b": old_weight,
        "head.rosa_refiner.b": modern_weight,
    }
    consumed = remap_legacy_refiner_state_dict(state, prefix="head.")
    assert consumed == 2
    assert not any("canonical_sixth_block" in key for key in state)
    assert state["head.rosa_refiner.a"] is old_weight
    assert state["head.rosa_refiner.b"] is modern_weight


def test_zero_quality_delta_is_exact_native_joint_score():
    classification = torch.tensor([[[0.5, -0.25], [1.0, 0.1]]])
    quality = torch.tensor([[[0.2], [-0.3]]])
    delta = torch.zeros(1, 2, 1, requires_grad=True)
    supported = torch.ones(1, 2, dtype=torch.bool)
    rank = compose_quality_rank(
        classification,
        quality,
        delta,
        supported,
        centerness_index=0,
    )
    assert torch.equal(
        rank["emitted_joint_score"],
        rank["native_joint_score"],
    )
    stable_joint_probability(
        rank["joint_logit"],
        rank["native_joint_score"],
        rank["effective_delta"],
    ).sum().backward()
    assert delta.grad is not None
