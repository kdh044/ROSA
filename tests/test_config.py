import pytest

from rosa import ROSAConfig


def test_release_config_accepts_published_option_names():
    config = ROSAConfig.from_mapping(
        {
            "enable_shape1": False,
            "max_shape2_log_size_delta": 0.04,
            "max_shape2_yaw_delta_rad": 0.07,
            "radial_proposal_mode": "learned_log_gaussian",
            "second_step_candidate_search": False,
            "enable_shape2_feedback": True,
        }
    )
    assert not config.enable_shape_refinement
    assert config.max_second_log_size_delta == 0.04
    assert config.max_second_yaw_delta_rad == 0.07


def test_removed_experiment_path_is_rejected():
    with pytest.raises(ValueError, match="release model"):
        ROSAConfig.from_mapping({"radial_proposal_mode": "fixed"})


def test_direct_construction_is_validated():
    with pytest.raises(ValueError, match="one or two"):
        ROSAConfig(num_refinement_steps=3)


def test_balanced_profile_applies_one_auditable_preset():
    config = ROSAConfig.from_mapping({"profile": "balanced"})
    assert config.architecture_profile == "compact"
    assert config.hidden_dims == 64
    assert config.max_candidate_range_m == 68.0
    assert config.mask_unsupported_updates
    assert not config.enable_quality_score
    assert config.center_target_mode == "sequential"
    assert config.center_supervision_mode == "move_or_keep"
    assert config.yaw_update_mode == "guarded_shared"
    assert config.anchor_z_is_bottom


def test_fast_profile_only_changes_the_compute_budget():
    balanced = ROSAConfig.from_mapping({"profile": "balanced"})
    fast = ROSAConfig.from_mapping({"profile": "fast"})
    differing = {
        name
        for name in balanced.__dataclass_fields__
        if getattr(balanced, name) != getattr(fast, name)
    }
    assert differing == {"refine_topk", "num_refinement_steps"}
    assert fast.refine_topk == 50
    assert fast.num_refinement_steps == 1


def test_early_compact_profile_name_remains_an_exact_alias():
    assert ROSAConfig.from_mapping(
        {"profile": "compact_v2"}
    ) == ROSAConfig.from_mapping({"profile": "balanced"})


def test_published_profile_keeps_the_release_alias():
    assert ROSAConfig.from_mapping(
        {"profile": "published"}
    ) == ROSAConfig.from_mapping({"profile": "legacy_release"})


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown ROSA profile"):
        ROSAConfig.from_mapping({"profile": "guess"})


@pytest.mark.parametrize(
    "name",
    (
        "distribution_nll_weight",
        "intermediate_step_loss_weight",
        "yaw_loss_beta_rad",
    ),
)
def test_release_loss_weights_must_be_positive(name):
    with pytest.raises(ValueError, match=name):
        ROSAConfig(**{name: float("nan")})


@pytest.mark.parametrize(
    "name",
    (
        "shape_loss_weight",
        "yaw_loss_weight",
        "yaw_keep_loss_weight",
        "yaw_improvement_loss_weight",
        "candidate_loss_weight",
        "radial_offset_loss_weight",
        "center_gate_loss_weight",
        "center_move_loss_weight",
        "center_keep_loss_weight",
        "tangent_loss_weight",
        "quality_loss_weight",
    ),
)
def test_all_loss_weights_reject_negative_or_nonfinite(name):
    with pytest.raises(ValueError, match=name):
        ROSAConfig(**{name: -1.0})
    with pytest.raises(ValueError, match=name):
        ROSAConfig(**{name: float("inf")})
