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
    assert not config.anchor_z_is_bottom
    assert config.yaw_class_loss_weights is None


def test_fixed_base_center_frame_is_an_explicit_corrected_ablation():
    config = ROSAConfig.from_mapping(
        {"profile": "fast", "center_target_mode": "fixed_base"}
    )
    assert config.center_target_mode == "fixed_base"


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


@pytest.mark.parametrize("profile", ("compact_v2", "compact_v2_fast"))
def test_dense_compact_v2_profile_names_fail_closed(profile):
    with pytest.raises(ValueError, match="retired dense compact v2 topology"):
        ROSAConfig.from_mapping({"profile": profile})


def test_rfl_topology_is_owned_by_the_behavior_profile():
    assert (
        ROSAConfig.from_mapping({"profile": "balanced"}).rfl_topology_contract
        == "rosa.rfl.signed_gated_cp16.v3"
    )
    assert (
        ROSAConfig.from_mapping({"profile": "fast"}).rfl_topology_contract
        == "rosa.rfl.signed_gated_cp16.v3"
    )
    assert (
        ROSAConfig.from_mapping({"profile": "published"}).rfl_topology_contract
        == "rosa.rfl.dense_outer_product.v2"
    )


def test_published_profile_keeps_the_release_alias():
    assert ROSAConfig.from_mapping({"profile": "published"}) == ROSAConfig.from_mapping(
        {"profile": "legacy_release"}
    )


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


@pytest.mark.parametrize(
    "weights",
    (
        1.0,
        (1.0,) * 5,
        (1.0,) * 5 + (0.0,),
        (1.0,) * 5 + (float("inf"),),
    ),
)
def test_yaw_class_loss_weights_require_six_positive_finite_values(weights):
    with pytest.raises(ValueError, match="six finite positive"):
        ROSAConfig.from_mapping(
            {
                "profile": "balanced",
                "yaw_class_loss_weights": weights,
            }
        )


def test_yaw_class_loss_weights_are_compact_ablation_only():
    with pytest.raises(ValueError, match="compact wrapped-yaw"):
        ROSAConfig(yaw_class_loss_weights=(1.0,) * 6)

    weights = (0.765, 1.320, 1.509, 2.041, 1.022, 1.264)
    config = ROSAConfig.from_mapping(
        {
            "profile": "balanced",
            "yaw_class_loss_weights": list(weights),
        }
    )
    assert config.yaw_class_loss_weights == weights
