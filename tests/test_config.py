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


@pytest.mark.parametrize(
    "name",
    ("distribution_nll_weight", "intermediate_step_loss_weight"),
)
def test_release_loss_weights_must_be_positive(name):
    with pytest.raises(ValueError, match=name):
        ROSAConfig(**{name: float("nan")})
