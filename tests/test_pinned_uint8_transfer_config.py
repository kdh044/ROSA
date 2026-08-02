"""Configuration contract for opt-in pinned C32 image transfer."""

from copy import deepcopy
from pathlib import Path

from mmengine import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
STEM = (
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer"
)
PARENT_CONFIG = f"{STEM}.py"
CANDIDATE_CONFIG = f"{STEM}_pinned.py"


def _resolved(name: str) -> dict:
    return Config.fromfile(
        CONFIG_DIR / name,
        import_custom_modules=False,
    ).to_dict()


def test_pinned_candidate_changes_only_transfer_runtime_flags():
    parent = _resolved(PARENT_CONFIG)
    candidate = _resolved(CANDIDATE_CONFIG)

    assert "pin_memory" not in parent["train_dataloader"]
    assert "pin_memory" not in parent["val_dataloader"]
    assert "pin_memory" not in parent["test_dataloader"]
    for loader in ("train_dataloader", "val_dataloader", "test_dataloader"):
        assert candidate[loader]["pin_memory"] is True

    assert parent["model"]["data_preprocessor"].get("non_blocking", False) is False
    assert candidate["model"]["data_preprocessor"]["non_blocking"] is True

    normalized = deepcopy(candidate)
    for loader in ("train_dataloader", "val_dataloader", "test_dataloader"):
        normalized[loader].pop("pin_memory")
    normalized["model"]["data_preprocessor"].pop("non_blocking")
    for view in (parent, normalized):
        view.pop("work_dir")
        view.pop("experiment_contract")
    assert normalized == parent


def test_pinned_candidate_adds_no_model_state_or_numeric_knob():
    candidate = Config.fromfile(
        CONFIG_DIR / CANDIDATE_CONFIG,
        import_custom_modules=False,
    )

    assert candidate.experiment_contract.learned_parameters_added == 0
    assert candidate.experiment_contract.buffers_added == 0
    assert candidate.experiment_contract.user_numeric_hyperparameters_added == 0
    assert candidate.experiment_contract.model_math == "unchanged"
    assert candidate.experiment_contract.runtime_claim.startswith("none;")
