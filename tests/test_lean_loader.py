"""Contracts for the C32 loader-only runtime optimization."""

from copy import deepcopy
from pathlib import Path

from mmengine import Config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
C32_STEM = "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual"
BASE_CONFIG = f"{C32_STEM}_cpu_metadata.py"
LEAN_CONFIG = f"{C32_STEM}_cpu_metadata_lean_loader.py"


def _config(name):
    return Config.fromfile(
        CONFIG_DIR / name,
        import_custom_modules=False,
    ).to_dict()


def _loader_types(config):
    return (
        config["train_dataloader"]["dataset"]["dataset"]["pipeline"][0]["type"],
        config["val_dataloader"]["dataset"]["pipeline"][0]["type"],
        config["test_dataloader"]["dataset"]["pipeline"][0]["type"],
    )


def test_lean_config_changes_only_all_three_loader_types():
    baseline = _config(BASE_CONFIG)
    candidate = _config(LEAN_CONFIG)

    assert _loader_types(baseline) == ("LoadOfficialFisheye3DODForSparse4D",) * 3
    assert _loader_types(candidate) == ("LoadOfficialFisheye3DODLeanForSparse4D",) * 3
    normalized = deepcopy(candidate)
    normalized["train_dataloader"]["dataset"]["dataset"]["pipeline"][0][
        "type"
    ] = "LoadOfficialFisheye3DODForSparse4D"
    normalized["val_dataloader"]["dataset"]["pipeline"][0][
        "type"
    ] = "LoadOfficialFisheye3DODForSparse4D"
    normalized["test_dataloader"]["dataset"]["pipeline"][0][
        "type"
    ] = "LoadOfficialFisheye3DODForSparse4D"
    for view in (baseline, normalized):
        view.pop("work_dir")
        view.pop("experiment_contract")
    assert normalized == baseline


def test_lean_loader_adds_no_numeric_user_setting():
    candidate = Config.fromfile(
        CONFIG_DIR / LEAN_CONFIG,
        import_custom_modules=False,
    )

    assert candidate.experiment_contract.learned_parameters_added == 0
    assert candidate.experiment_contract.buffers_added == 0
    assert candidate.experiment_contract.user_numeric_hyperparameters_added == 0
