"""Contracts for the prediction-only owned-buffer normalization path."""

from copy import deepcopy
from pathlib import Path

from mmengine import Config
import pytest
import torch

from rosa.integrations.fisheye3dod.model import (
    Fisheye3DODSingleFrameDataPreprocessor,
    Fisheye3DODSingleFrameInplaceDataPreprocessor,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
STEM = (
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer_pinned"
)
PARENT_CONFIG = f"{STEM}.py"
CANDIDATE_CONFIG = f"{STEM}_inplace.py"


def _preprocessor(kind):
    return kind(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True,
        pad_size_divisor=32,
        non_blocking=True,
    )


@pytest.mark.parametrize("shape", ((2, 4, 3, 32, 64), (1, 4, 3, 33, 35)))
@pytest.mark.parametrize("seed", (3407, 7101080))
def test_inplace_uint8_processing_is_cpu_raw_bit_equal_and_input_safe(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    image = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
    original = image.clone()

    expected = _preprocessor(Fisheye3DODSingleFrameDataPreprocessor)._process_img(
        image
    )
    actual = _preprocessor(
        Fisheye3DODSingleFrameInplaceDataPreprocessor
    )._process_img(image)

    assert torch.equal(actual, expected)
    assert actual.dtype is torch.float32
    assert torch.equal(image, original)
    assert actual.untyped_storage().data_ptr() != image.untyped_storage().data_ptr()


def test_inplace_uint8_processing_fails_closed_outside_owned_ingress():
    candidate = _preprocessor(Fisheye3DODSingleFrameInplaceDataPreprocessor)
    with pytest.raises(TypeError, match="torch.uint8 ingress"):
        candidate._process_img(torch.zeros(1, 4, 3, 32, 32))
    with pytest.raises(RuntimeError, match="prediction-only"):
        candidate({}, training=True)

    no_conversion = Fisheye3DODSingleFrameInplaceDataPreprocessor(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=False,
        pad_size_divisor=32,
    )
    with pytest.raises(ValueError, match="BGR-to-RGB"):
        no_conversion._process_img(
            torch.zeros(1, 4, 3, 32, 32, dtype=torch.uint8)
        )


def test_inplace_uint8_full_prediction_preprocessor_is_cpu_raw_bit_equal():
    generator = torch.Generator().manual_seed(3407)
    image = torch.randint(
        0,
        256,
        (2, 4, 3, 33, 35),
        generator=generator,
        dtype=torch.uint8,
    )
    data = {
        "inputs": {
            "img": image,
            "T_global": torch.eye(4).repeat(2, 1, 1),
            "T_global_inv": torch.eye(4).repeat(2, 1, 1),
            "timestamp": torch.tensor((1.0, 2.0)),
            "image_wh": torch.full((2, 4, 2), 800.0),
        }
    }
    parent = _preprocessor(Fisheye3DODSingleFrameDataPreprocessor)(
        deepcopy(data),
        training=False,
    )
    candidate = _preprocessor(Fisheye3DODSingleFrameInplaceDataPreprocessor)(
        deepcopy(data),
        training=False,
    )

    assert tuple(parent["inputs"]) == tuple(candidate["inputs"])
    for name in parent["inputs"]:
        assert torch.equal(parent["inputs"][name], candidate["inputs"][name])


def test_inplace_config_changes_only_preprocessor_type():
    parent = Config.fromfile(
        CONFIG_DIR / PARENT_CONFIG,
        import_custom_modules=False,
    ).to_dict()
    candidate = Config.fromfile(
        CONFIG_DIR / CANDIDATE_CONFIG,
        import_custom_modules=False,
    ).to_dict()

    assert parent["model"]["data_preprocessor"]["type"] == (
        "Fisheye3DODSingleFrameDataPreprocessor"
    )
    assert candidate["model"]["data_preprocessor"]["type"] == (
        "Fisheye3DODSingleFrameInplaceDataPreprocessor"
    )
    normalized = deepcopy(candidate)
    normalized["model"]["data_preprocessor"]["type"] = parent["model"][
        "data_preprocessor"
    ]["type"]
    for view in (parent, normalized):
        view.pop("work_dir")
        view.pop("experiment_contract")
    assert normalized == parent


def test_inplace_config_adds_no_state_or_numeric_knob():
    candidate = Config.fromfile(
        CONFIG_DIR / CANDIDATE_CONFIG,
        import_custom_modules=False,
    )
    contract = candidate.experiment_contract
    assert contract.logical_fp32_temporary_bytes_removed_per_batch4 == (
        2
        * 4
        * 4
        * 3
        * 800
        * 800
        * torch.empty((), dtype=torch.float32).element_size()
    )
    assert contract.learned_parameters_added == 0
    assert contract.buffers_added == 0
    assert contract.user_numeric_hyperparameters_added == 0
