"""Contracts for RGB ordering fused into compact uint8 packing."""

from copy import deepcopy
from pathlib import Path

from mmengine import Config
import numpy as np
import pytest
import torch

from rosa.integrations.fisheye3dod.model import (
    Fisheye3DODSingleFrameInplaceDataPreprocessor,
)
from rosa.integrations.fisheye3dod.runtime_transforms import (
    PackFisheye3DODRGBUint8Inputs,
    PackFisheye3DODUint8Inputs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
PARENT_CONFIG = (
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_uint8_transfer_pinned_inplace.py"
)
CANDIDATE_CONFIG = (
    "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual_"
    "cpu_metadata_lean_loader_rgb_uint8_transfer_pinned_inplace.py"
)


def _preprocessor(*, bgr_to_rgb: bool):
    return Fisheye3DODSingleFrameInplaceDataPreprocessor(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=bgr_to_rgb,
        pad_size_divisor=32,
    )


@pytest.mark.parametrize("shape", ((4, 32, 64, 3), (4, 33, 35, 3)))
@pytest.mark.parametrize("seed", (3407, 7101080))
@pytest.mark.parametrize("as_list", (False, True))
def test_rgb_fused_pack_reaches_model_as_cpu_raw_bit_equal(shape, seed, as_list):
    rng = np.random.default_rng(seed)
    bgr = rng.integers(0, 256, shape, dtype=np.uint8)
    ingress = list(bgr) if as_list else bgr
    input_bytes = bgr.tobytes()
    parent_packed = PackFisheye3DODUint8Inputs._pack_img(ingress)
    candidate_packed = PackFisheye3DODRGBUint8Inputs._pack_img(ingress)

    assert torch.equal(candidate_packed, parent_packed.flip(dims=(1,)))
    assert bgr.tobytes() == input_bytes
    assert candidate_packed.is_contiguous()
    assert candidate_packed.dtype == torch.uint8
    parent = _preprocessor(bgr_to_rgb=True)._process_img(parent_packed[None])
    candidate = _preprocessor(bgr_to_rgb=False)._process_img(candidate_packed[None])
    assert torch.equal(candidate, parent)


def test_rgb_list_packer_writes_stack_directly_to_the_output(monkeypatch):
    rng = np.random.default_rng(3407)
    images = [rng.integers(0, 256, (32, 64, 3), dtype=np.uint8) for _ in range(4)]
    expected = torch.from_numpy(
        np.ascontiguousarray(np.transpose(np.stack(images)[..., ::-1], (0, 3, 1, 2)))
    )

    original_stack = np.stack

    def require_output(*args, **kwargs):
        assert kwargs.get("out") is not None
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(np, "stack", require_output)
    actual = PackFisheye3DODRGBUint8Inputs._pack_img(images)
    assert torch.equal(actual, expected)


def test_rgb_list_packer_rejects_mixed_dtype_and_shape():
    valid = np.zeros((32, 64, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="released uint8"):
        PackFisheye3DODRGBUint8Inputs._pack_img([valid, valid.astype(np.float32)])
    with pytest.raises(ValueError, match="share one HWC shape"):
        PackFisheye3DODRGBUint8Inputs._pack_img(
            [valid, np.zeros((33, 64, 3), dtype=np.uint8)]
        )


def test_rgb_fused_pack_requires_released_three_channel_uint8():
    with pytest.raises(TypeError, match="released uint8"):
        PackFisheye3DODRGBUint8Inputs._pack_img(
            np.zeros((4, 32, 32, 3), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="exactly three"):
        PackFisheye3DODRGBUint8Inputs._pack_img(
            np.zeros((4, 32, 32, 1), dtype=np.uint8)
        )
    with pytest.raises(ValueError, match="released HWC"):
        PackFisheye3DODRGBUint8Inputs._pack_img(
            np.zeros((4, 3, 32, 32), dtype=np.uint8)
        )


def test_rgb_fused_config_changes_only_test_representation_contract():
    parent = Config.fromfile(
        CONFIG_DIR / PARENT_CONFIG,
        import_custom_modules=False,
    ).to_dict()
    candidate = Config.fromfile(
        CONFIG_DIR / CANDIDATE_CONFIG,
        import_custom_modules=False,
    ).to_dict()

    assert parent["model"]["data_preprocessor"]["bgr_to_rgb"] is True
    assert candidate["model"]["data_preprocessor"]["bgr_to_rgb"] is False
    for loader in ("val_dataloader", "test_dataloader"):
        parent_pipeline = parent[loader]["dataset"]["pipeline"]
        candidate_pipeline = candidate[loader]["dataset"]["pipeline"]
        assert parent_pipeline[:-1] == candidate_pipeline[:-1]
        assert parent_pipeline[-1]["type"] == "PackFisheye3DODUint8Inputs"
        assert candidate_pipeline[-1]["type"] == ("PackFisheye3DODRGBUint8Inputs")

    normalized = deepcopy(candidate)
    normalized["model"]["data_preprocessor"]["bgr_to_rgb"] = True
    for loader in ("val_dataloader", "test_dataloader"):
        normalized[loader]["dataset"]["pipeline"][-1][
            "type"
        ] = "PackFisheye3DODUint8Inputs"
    for view in (parent, normalized):
        view.pop("work_dir")
        view.pop("experiment_contract")
    assert normalized == parent
