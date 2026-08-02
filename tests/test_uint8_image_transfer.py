"""Contracts for compact C32 host-to-device image transfer."""

from copy import deepcopy
from pathlib import Path

from mmengine import Config
from mmengine.dataset import default_collate
import numpy as np
import pytest
import torch
from mmdet3d.structures import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.datasets.transforms.pack_sparse4d import (
    PackSparse4DInputs,
)

from rosa.integrations.fisheye3dod.dataset import (
    LoadOfficialFisheye3DODLeanForSparse4D,
)
from rosa.integrations.fisheye3dod.model import (
    Fisheye3DODSingleFrameDataPreprocessor,
)
from rosa.integrations.fisheye3dod.runtime_transforms import (
    PackFisheye3DODUint8Inputs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
C32_STEM = "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual"
PARENT_CONFIG = f"{C32_STEM}_cpu_metadata_lean_loader.py"
CANDIDATE_CONFIG = f"{C32_STEM}_cpu_metadata_lean_loader_uint8_transfer.py"
CAMERAS = (
    "fisheye_camera_front",
    "fisheye_camera_left",
    "fisheye_camera_right",
    "fisheye_camera_rear",
)


class _Calibration:
    _invpol = [100.0, 0.0]
    _xc = 400.0
    _yc = 400.0
    _affine = [1.0, 0.0, 0.0]
    _fov = 220.0


def _results():
    return {
        "cam_info": {
            "cam_fisheye": {
                camera: {
                    "cam_path": f"{camera}.png",
                    "cam2img": _Calibration(),
                    "lidar2cam": np.eye(4, dtype=np.float32),
                }
                for camera in CAMERAS
            }
        },
        "ego2global": np.eye(4, dtype=np.float32),
        "timestamp": 123_000,
        "sample_idx": 7,
        "scene_token": "test-scene",
        "ann_info": {
            "gt_bboxes_3d": LiDARInstance3DBoxes(
                torch.tensor(
                    ((5.0, -2.0, -1.0, 2.0, 4.0, 1.5, 0.25),),
                    dtype=torch.float32,
                ),
                origin=(0.5, 0.5, 0.5),
            ),
            "gt_labels_3d": np.asarray((3,), dtype=np.int64),
            "instance_inds": np.asarray((91,), dtype=np.int64),
        },
        "eval_ann_info": {
            "gt_names": np.asarray(("Bus",)),
            "sample_idx": 7,
        },
    }


def _preprocessor():
    return Fisheye3DODSingleFrameDataPreprocessor(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True,
        pad_size_divisor=32,
    )


def _assert_exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif hasattr(left, "tensor") and isinstance(left.tensor, torch.Tensor):
        _assert_exact(left.tensor, right.tensor)
        assert left.box_dim == right.box_dim
    elif isinstance(left, dict):
        assert tuple(left) == tuple(right)
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_exact(left_item, right_item)
    else:
        assert left == right


@pytest.mark.parametrize("training", (False, True))
def test_uint8_transfer_reaches_model_as_bitwise_equal_fp32(monkeypatch, training):
    image = np.arange(800 * 800 * 3, dtype=np.uint8).reshape(800, 800, 3)
    monkeypatch.setattr(
        "rosa.integrations.fisheye3dod.dataset.mmcv.imread",
        lambda *_args, **_kwargs: image.copy(),
    )
    parent_loaded = LoadOfficialFisheye3DODLeanForSparse4D().transform(_results())
    candidate_loaded = LoadOfficialFisheye3DODLeanForSparse4D(
        to_float32=False
    ).transform(_results())
    parent_packed = PackSparse4DInputs(ocam_lens_fov_deg=220.0).transform(parent_loaded)
    candidate_packed = PackFisheye3DODUint8Inputs(ocam_lens_fov_deg=220.0).transform(
        candidate_loaded
    )

    assert parent_packed["inputs"]["img"].dtype == torch.float32
    assert candidate_packed["inputs"]["img"].dtype == torch.uint8
    assert parent_packed["inputs"]["img"].numel() == 4 * 3 * 800 * 800
    assert candidate_packed["inputs"]["img"].element_size() == 1
    assert parent_packed["inputs"]["img"].element_size() == 4

    parent_batch = default_collate([parent_packed])
    candidate_batch = default_collate([candidate_packed])
    parent_output = _preprocessor()(parent_batch, training=training)
    candidate_output = _preprocessor()(candidate_batch, training=training)
    _assert_exact(parent_output["inputs"], candidate_output["inputs"])
    assert len(parent_output["data_samples"]) == len(candidate_output["data_samples"])
    for parent_sample, candidate_sample in zip(
        parent_output["data_samples"], candidate_output["data_samples"]
    ):
        _assert_exact(parent_sample.metainfo, candidate_sample.metainfo)
        _assert_exact(parent_sample.to_dict(), candidate_sample.to_dict())


def test_uint8_packer_rejects_nonreleased_dtype_and_shape():
    with pytest.raises(TypeError, match="requires released uint8"):
        PackFisheye3DODUint8Inputs._pack_img(
            np.zeros((4, 800, 800, 3), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="camera, channel"):
        PackFisheye3DODUint8Inputs._pack_img(np.zeros((4, 800, 800, 2), dtype=np.uint8))


def test_uint8_config_has_only_the_audited_pipeline_and_import_deltas():
    parent = Config.fromfile(
        CONFIG_DIR / PARENT_CONFIG,
        import_custom_modules=False,
    ).to_dict()
    candidate = Config.fromfile(
        CONFIG_DIR / CANDIDATE_CONFIG,
        import_custom_modules=False,
    ).to_dict()

    parent_imports = parent["custom_imports"]["imports"]
    candidate_imports = candidate["custom_imports"]["imports"]
    assert candidate_imports == parent_imports + [
        "rosa.integrations.fisheye3dod.runtime_transforms"
    ]
    normalized = deepcopy(candidate)
    normalized["custom_imports"] = deepcopy(parent["custom_imports"])
    pipeline_paths = (
        ("train_dataloader", "dataset", "dataset", "pipeline"),
        ("val_dataloader", "dataset", "pipeline"),
        ("test_dataloader", "dataset", "pipeline"),
    )
    for path in pipeline_paths:
        parent_pipeline = parent
        candidate_pipeline = normalized
        for key in path:
            parent_pipeline = parent_pipeline[key]
            candidate_pipeline = candidate_pipeline[key]
        assert parent_pipeline[0]["type"] == candidate_pipeline[0]["type"]
        assert parent_pipeline[0]["to_float32"] is True
        candidate_pipeline[0]["to_float32"] = True
        assert parent_pipeline[-1]["type"] == "PackSparse4DInputs"
        assert candidate_pipeline[-1]["type"] == "PackFisheye3DODUint8Inputs"
        candidate_pipeline[-1]["type"] = "PackSparse4DInputs"
    for view in (parent, normalized):
        view.pop("work_dir")
        view.pop("experiment_contract")
    assert normalized == parent


def test_uint8_config_adds_no_model_state_or_numeric_user_setting():
    candidate = Config.fromfile(
        CONFIG_DIR / CANDIDATE_CONFIG,
        import_custom_modules=False,
    )

    assert candidate.experiment_contract.learned_parameters_added == 0
    assert candidate.experiment_contract.buffers_added == 0
    assert candidate.experiment_contract.user_numeric_hyperparameters_added == 0
    uint8_bytes = 4 * 4 * 3 * 800 * 800
    fp32_bytes = uint8_bytes * torch.empty((), dtype=torch.float32).element_size()
    assert candidate.experiment_contract.host_image_bytes_per_batch4_parent == (
        fp32_bytes
    )
    assert candidate.experiment_contract.host_image_bytes_per_batch4_candidate == (
        uint8_bytes
    )
    assert (
        candidate.experiment_contract.theoretical_h2d_byte_reduction_per_batch4
        == fp32_bytes - uint8_bytes
    )
