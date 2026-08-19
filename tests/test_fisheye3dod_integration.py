from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mmdet3d")
pytest.importorskip("projects.mmdet3d_plugin")

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.datasets.transforms.pack_sparse4d import (
    PackSparse4DInputs,
)

from rosa.integrations.fisheye3dod.calibration import (
    convert_lidar2cam,
    convert_ocamcamera,
    project_rosa_compatible_numpy,
)
from rosa.integrations.fisheye3dod.constants import (
    CAMERAS,
    EXPECTED_SCENES,
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_TRAIN_FRAMES_PER_SCENE,
    EXPECTED_VAL_FRAMES,
    EXPECTED_VAL_FRAMES_PER_SCENE,
)
from rosa.integrations.fisheye3dod.dataset import (
    Fisheye3DODEvaluatorRangeFilter,
    LoadOfficialFisheye3DODLeanForSparse4D,
    LoadOfficialFisheye3DODForSparse4D,
)


class _Calibration:
    _invpol = [100.0, 0.0]
    _xc = 400.0
    _yc = 400.0
    _affine = [1.0, 0.0, 0.0]
    _fov = 220.0


def _camera_results():
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
    }


def test_calibration_conversion_preserves_released_coefficients():
    calibration = convert_ocamcamera(_Calibration())
    np.testing.assert_array_equal(
        calibration.polynomial,
        np.asarray(_Calibration._invpol, dtype=np.float32),
    )
    assert calibration.lens_fov_deg == 220.0
    assert convert_lidar2cam(np.eye(4)).shape == (4, 4)


def test_numpy_reference_separates_legacy_and_full_fov_contracts():
    calibration = convert_ocamcamera(_Calibration())
    angle = np.deg2rad(100.0)
    # ROSA OCam axes are [down, right, back].  This ray lies ten degrees
    # behind the optical plane but remains inside the declared 220° cone.
    ray = np.asarray(
        [[np.sin(angle), 0.0, -np.cos(angle)]],
        dtype=np.float64,
    )
    legacy = project_rosa_compatible_numpy(
        ray,
        np.eye(4),
        calibration,
        contract="legacy_front_only",
    )
    full_fov = project_rosa_compatible_numpy(
        ray,
        np.eye(4),
        calibration,
        contract="full_fov",
    )
    assert np.array_equal(legacy, np.full((1, 2), -1.0))
    assert np.all((full_fov >= 0) & (full_fov < 800))


def test_lean_native_loader_does_not_compute_or_emit_unused_cam2ego(monkeypatch):
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "rosa.integrations.fisheye3dod.dataset.mmcv.imread",
        lambda *_args, **_kwargs: image.copy(),
    )

    def forbidden_inverse(*_args, **_kwargs):
        raise AssertionError("the native pipeline does not consume cam2ego")

    monkeypatch.setattr(np.linalg, "inv", forbidden_inverse)
    output = LoadOfficialFisheye3DODLeanForSparse4D().transform(
        _camera_results()
    )

    assert "cam2ego" not in output
    assert output["lidar2cam"].shape == (4, 4, 4)
    assert output["lidar2cam"].dtype == np.float32
    assert len(output["img"]) == 4


def test_default_native_loader_keeps_optional_cam2ego_contract(monkeypatch):
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "rosa.integrations.fisheye3dod.dataset.mmcv.imread",
        lambda *_args, **_kwargs: image.copy(),
    )
    output = LoadOfficialFisheye3DODForSparse4D().transform(
        _camera_results()
    )

    assert output["cam2ego"].shape == (4, 4, 4)
    assert output["cam2ego"].dtype == np.float32
    identity = np.tile(np.eye(4, dtype=np.float32), (len(CAMERAS), 1, 1))
    np.testing.assert_allclose(
        output["cam2ego"] @ output["lidar2cam"],
        identity,
        rtol=0.0,
        atol=1e-6,
    )


def test_lean_loader_is_registered_without_replacing_the_default():
    assert TRANSFORMS.get("LoadOfficialFisheye3DODForSparse4D") is (
        LoadOfficialFisheye3DODForSparse4D
    )
    assert TRANSFORMS.get("LoadOfficialFisheye3DODLeanForSparse4D") is (
        LoadOfficialFisheye3DODLeanForSparse4D
    )


def test_lean_loader_preserves_every_packed_model_input_bitwise(monkeypatch):
    image = np.arange(800 * 800 * 3, dtype=np.uint8).reshape(800, 800, 3)
    monkeypatch.setattr(
        "rosa.integrations.fisheye3dod.dataset.mmcv.imread",
        lambda *_args, **_kwargs: image.copy(),
    )
    baseline = LoadOfficialFisheye3DODForSparse4D().transform(_camera_results())
    candidate = LoadOfficialFisheye3DODLeanForSparse4D().transform(
        _camera_results()
    )
    for result in (baseline, candidate):
        result["ann_info"] = {
            "gt_bboxes_3d": LiDARInstance3DBoxes(
                torch.tensor(
                    ((5.0, -2.0, -1.0, 2.0, 4.0, 1.5, 0.25),),
                    dtype=torch.float32,
                ),
                origin=(0.5, 0.5, 0.5),
            ),
            "gt_labels_3d": np.asarray((3,), dtype=np.int64),
            "instance_inds": np.asarray((91,), dtype=np.int64),
        }
        result["eval_ann_info"] = {
            "gt_names": np.asarray(("Bus",)),
            "sample_idx": 7,
        }
    packer = PackSparse4DInputs(ocam_lens_fov_deg=220.0)

    baseline_packed = packer.transform(baseline)
    candidate_packed = packer.transform(candidate)

    assert tuple(baseline_packed["inputs"]) == tuple(candidate_packed["inputs"])
    for name, baseline_value in baseline_packed["inputs"].items():
        candidate_value = candidate_packed["inputs"][name]
        if isinstance(baseline_value, torch.Tensor):
            assert baseline_value.dtype == candidate_value.dtype
            assert torch.equal(baseline_value, candidate_value), name
        else:
            assert baseline_value == candidate_value, name
    assert baseline_packed["data_samples"].metainfo == (
        candidate_packed["data_samples"].metainfo
    )
    baseline_gt = baseline_packed["data_samples"].gt_instances_3d
    candidate_gt = candidate_packed["data_samples"].gt_instances_3d
    assert torch.equal(baseline_gt.bboxes_3d.tensor, candidate_gt.bboxes_3d.tensor)
    assert torch.equal(baseline_gt.labels_3d, candidate_gt.labels_3d)
    assert torch.equal(baseline_gt.instance_inds, candidate_gt.instance_inds)
    baseline_eval = baseline_packed["data_samples"].eval_ann_info
    candidate_eval = candidate_packed["data_samples"].eval_ann_info
    assert baseline_eval.keys() == candidate_eval.keys()
    np.testing.assert_array_equal(
        baseline_eval["gt_names"], candidate_eval["gt_names"]
    )
    assert baseline_eval["sample_idx"] == candidate_eval["sample_idx"]


def test_evaluator_range_filter_matches_strict_class_radial_contract():
    raw = torch.tensor(
        [
            [47.9, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [48.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [37.0, 0.0, 0.0, 1.0, 1.0, 1.7, 0.0],
            [35.9, 0.0, 0.0, 1.0, 1.0, 1.7, 0.0],
            [40.0, 40.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [1.0, 0.0, 100.0, 2.0, 4.0, 1.5, 0.0],
        ]
    )
    labels = np.asarray([0, 0, 4, 4, 0, 0], dtype=np.int64)
    annotation = {
        "gt_bboxes_3d": LiDARInstance3DBoxes(raw),
        "gt_labels_3d": labels,
        "instance_inds": np.arange(len(labels)) + 100,
    }
    result = Fisheye3DODEvaluatorRangeFilter()(
        {"ann_info": annotation}
    )
    filtered = result["ann_info"]
    assert filtered["gt_labels_3d"].tolist() == [0, 4, 0]
    assert filtered["instance_inds"].tolist() == [100, 103, 105]
    torch.testing.assert_close(
        filtered["gt_bboxes_3d"].tensor[:, :3],
        raw[[0, 3, 5], :3],
    )


def test_main126_raw_z_is_restored_as_gravity_center():
    raw = torch.tensor(
        [[4.0, -2.0, -1.3, 2.0, 4.0, 1.6, 0.0]],
    )
    boxes = LiDARInstance3DBoxes(
        raw,
        origin=(0.5, 0.5, 0.5),
    )

    # MMDetection3D stores LiDAR boxes at bottom center internally, while the
    # released Main126 pickle declares its input at the gravity center.
    torch.testing.assert_close(boxes.tensor[:, 2], raw[:, 2] - raw[:, 5] * 0.5)
    torch.testing.assert_close(boxes.gravity_center, raw[:, :3])


def test_main126_split_totals_match_the_released_scene_contract():
    assert EXPECTED_SCENES == 126
    assert EXPECTED_TRAIN_FRAMES == (
        EXPECTED_SCENES * EXPECTED_TRAIN_FRAMES_PER_SCENE
    )
    assert EXPECTED_VAL_FRAMES == (
        EXPECTED_SCENES * EXPECTED_VAL_FRAMES_PER_SCENE
    )
