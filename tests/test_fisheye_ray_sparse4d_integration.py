"""Optional host-level integration for the V2 fisheye ray head."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("mmdet3d")
pytest.importorskip("projects.mmdet3d_plugin")

from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import MODELS


def _metadata():
    transform = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 4, 1, 1)
    for camera in range(4):
        angle = torch.tensor(camera * torch.pi / 2)
        cosine, sine = angle.cos(), angle.sin()
        transform[:, camera, 0, 0] = cosine
        transform[:, camera, 0, 1] = -sine
        transform[:, camera, 1, 0] = sine
        transform[:, camera, 1, 1] = cosine
    polynomial = torch.zeros(1, 4, 5)
    polynomial[..., 0] = -235.0
    polynomial[..., 1] = -150.0
    intrinsic = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 4, 1, 1)
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    return {
        "lidar2cam": transform,
        "image_wh": torch.full((1, 4, 2), 800.0),
        "ocam_pol": polynomial,
        "ocam_intrinsic": intrinsic,
        "ocam_lens_fov_deg": 220.0,
        "gt_labels_3d": [torch.tensor([1], dtype=torch.long)],
        "gt_bboxes_3d": [
            torch.tensor([[10.0, 1.0, 0.5, 2.0, 4.0, 1.5, 0.1, 0.0, 0.0]])
        ],
    }


def test_real_head_direction_only_dn_loss_and_backward():
    root = Path(__file__).resolve().parents[1]
    config = Config.fromfile(
        root / "configs" / "fisheye_ray_decoder_v2_raw_uniform_20e.py"
    )
    forbidden_config = {
        "instance_bank",
        "anchor_encoder",
        "graph_model",
        "temp_graph_model",
        "deformable_model",
        "refine_layer",
        "rosa_cfg",
    }
    assert forbidden_config.isdisjoint(config.model.head)
    assert "initial_anchor" in config.model.head
    assert config.model.head.type == "FisheyeSparseRayHead"
    import_modules_from_strings(**config.custom_imports)
    head = MODELS.build(config.model.head).train()
    assert not hasattr(head, "instance_bank")
    assert not hasattr(head, "anchor_encoder")
    assert not hasattr(head, "layers")
    assert not hasattr(head, "rosa_refiner")
    assert "deformable" not in " ".join(head.operation_order)
    head.init_weights()
    features = [
        torch.randn(1, 4, 256, size, size, requires_grad=True) for size in (16, 8, 4, 2)
    ]
    metadata = _metadata()

    output = head(features, metadata)

    assert len(output["prediction"]) == len(output["dn_prediction"]) == 6
    assert output["prediction"][0].shape == (1, 300, 10)
    assert output["dn_prediction"][0].shape == (1, 10, 10)
    assert set(output["dn_cls_target"].unique().tolist()) == {1, 6}

    losses = head.loss(output, metadata, features)
    assert len([name for name in losses if "dn" in name]) == 12
    assert all(torch.isfinite(value).all() for value in losses.values())
    sum(losses.values()).backward()

    query_gradient = head.fisheye_ray_decoder.query_bank.feature.grad
    assert query_gradient is not None and torch.isfinite(query_gradient).all()
    assert all(feature.grad is not None for feature in features)
    assert all(torch.isfinite(feature.grad).all() for feature in features)
    trainable = [
        (name, parameter)
        for name, parameter in head.named_parameters()
        if parameter.requires_grad
    ]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in trainable
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert not missing, f"DDP-unsafe unused trainable parameters: {missing}"
    assert not nonfinite, f"non-finite trainable parameter gradients: {nonfinite}"
