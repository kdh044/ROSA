"""Host-level contract test for the V3 fisheye sparse-box head."""

import copy
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


def test_real_v3_head_preserves_c0_bank_dn_and_backward_contract():
    root = Path(__file__).resolve().parents[1]
    config = Config.fromfile(
        root / "configs" / "fisheye_box_decoder_v3_raw_uniform_20e.py"
    )
    assert config.model.head.type == "FisheyeSparseBoxHead"
    assert config.model.head.instance_bank.num_anchor == 300
    assert config.model.head.instance_bank.num_temp_instances == 0
    assert config.model.head.sampler.num_dn_groups == 5
    assert config.model.head.sampler.max_dn_gt == 32
    assert config.model.head.sampler.add_neg_dn
    assert config.model.head.box_decoder_cfg.num_learnable_points == 6
    assert config.model.head.box_decoder_cfg.num_layers == 6
    assert not config.model.head.box_decoder_cfg.checkpoint_point_chunks
    assert config.model.head.box_decoder_cfg.use_fused_daf
    assert "deformable_model" not in config.model.head
    assert "rosa_cfg" not in config.model.head

    import_modules_from_strings(**config.custom_imports)
    # The repository-wide contract runs on CPU, where the CUDA DAF extension
    # is unavailable.  Exercise the exact same head through its deterministic
    # reference sampler; real-batch GPU tests cover the fused production path.
    head_config = copy.deepcopy(config.model.head)
    head_config.box_decoder_cfg.use_fused_daf = False
    head = MODELS.build(head_config).train()
    head.init_weights()
    assert head.instance_bank.num_anchor == 300
    assert head.instance_bank.num_temp_instances == 0
    assert not hasattr(head, "rosa_refiner")

    features = [
        torch.randn(1, 4, 256, size, size, requires_grad=True)
        for size in (16, 8, 4, 2)
    ]
    metadata = _metadata()
    output = head(features, metadata)

    assert len(output["prediction"]) == len(output["dn_prediction"]) == 6
    assert output["prediction"][0].shape == (1, 300, 10)
    # One GT, five DN groups, and paired negative DN anchors.
    assert output["dn_prediction"][0].shape == (1, 10, 10)
    assert set(output["dn_cls_target"].unique().tolist()) == {1, 6}
    assert all(value.shape[2] == 13 for value in output["fisheye_point_attention"])

    losses = head.loss(output, metadata, features)
    assert all(torch.isfinite(value).all() for value in losses.values())
    sum(losses.values()).backward()

    assert all(feature.grad is not None for feature in features)
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
