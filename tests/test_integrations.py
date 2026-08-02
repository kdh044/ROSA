from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

from rosa import ROSAConfig
from rosa.integrations.sparse4d_runtime import (
    NativeRefineCall,
    SingleCallCapture,
    detached_fpn,
    remap_legacy_refiner_state_dict,
    validate_native_call,
)
from rosa.integrations.sparse4d_scores import (
    compose_quality_rank,
    decode_frozen_jointscore_rank,
    stable_joint_probability,
)


ROOT = Path(__file__).resolve().parents[1]


def _detector_host() -> Path:
    return Path(
        os.environ.get(
            "ROSA_DETECTOR_HOST_ROOT",
            ROOT / "external" / "detector_host",
        )
    ).resolve()


def test_integration_package_does_not_eagerly_import_sparse4d():
    code = textwrap.dedent(
        """
        import sys
        import rosa.integrations

        assert rosa.integrations.__name__ == "rosa.integrations"
        assert "rosa.integrations.sparse4d" not in sys.modules
        """
    )
    environment = os.environ.copy()
    pythonpath = [str(ROOT / "src")]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gravity_center_keypoints_are_symmetric_about_anchor_z():
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent(
        """
        import torch
        from rosa.integrations.sparse4d import ROSANormalizedKeyPointsGenerator

        generator = ROSANormalizedKeyPointsGenerator(
            embed_dims=16,
            num_learnable_pts=0,
            fix_scale=((0.0, 0.0, 0.0),
                       (0.0, 0.0, 0.45),
                       (0.0, 0.0, -0.45)),
            anchor_z_is_bottom=False,
        )
        anchor = torch.zeros(1, 1, 11)
        anchor[..., :3] = torch.tensor((3.0, -2.0, -1.3))
        anchor[..., 3:6] = torch.log(torch.tensor((2.0, 4.0, 1.6)))
        anchor[..., 7] = 1.0
        points = generator(anchor)
        torch.testing.assert_close(points[..., 0, :], anchor[..., :3])
        upper = points[..., 1, 2] - anchor[..., 2]
        lower = points[..., 2, 2] - anchor[..., 2]
        torch.testing.assert_close(upper, -lower)
        """
    )
    environment = os.environ.copy()
    pythonpath = [str(ROOT / "src"), str(host)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_compact_sidecar_preserves_paired_native_initialization_and_rng():
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent(
        """
        import random
        import numpy as np
        import torch
        from mmengine.config import Config
        from mmengine.utils import import_modules_from_strings
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules

        register_all_modules(init_default_scope=True)

        def build(config_name):
            random.seed(3407)
            np.random.seed(3407)
            torch.manual_seed(3407)
            config = Config.fromfile("configs/" + config_name)
            import_modules_from_strings(**config.custom_imports)
            model = MODELS.build(config.model)
            model.init_weights()
            return model, torch.get_rng_state().clone()

        baseline, baseline_rng = build("rosa_r18_main126_v2_b0_native.py")
        rosa, rosa_rng = build("rosa_r18_main126_v2.py")
        baseline_state = baseline.state_dict()
        rosa_state = rosa.state_dict()
        common = set(baseline_state).intersection(rosa_state)
        assert common
        assert all(
            torch.equal(baseline_state[name], rosa_state[name])
            for name in common
        )
        torch.testing.assert_close(baseline_rng, rosa_rng, rtol=0, atol=0)
        """
    )
    environment = os.environ.copy()
    pythonpath = [str(ROOT / "src"), str(host)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_released_config_aliases_keep_legacy_projection_default():
    config = ROSAConfig.from_mapping(
        {
            "radial_proposal_mode": "learned_log_gaussian",
            "second_step_candidate_search": False,
            "enable_shape1": True,
            "enable_shape2_feedback": True,
            "max_shape2_log_size_delta": 0.05,
            "max_shape2_yaw_delta_rad": 0.08,
        }
    )
    assert config.projection_contract == "legacy_front_only"
    assert config.enable_shape_refinement
    assert config.radial_proposal_mode == "learned_log_gaussian"


def test_native_refine_capture_forces_and_validates_scores():
    capture = SingleCallCapture(force_return_cls=True)
    args = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    kwargs = {
        "time_interval": None,
        "return_cls": False,
        "classification_feature": None,
        "regression_feature": None,
        "score_alignment_feature": None,
    }
    changed = capture(None, args, kwargs)
    assert changed is not None
    assert changed[1]["return_cls"] is True
    call = capture.take()
    validate_native_call(call)


def test_native_refine_contract_accepts_upstream_minimal_signature():
    call = NativeRefineCall(
        (torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        {"time_interval": None, "return_cls": True},
    )
    validate_native_call(call)

    extended = NativeRefineCall(
        call.args,
        {
            **call.kwargs,
            "classification_feature": None,
            "regression_feature": None,
            "score_alignment_feature": None,
        },
    )
    validate_native_call(extended)

    with pytest.raises(RuntimeError, match="signature changed"):
        validate_native_call(
            NativeRefineCall(
                call.args,
                {**call.kwargs, "unknown_sidecar": None},
            )
        )
    with pytest.raises(RuntimeError, match="signature changed"):
        validate_native_call(
            NativeRefineCall(
                call.args,
                {**call.kwargs, "classification_feature": None},
            )
        )
    with pytest.raises(RuntimeError, match="sidecars must be None"):
        validate_native_call(
            NativeRefineCall(
                extended.args,
                {**extended.kwargs, "classification_feature": torch.zeros(1)},
            )
        )


def test_fpn_detach_accepts_framework_neutral_level_tensors():
    source = [
        torch.randn(2, 4, 8, 5, 7, requires_grad=True),
        torch.randn(2, 4, 8, 3, 4, requires_grad=True),
    ]
    restored = detached_fpn(source)
    assert [value.shape for value in restored] == [value.shape for value in source]
    assert all(not value.requires_grad and value.grad_fn is None for value in restored)


def test_packed_fpn_restore_is_zero_copy_and_sampler_equivalent():
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent(
        """
        import torch

        from projects.mmdet3d_plugin.ops import feature_maps_format
        from rosa.integrations.sparse4d_runtime import detached_fpn, restore_fpn
        from rosa.sampling import (
            DeterministicBilinearSampler,
            FusedBilinearSampler,
        )

        torch.manual_seed(19)
        batch, cameras, channels = 2, 4, 7
        levels = [
            torch.randn(batch, cameras, channels, height, width)
            for height, width in ((9, 11), (5, 7), (3, 4), (2, 2))
        ]
        packed = feature_maps_format(levels)
        restored = restore_fpn(packed)
        detached = detached_fpn(packed)
        packed_pointer = packed[0].untyped_storage().data_ptr()

        assert len(restored) == len(levels)
        for source, view, detached_view in zip(levels, restored, detached):
            assert torch.equal(view, source)
            assert view.untyped_storage().data_ptr() == packed_pointer
            assert detached_view.untyped_storage().data_ptr() == packed_pointer
            assert not detached_view.requires_grad
            assert detached_view.grad_fn is None
            flattened = view.reshape(
                batch * cameras,
                channels,
                *view.shape[-2:],
            )
            assert flattened.untyped_storage().data_ptr() == packed_pointer

        uv = torch.rand(batch, 6, cameras, 1, 5, 2) * 1.4 - 0.2
        uv[0, 0, 0, 0, 0] = torch.tensor((float("nan"), float("inf")))
        for sampler in (
            FusedBilinearSampler(),
            DeterministicBilinearSampler(),
        ):
            for view in restored:
                contiguous = view.contiguous()
                expected = sampler(contiguous, uv)
                actual = sampler(view, uv)
                assert torch.equal(actual, expected)

        # The production boundary detaches packed FPN features.  Also prove
        # that the deterministic fallback preserves the semantic feature
        # gradient if a future caller intentionally keeps that boundary live.
        sampler = DeterministicBilinearSampler()
        for level_index, restored_level in enumerate(restored):
            contiguous = restored_level.detach().clone().requires_grad_(True)
            packed_values = packed[0].detach().clone().requires_grad_(True)
            packed_level = restore_fpn(
                (packed_values, packed[1], packed[2])
            )[level_index]
            weight = torch.randn(
                batch,
                6,
                cameras,
                1,
                5,
                channels,
            )
            (sampler(contiguous, uv) * weight).sum().backward()
            (sampler(packed_level, uv) * weight).sum().backward()
            packed_gradient = restore_fpn(
                (packed_values.grad, packed[1], packed[2])
            )[level_index]
            assert torch.equal(packed_gradient, contiguous.grad)
        """
    )
    environment = os.environ.copy()
    pythonpath = [str(ROOT / "src"), str(host)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_checkpoint_prefix_remaps_without_overwriting_modern_key():
    old_weight = torch.tensor([1.0])
    modern_weight = torch.tensor([2.0])
    state = {
        "head.canonical_sixth_block.a": old_weight,
        "head.canonical_sixth_block.b": old_weight,
        "head.rosa_refiner.b": modern_weight,
    }
    consumed = remap_legacy_refiner_state_dict(state, prefix="head.")
    assert consumed == 2
    assert not any("canonical_sixth_block" in key for key in state)
    assert state["head.rosa_refiner.a"] is old_weight
    assert state["head.rosa_refiner.b"] is modern_weight


def test_zero_quality_delta_is_exact_native_joint_score():
    classification = torch.tensor([[[0.5, -0.25], [1.0, 0.1]]])
    quality = torch.tensor([[[0.2], [-0.3]]])
    delta = torch.zeros(1, 2, 1, requires_grad=True)
    supported = torch.ones(1, 2, dtype=torch.bool)
    rank = compose_quality_rank(
        classification,
        quality,
        delta,
        supported,
        centerness_index=0,
    )
    assert torch.equal(
        rank["emitted_joint_score"],
        rank["native_joint_score"],
    )
    stable_joint_probability(
        rank["joint_logit"],
        rank["native_joint_score"],
        rank["effective_delta"],
    ).sum().backward()
    assert delta.grad is not None


class _IdentityBoxDecoder:
    selection_score_mode = "legacy"
    sorted = True
    num_output = 3
    score_threshold = None

    @staticmethod
    def decode_box(boxes):
        return boxes


def test_quality_residual_changes_only_query_ranking_at_inference():
    classification = torch.tensor(
        [[[2.0, -1.0], [-1.0, 1.5], [1.0, -2.0]]]
    )
    native_quality = torch.zeros(1, 3, 1)
    boxes = torch.arange(33, dtype=torch.float32).reshape(1, 3, 11)
    instance_id = torch.tensor(((10, 20, 30),))
    supported = torch.ones(1, 3, dtype=torch.bool)
    classification_before = classification.clone()
    quality_before = native_quality.clone()

    native = compose_quality_rank(
        classification,
        native_quality,
        torch.zeros(1, 3, 1),
        supported,
        centerness_index=0,
    )
    reranked = compose_quality_rank(
        classification,
        native_quality,
        torch.tensor([[[0.0], [1.0], [0.0]]]),
        supported,
        centerness_index=0,
    )
    decoder = _IdentityBoxDecoder()
    native_result = decode_frozen_jointscore_rank(
        decoder,
        [classification],
        [boxes],
        native["frozen_label"],
        native["native_selected_probability"],
        native["emitted_joint_score"],
        instance_id=instance_id,
        output_idx=0,
    )[0]
    reranked_result = decode_frozen_jointscore_rank(
        decoder,
        [classification],
        [boxes],
        reranked["frozen_label"],
        reranked["native_selected_probability"],
        reranked["emitted_joint_score"],
        instance_id=instance_id,
        output_idx=0,
    )[0]

    assert torch.equal(classification, classification_before)
    assert torch.equal(native_quality, quality_before)
    assert not torch.equal(
        native_result["query_indices"],
        reranked_result["query_indices"],
    )
    for query_index in range(3):
        native_position = (
            native_result["query_indices"] == query_index
        ).nonzero(as_tuple=True)[0]
        reranked_position = (
            reranked_result["query_indices"] == query_index
        ).nonzero(as_tuple=True)[0]
        assert torch.equal(
            native_result["boxes_3d"][native_position],
            reranked_result["boxes_3d"][reranked_position],
        )
        assert torch.equal(
            native_result["labels_3d"][native_position],
            reranked_result["labels_3d"][reranked_position],
        )
        assert torch.equal(
            native_result["cls_scores"][native_position],
            reranked_result["cls_scores"][reranked_position],
        )
        assert torch.equal(
            native_result["instance_ids"][native_position],
            reranked_result["instance_ids"][reranked_position],
        )
