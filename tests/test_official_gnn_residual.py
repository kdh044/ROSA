"""Contracts for restoring Sparse4D's released MHA wrapper residual."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("mmengine")
pytest.importorskip("mmdet3d")
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]


def _detector_host() -> Path:
    return Path(
        os.environ.get(
            "ROSA_DETECTOR_HOST_ROOT",
            ROOT / "external" / "detector_host",
        )
    ).resolve()


def _config(name: str) -> Config:
    return Config.fromfile(ROOT / "configs" / name)


def test_c32_changes_only_the_registered_native_head_behavior() -> None:
    baseline = _config("rosa_r18_main126_v2_c0_cbgs_native.py")
    candidate = _config("rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py")

    assert baseline.model.head.type == "ROSANativeBaselineHead"
    assert candidate.model.head.type == "ROSAOfficialResidualNativeBaselineHead"
    reference = copy.deepcopy(baseline.to_dict())
    observed = copy.deepcopy(candidate.to_dict())
    for config in (reference, observed):
        config.pop("work_dir")
        config.pop("experiment_contract")
    observed["model"]["head"]["type"] = reference["model"]["head"]["type"]
    assert observed == reference

    operation_order = tuple(candidate.model.head.operation_order)
    gnn_indices = [
        index for index, operation in enumerate(operation_order) if operation == "gnn"
    ]
    assert len(gnn_indices) == 5
    assert "temp_gnn" not in operation_order
    assert all(operation_order[index + 1] == "norm" for index in gnn_indices)
    assert candidate.model.head.decouple_attn is True
    assert candidate.model.head.graph_model.dropout == 0.1
    assert candidate.experiment_contract.learned_parameters_added == 0
    assert candidate.experiment_contract.buffers_added == 0
    assert candidate.experiment_contract.user_numeric_hyperparameters_added == 0


def test_c32_matches_the_released_mmcv_wrapper_and_preserves_state() -> None:
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent("""
        import gc
        import random
        import numpy as np
        import torch
        from mmcv.cnn.bricks.transformer import (
            MultiheadAttention as MMCVMultiheadAttention,
        )
        from mmengine.config import Config
        from mmengine.utils import import_modules_from_strings
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules

        register_all_modules(init_default_scope=True)

        def build(name):
            random.seed(3407)
            np.random.seed(3407)
            torch.manual_seed(3407)
            config = Config.fromfile("configs/" + name)
            import_modules_from_strings(**config.custom_imports)
            model = MODELS.build(config.model)
            model.init_weights()
            return model, torch.get_rng_state().clone()

        baseline, baseline_init_rng = build(
            "rosa_r18_main126_v2_c0_cbgs_native.py"
        )
        candidate, candidate_init_rng = build(
            "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py"
        )
        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        assert baseline_state.keys() == candidate_state.keys()
        assert all(
            torch.equal(value, candidate_state[name])
            for name, value in baseline_state.items()
        )
        assert sum(value.numel() for value in baseline.parameters()) == sum(
            value.numel() for value in candidate.parameters()
        )
        assert sum(value.numel() for value in baseline.buffers()) == sum(
            value.numel() for value in candidate.buffers()
        )
        torch.testing.assert_close(
            baseline_init_rng,
            candidate_init_rng,
            rtol=0,
            atol=0,
        )
        candidate.load_state_dict(baseline_state, strict=True)

        gnn_indices = [
            index
            for index, operation in enumerate(candidate.head.operation_order)
            if operation == "gnn"
        ]
        assert len(gnn_indices) == 5
        index = gnn_indices[0]
        layer = candidate.head.layers[index]
        assert isinstance(layer, torch.nn.MultiheadAttention)
        assert layer.batch_first is True
        assert layer.dropout == 0.1
        assert isinstance(candidate.head.fc_before, torch.nn.Linear)
        assert isinstance(candidate.head.fc_after, torch.nn.Linear)
        assert candidate.head.fc_before.bias is None
        assert candidate.head.fc_after.bias is None

        released = MMCVMultiheadAttention(
            embed_dims=layer.embed_dim,
            num_heads=layer.num_heads,
            batch_first=True,
            dropout=layer.dropout,
        )
        released.attn.load_state_dict(layer.state_dict(), strict=True)

        generator = torch.Generator().manual_seed(1701)
        query_feature = torch.randn(2, 11, 256, generator=generator)
        query_position = torch.randn(2, 11, 256, generator=generator)
        decoupled_query = torch.cat(
            (query_feature, query_position),
            dim=-1,
        )
        decoupled_value = candidate.head.fc_before(query_feature)

        candidate.eval()
        released.eval()
        eval_rng_before = torch.get_rng_state().clone()
        actual_eval = candidate.head.graph_model(
            index,
            query_feature,
            value=query_feature,
            query_pos=query_position,
            attn_mask=None,
        )
        eval_rng_after = torch.get_rng_state().clone()
        expected_eval = candidate.head.fc_after(
            released(
                decoupled_query,
                value=decoupled_value,
                attn_mask=None,
            )
        )
        torch.testing.assert_close(actual_eval, expected_eval, rtol=0, atol=0)
        torch.testing.assert_close(
            eval_rng_before,
            eval_rng_after,
            rtol=0,
            atol=0,
        )
        assert actual_eval.shape == query_feature.shape

        baseline.eval()
        current_port = baseline.head.graph_model(
            index,
            query_feature,
            value=query_feature,
            query_pos=query_position,
            attn_mask=None,
        )
        branch_only, _ = released.attn(
            decoupled_query.transpose(0, 1),
            decoupled_query.transpose(0, 1),
            decoupled_value.transpose(0, 1),
            attn_mask=None,
        )
        branch_only = branch_only.transpose(0, 1)
        expected_current_port = baseline.head.fc_after(branch_only)
        torch.testing.assert_close(
            current_port,
            expected_current_port,
            rtol=0,
            atol=0,
        )
        assert not torch.equal(current_port, expected_eval)
        assert float((current_port - expected_eval).abs().max()) > 0.0

        candidate.train()
        released.train()
        train_rng = torch.Generator().manual_seed(991).get_state()
        torch.set_rng_state(train_rng)
        expected_train = candidate.head.fc_after(
            released(
                decoupled_query,
                value=decoupled_value,
                attn_mask=None,
            )
        )
        expected_train_rng = torch.get_rng_state().clone()
        torch.set_rng_state(train_rng)
        actual_train = candidate.head.graph_model(
            index,
            query_feature,
            value=query_feature,
            query_pos=query_position,
            attn_mask=None,
        )
        actual_train_rng = torch.get_rng_state().clone()
        torch.testing.assert_close(actual_train, expected_train, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_train_rng,
            expected_train_rng,
            rtol=0,
            atol=0,
        )
        assert actual_train.shape == query_feature.shape

        del baseline, candidate, baseline_state, candidate_state
        gc.collect()
        """)
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
