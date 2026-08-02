"""Contracts for pairing fast CP16 ROSA with the corrected C32 detector."""

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


def test_fast_cp16_has_a_schedule_matched_c32_control() -> None:
    historical = _config(
        "rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py"
    )
    c32 = _config("rosa_r18_main126_v3_c32_official_residual_30e.py")
    combined = _config("rosa_r18_main126_v3_fast_cp16_official_residual.py")

    # The control preserves C32 and changes only its horizon/retention.
    historical_dict = copy.deepcopy(historical.to_dict())
    control_dict = copy.deepcopy(c32.to_dict())
    for value in (historical_dict, control_dict):
        value.pop("work_dir")
        value.pop("experiment_contract")
    historical_dict["param_scheduler"] = control_dict["param_scheduler"]
    historical_dict["train_cfg"] = control_dict["train_cfg"]
    historical_dict["default_hooks"]["checkpoint"]["max_keep_ckpts"] = 5
    assert historical_dict == control_dict

    assert c32.model.head.type == "ROSAOfficialResidualNativeBaselineHead"
    assert combined.model.head.type == "ROSAOfficialResidualRefinementHead"
    assert combined.model.head.rosa_cfg == {"profile": "fast"}

    expected_head = copy.deepcopy(c32.model.head.to_dict())
    observed_head = copy.deepcopy(combined.model.head.to_dict())
    assert expected_head.pop("rosa_cfg") is None
    observed_head.pop("rosa_cfg")
    observed_head["type"] = expected_head["type"]
    assert observed_head == expected_head

    # Freeze the complete scientific pair, not only a hand-picked list of
    # optimizer fields.  Any future dataset, augmentation, batch, schedule,
    # evaluator, or detector drift must fail this test.
    control_pair = copy.deepcopy(c32.to_dict())
    candidate_pair = copy.deepcopy(combined.to_dict())
    for value in (control_pair, candidate_pair):
        value.pop("work_dir")
        value.pop("experiment_contract")
    candidate_pair["model"]["head"]["rosa_cfg"] = None
    candidate_pair["model"]["head"]["type"] = control_pair["model"]["head"][
        "type"
    ]
    assert candidate_pair == control_pair

    for field in ("optim_wrapper", "param_scheduler", "train_cfg"):
        assert copy.deepcopy(combined[field]) == copy.deepcopy(c32[field])
    assert combined.default_hooks.checkpoint == c32.default_hooks.checkpoint
    assert c32.train_cfg.max_epochs == combined.train_cfg.max_epochs == 30
    assert c32.param_scheduler[1].end == combined.param_scheduler[1].end == 30
    assert c32.param_scheduler[1].T_max == combined.param_scheduler[1].T_max == 30
    assert c32.default_hooks.checkpoint.max_keep_ckpts == 5
    assert combined.experiment_contract.backbone == "ImageNet-pretrained ResNet-18"
    assert combined.experiment_contract.learned_refiner_parameters == 87_935
    assert combined.experiment_contract.official_residual_parameters_added == 0
    assert combined.experiment_contract.official_residual_numeric_hyperparameters_added == 0
    assert combined.experiment_contract.sidecar_profile_selection == (
        "fast (fixed CP16 topology)"
    )


def test_combined_head_preserves_c32_native_state_and_residual_math() -> None:
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent("""
        import random
        import numpy as np
        import torch
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
            return model

        c32 = build("rosa_r18_main126_v2_c32_cbgs_native_official_gnn_residual.py")
        combined = build("rosa_r18_main126_v3_fast_cp16_official_residual.py")

        c32_state = c32.state_dict()
        combined_state = combined.state_dict()
        assert set(c32_state).issubset(combined_state)
        assert all(
            torch.equal(value, combined_state[name])
            for name, value in c32_state.items()
        )
        sidecar = {
            name: value
            for name, value in combined_state.items()
            if name not in c32_state
        }
        assert sidecar
        assert all(name.startswith("head.rosa_refiner.") for name in sidecar)
        assert sum(parameter.numel() for parameter in combined.parameters()) - sum(
            parameter.numel() for parameter in c32.parameters()
        ) == 87_935
        assert combined.head.rosa_cfg.refine_topk == 50
        assert combined.head.rosa_cfg.num_refinement_steps == 1
        assert combined.head.rosa_refiner.rosa.rfl.factorized is True

        gnn_indices = [
            index
            for index, operation in enumerate(combined.head.operation_order)
            if operation == "gnn"
        ]
        assert len(gnn_indices) == 5
        index = gnn_indices[0]
        generator = torch.Generator().manual_seed(1701)
        query = torch.randn(2, 11, 256, generator=generator)
        query_pos = torch.randn(2, 11, 256, generator=generator)

        c32.eval()
        combined.eval()
        with torch.inference_mode():
            import rosa.integrations.sparse4d as rosa_sparse4d

            owned_decoupled_query = torch.cat((query, query_pos), dim=-1)
            assert rosa_sparse4d._can_use_inplace_official_residual(
                combined.head,
                combined.head.layers[index],
                owned_decoupled_query,
                torch.zeros_like(owned_decoupled_query),
            ) is True
            expected = c32.head.graph_model(
                index,
                query,
                value=query,
                query_pos=query_pos,
                attn_mask=None,
            )
            actual = combined.head.graph_model(
                index,
                query,
                value=query,
                query_pos=query_pos,
                attn_mask=None,
            )
        assert torch.equal(actual, expected)

        c32.train()
        combined.train()
        dropout_rng = torch.Generator().manual_seed(991).get_state()
        torch.set_rng_state(dropout_rng)
        expected = c32.head.graph_model(
            index,
            query,
            value=query,
            query_pos=query_pos,
            attn_mask=None,
        )
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(dropout_rng)
        actual = combined.head.graph_model(
            index,
            query,
            value=query,
            query_pos=query_pos,
            attn_mask=None,
        )
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), expected_rng)
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
