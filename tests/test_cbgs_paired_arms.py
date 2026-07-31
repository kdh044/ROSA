from __future__ import annotations

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
CONFIG_DIR = ROOT / "configs"


def _config(name: str) -> Config:
    return Config.fromfile(CONFIG_DIR / name)


def test_c1_restores_only_the_balanced_rosa_head_without_merge_residue() -> None:
    c0 = _config("rosa_r18_main126_v2_c0_cbgs_native.py")
    c1 = _config("rosa_r18_main126_v2_c1_cbgs_balanced.py")

    assert c0.model.head.type == "ROSANativeBaselineHead"
    assert c0.model.head.rosa_cfg is None
    assert c1.model.head.type == "ROSARefinementHead"
    assert c1.model.head.rosa_cfg == dict(profile="balanced")

    for key in (
        "instance_bank",
        "deformable_model",
        "loss_reg",
        "operation_order",
        "num_decoder",
    ):
        assert c1.model.head[key] == c0.model.head[key]

    assert c1.train_dataloader == c0.train_dataloader
    assert c1.train_cfg == c0.train_cfg
    assert c1.param_scheduler == c0.param_scheduler
    assert c1.optim_wrapper == c0.optim_wrapper
    assert c1.randomness == c0.randomness
    assert c1.experiment_contract.cbgs_train_samples == 25_457
    assert c1.experiment_contract.global_batch_size == 8
    assert c1.experiment_contract.max_epochs == 20
    assert c1.experiment_contract.cbgs_optimizer_steps_20e == 63_660
    assert c1.experiment_contract.rosa_sidecar is True
    assert c1.experiment_contract.learned_refiner_parameters == 120_559


def test_c2_reuses_q1_values_without_changing_the_c1_train_contract() -> None:
    c1 = _config("rosa_r18_main126_v2_c1_cbgs_balanced.py")
    c2 = _config("rosa_r18_main126_v2_c2_cbgs_q1_quality.py")
    q1 = _config("rosa_r18_main126_v2_q1_quality_residual.py")

    assert c2.model.head.type == "ROSARefinementHead"
    assert c2.model.head.rosa_cfg.profile == "balanced"
    assert c2.model.head.rosa_cfg.enable_quality_score is True
    assert c2.model.head.rosa_cfg.quality_loss_weight == 0.5
    assert c2.model.head.rosa_cfg.enable_quality_score == (
        q1.model.head.rosa_cfg.enable_quality_score
    )
    assert c2.model.head.rosa_cfg.quality_loss_weight == (
        q1.model.head.rosa_cfg.quality_loss_weight
    )
    assert c2.train_dataloader == c1.train_dataloader
    assert c2.train_cfg == c1.train_cfg
    assert c2.param_scheduler == c1.param_scheduler
    assert c2.optim_wrapper == c1.optim_wrapper
    assert c2.randomness == c1.randomness
    assert c2.experiment_contract.learned_refiner_parameters == 122_117


def test_c1_actual_optimizer_parameters_and_paired_initialization() -> None:
    host = ROOT / "external" / "detector_host"
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent("""
        import random
        import numpy as np
        import torch
        from mmengine.config import Config
        from mmengine.optim import build_optim_wrapper
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
            rng = torch.get_rng_state().clone()
            wrapper = build_optim_wrapper(model, config.optim_wrapper)
            return model, wrapper, rng

        baseline, baseline_wrapper, baseline_rng = build(
            "rosa_r18_main126_v2_c0_cbgs_native.py"
        )
        rosa, rosa_wrapper, rosa_rng = build(
            "rosa_r18_main126_v2_c1_cbgs_balanced.py"
        )

        baseline_state = baseline.state_dict()
        rosa_state = rosa.state_dict()
        assert set(baseline_state).issubset(rosa_state)
        assert all(
            torch.equal(value, rosa_state[name])
            for name, value in baseline_state.items()
        )
        torch.testing.assert_close(baseline_rng, rosa_rng, rtol=0, atol=0)

        refiner_parameters = list(rosa.head.rosa_refiner.parameters())
        assert sum(parameter.numel() for parameter in refiner_parameters) == 120_559
        refiner_ids = {id(parameter) for parameter in refiner_parameters}
        marked_groups = [
            group
            for group in rosa_wrapper.optimizer.param_groups
            if group.get("rosa_sidecar", False)
        ]
        marked_ids = {
            id(parameter)
            for group in marked_groups
            for parameter in group["params"]
        }
        assert marked_ids == refiner_ids
        assert sum(parameter.numel() for group in marked_groups
                   for parameter in group["params"]) == 120_559
        assert not any(
            group.get("rosa_sidecar", False)
            for group in baseline_wrapper.optimizer.param_groups
        )

        def native_group_options(model, wrapper):
            names = {id(parameter): name for name, parameter in model.named_parameters()}
            result = {}
            for group in wrapper.optimizer.param_groups:
                if group.get("rosa_sidecar", False):
                    continue
                assert len(group["params"]) == 1
                name = names[id(group["params"][0])]
                result[name] = {
                    key: value
                    for key, value in group.items()
                    if key not in {"params", "rosa_sidecar"}
                }
            return result

        assert native_group_options(baseline, baseline_wrapper) == (
            native_group_options(rosa, rosa_wrapper)
        )
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
