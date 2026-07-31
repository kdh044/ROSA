from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("mmengine")
from mmengine.config import Config
from mmengine.optim import AmpOptimWrapper, DefaultOptimWrapperConstructor, OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS

from rosa.integrations.optim import (
    ROSAPairedAmpOptimWrapper,
    ROSAPairedClipOptimWrapper,
)

ROOT = Path(__file__).resolve().parents[1]
CLIP_GRAD = dict(max_norm=25, norm_type=2, error_if_nonfinite=True)


def _assert_exact(left, right, path: str = "root") -> None:
    if torch.is_tensor(left):
        assert torch.is_tensor(right), path
        assert left.dtype == right.dtype, path
        assert left.device == right.device, path
        assert left.shape == right.shape, path
        assert torch.equal(left, right), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict), path
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_exact(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert type(left) is type(right), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_exact(left_item, right_item, f"{path}[{index}]")
        return
    assert left == right, path


def _make_adamw(params: list[nn.Parameter]) -> torch.optim.AdamW:
    groups = [
        {"params": [params[0]], "lr": 1e-4, "lr_mult": 0.5},
        {"params": [params[1]]},
        {"params": [params[2]]},
    ]
    return torch.optim.AdamW(groups, lr=2e-4, weight_decay=0.001)


def _assign_b0_grads(params: list[nn.Parameter], step: int) -> None:
    grads = [
        torch.tensor([30.0 + step, -40.0, 4.0], dtype=torch.float64),
        torch.tensor([8.0, -15.0 - step], dtype=torch.float64),
        torch.tensor([12.0 + 2 * step], dtype=torch.float64),
    ]
    for param, grad in zip(params, grads):
        param.grad = grad


def test_native_only_steps_and_state_are_bitwise_identical() -> None:
    initial = [
        torch.tensor([0.25, -0.75, 1.25], dtype=torch.float64),
        torch.tensor([-1.5, 0.5], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
    ]
    stock_params = [nn.Parameter(value.clone()) for value in initial]
    paired_params = [nn.Parameter(value.clone()) for value in initial]
    stock = OptimWrapper(_make_adamw(stock_params), clip_grad=CLIP_GRAD.copy())
    paired = ROSAPairedClipOptimWrapper(
        _make_adamw(paired_params),
        clip_grad=CLIP_GRAD.copy(),
    )

    for step in range(5):
        _assign_b0_grads(stock_params, step)
        _assign_b0_grads(paired_params, step)
        stock.step()
        paired.step()
        for stock_param, paired_param in zip(stock_params, paired_params):
            assert torch.equal(stock_param, paired_param)
        _assert_exact(stock.state_dict(), paired.state_dict())

    paired.load_state_dict(copy.deepcopy(stock.state_dict()))
    _assign_b0_grads(stock_params, 5)
    _assign_b0_grads(paired_params, 5)
    stock.step()
    paired.step()
    for stock_param, paired_param in zip(stock_params, paired_params):
        assert torch.equal(stock_param, paired_param)
    _assert_exact(stock.state_dict(), paired.state_dict())


def _grad_norm(param: nn.Parameter) -> float:
    assert param.grad is not None
    return float(torch.linalg.vector_norm(param.grad))


def test_native_and_sidecar_norms_are_clipped_independently() -> None:
    native = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    sidecar = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    native.grad = torch.tensor([24.0, 32.0], dtype=torch.float64)
    sidecar.grad = torch.tensor([18.0, 24.0], dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        [
            {"params": [native]},
            {"params": [sidecar], "rosa_sidecar": True},
        ],
        lr=2e-4,
    )
    wrapper = ROSAPairedClipOptimWrapper(
        optimizer,
        clip_grad=CLIP_GRAD.copy(),
    )

    wrapper._clip_grad()

    assert _grad_norm(native) == pytest.approx(25.0, abs=1e-5)
    assert _grad_norm(sidecar) == pytest.approx(25.0, abs=1e-5)


class _ToyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.native = nn.Linear(2, 2)
        self.rosa_refiner = nn.Linear(2, 2)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.img_backbone = nn.Linear(2, 2)
        self.head = _ToyHead()


def test_constructor_marks_only_refiner_without_redundant_multipliers() -> None:
    model = _ToyModel()
    constructor = DefaultOptimWrapperConstructor(
        dict(
            type="ROSAPairedClipOptimWrapper",
            optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.001),
            clip_grad=CLIP_GRAD.copy(),
        ),
        dict(
            custom_keys={
                "img_backbone": dict(lr_mult=0.5),
                "head.rosa_refiner": dict(rosa_sidecar=True),
            }
        ),
    )

    wrapper = constructor(model)

    marked = [
        group
        for group in wrapper.optimizer.param_groups
        if group.get("rosa_sidecar", False)
    ]
    marked_params = {id(param) for group in marked for param in group["params"]}
    expected_params = {id(param) for param in model.head.rosa_refiner.parameters()}
    assert marked_params == expected_params
    assert all(group["lr"] == 2e-4 for group in marked)
    assert all(group["weight_decay"] == 0.001 for group in marked)
    assert all("lr_mult" not in group for group in marked)
    assert all("decay_mult" not in group for group in marked)


def test_legacy_checkpoint_load_restores_config_owned_marker() -> None:
    legacy_native = nn.Parameter(torch.tensor([1.0, 2.0]))
    legacy_sidecar = nn.Parameter(torch.tensor([3.0, 4.0]))
    legacy = OptimWrapper(
        torch.optim.AdamW(
            [{"params": [legacy_native]}, {"params": [legacy_sidecar]}],
            lr=2e-4,
        ),
        clip_grad=CLIP_GRAD.copy(),
    )
    legacy_native.grad = torch.tensor([4.0, 3.0])
    legacy_sidecar.grad = torch.tensor([12.0, 5.0])
    legacy.step()
    legacy_state = copy.deepcopy(legacy.state_dict())
    assert all("rosa_sidecar" not in group for group in legacy_state["param_groups"])

    native = nn.Parameter(legacy_native.detach().clone())
    sidecar = nn.Parameter(legacy_sidecar.detach().clone())
    paired = ROSAPairedClipOptimWrapper(
        torch.optim.AdamW(
            [
                {"params": [native]},
                {"params": [sidecar], "rosa_sidecar": True},
            ],
            lr=2e-4,
        ),
        clip_grad=CLIP_GRAD.copy(),
    )

    paired.load_state_dict(legacy_state)

    assert "rosa_sidecar" not in paired.optimizer.param_groups[0]
    assert paired.optimizer.param_groups[1]["rosa_sidecar"] is True
    round_trip = copy.deepcopy(paired.state_dict())
    assert round_trip["param_groups"][1]["rosa_sidecar"] is True


def test_registry_amp_and_resolved_config_contracts() -> None:
    assert OPTIM_WRAPPERS.get("ROSAPairedClipOptimWrapper") is (
        ROSAPairedClipOptimWrapper
    )
    assert OPTIM_WRAPPERS.get("ROSAPairedAmpOptimWrapper") is (
        ROSAPairedAmpOptimWrapper
    )
    assert issubclass(ROSAPairedAmpOptimWrapper, AmpOptimWrapper)

    v2 = Config.fromfile(ROOT / "configs" / "rosa_r18_main126_v2.py")
    b0 = Config.fromfile(ROOT / "configs" / "rosa_r18_main126_v2_b0_native.py")
    for config in (v2, b0):
        assert config.optim_wrapper.type == "ROSAPairedClipOptimWrapper"
        assert config.optim_wrapper.clip_grad == CLIP_GRAD
        assert config.optim_wrapper.paramwise_cfg.custom_keys[
            "head.rosa_refiner"
        ] == dict(rosa_sidecar=True)
        assert "rosa.integrations.optim" in config.custom_imports.imports
