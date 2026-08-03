"""Adversarial tests for the sealed C32/fast-CP16 resume contract."""

from __future__ import annotations

import copy
import importlib.util
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from mmengine.config import Config
import pytest
import torch

from rosa import ROSARefiner
from rosa.checkpoint_contract import (
    checkpoint_payload_contract,
    model_state_contract,
    validate_runner_resume_contract,
)
from rosa.mmengine_hooks import (
    CheckpointContractHook,
    ROSASparseMilestoneCheckpointHook,
)

_SPEC = importlib.util.spec_from_file_location(
    "rosa_sealed_run_tool",
    Path(__file__).parents[1] / "tools" / "run.py",
)
assert _SPEC is not None and _SPEC.loader is not None
run_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_tool)


def _source_identity() -> dict:
    revision = {
        "head": "1" * 40,
        "tracked_dirty": False,
        "tracked_diff_sha256": "0" * 64,
        "stable_capture": True,
        "error": None,
    }
    return {
        "rosa": revision,
        "detector_host": revision,
        "fisheye3dod": revision,
    }


@pytest.fixture(autouse=True)
def _stable_source_identity(monkeypatch):
    monkeypatch.setattr(run_tool, "_current_source_identity", _source_identity)


def _config(*, role: str = "candidate", fixed_base: bool = False) -> Config:
    head_type = {
        "candidate": "ROSAOfficialResidualRefinementHead",
        "control": "ROSAOfficialResidualNativeBaselineHead",
    }[role]
    head = {"type": head_type}
    if role == "candidate":
        head["rosa_cfg"] = {"profile": "fast"}
        if fixed_base:
            head["rosa_cfg"]["center_target_mode"] = "fixed_base"
    else:
        head["rosa_cfg"] = None
    return Config(
        {
            "model": {"head": head},
            "train_cfg": {"max_epochs": 30},
            "param_scheduler": [
                {
                    "type": "LinearLR",
                    "start_factor": 1.0 / 3.0,
                    "begin": 0,
                    "end": 500,
                    "by_epoch": False,
                },
                {
                    "type": "CosineAnnealingLR",
                    "begin": 0,
                    "end": 30,
                    "T_max": 30,
                    "by_epoch": True,
                    "eta_min_ratio": 1e-3,
                    "convert_to_iter_based": True,
                },
            ],
            "optim_wrapper": {
                "type": "ROSAPairedClipOptimWrapper",
                "optimizer": {"type": "AdamW", "lr": 2e-4},
            },
            "default_hooks": {
                "checkpoint": {
                    "type": "CheckpointHook",
                    "interval": 1,
                    "max_keep_ckpts": 5,
                    "save_best": ["fisheye_bev/mAP", "fisheye_bev/NDS"],
                    "rule": ["greater", "greater"],
                }
            },
            "custom_hooks": [
                {"type": "SyncBuffersHook", "priority": "NORMAL"},
                {
                    "type": "ROSASparseMilestoneCheckpointHook",
                    "milestones": (8,),
                },
            ],
            "sealed_resume_contract": {
                "contract": "rosa.c32-fast-cp16.30e-pair.v1",
                "role": role,
            },
        }
    )


def _model_state() -> OrderedDict:
    effective = run_tool._effective_rosa_config(_config())
    with torch.random.fork_rng(devices=[]):
        refiner = ROSARefiner(effective).state_dict()
    state = OrderedDict(
        [("img_backbone.weight", torch.zeros(2, 3))]
        + [
            (f"head.rosa_refiner.{key}", torch.zeros_like(value))
            for key, value in refiner.items()
        ]
    )
    state._metadata = OrderedDict(
        {
            "": {"version": 1},
            "img_backbone": {"version": 1},
            "head.rosa_refiner": {"version": 1},
        }
    )
    return state


def _optimizer_state() -> dict:
    return {
        "state": {
            0: {
                "step": torch.tensor(8.0),
                "exp_avg": torch.zeros(2, 3),
                "exp_avg_sq": torch.zeros(2, 3),
            },
            1: {
                "step": torch.tensor(8.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            },
            2: {
                "step": torch.tensor(8.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            },
        },
        "param_groups": [
            {
                "params": [0, 1],
                "lr": 1.2e-4,
                "initial_lr": 2e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.001,
                "rosa_sidecar": False,
            },
            {
                "params": [2],
                "lr": 1.2e-4,
                "initial_lr": 2e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.001,
                "rosa_sidecar": True,
            },
        ],
        "base_param_settings": {
            "params": torch.tensor([0.0]),
            "lr": 1.2e-4,
            "initial_lr": 2e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.001,
        },
    }


def _scheduler_state(*, horizon: int = 3000, progress: int = 800) -> list[dict]:
    return [
        {
            "start_factor": 1.0 / 3.0,
            "end_factor": 1.0,
            "total_iters": 499,
            "param_name": "lr",
            "begin": 0,
            "end": 500,
            "by_epoch": False,
            "base_values": [2e-4, 2e-4],
            "last_step": 499,
            "_global_step": progress,
            "verbose": False,
            "_last_value": [2e-4, 2e-4],
        },
        {
            "T_max": horizon,
            "eta_min": None,
            "eta_min_ratio": 1e-3,
            "param_name": "lr",
            "begin": 0,
            "end": horizon,
            "by_epoch": False,
            "base_values": [2e-4, 2e-4],
            "last_step": progress,
            "_global_step": progress,
            "verbose": False,
            "_last_value": [1.2e-4, 1.2e-4],
        },
    ]


def _checkpoint(*, training_state: bool = True, fixed_base: bool = False) -> dict:
    config = _config(fixed_base=fixed_base)
    state = _model_state()
    metadata = run_tool._checkpoint_contract_metadata(config)
    assert metadata is not None
    metadata["sealed_training_contract"]["model_state_contract"] = model_state_contract(
        state
    )
    checkpoint = {
        "meta": {"epoch": 8, "iter": 800},
        "state_dict": state,
    }
    if training_state:
        checkpoint["optimizer"] = _optimizer_state()
        checkpoint["param_schedulers"] = _scheduler_state()
    CheckpointContractHook(metadata).before_save_checkpoint(
        SimpleNamespace(), checkpoint
    )
    return checkpoint


def _save(path: Path, checkpoint: dict) -> None:
    torch.save(checkpoint, path)


def test_exact_candidate_checkpoint_accepts_full_resume_state(tmp_path: Path):
    path = tmp_path / "epoch_8.pth"
    _save(path, _checkpoint())

    payload = run_tool._validate_checkpoint_contract(
        path,
        _config(),
        allow_mismatch=False,
        require_resume_state=True,
    )

    assert payload is not None
    assert payload["model_state"]["sidecar_key_count"] > 0
    assert payload["optimizer_state"]["group_count"] == 2
    assert payload["scheduler_state"]["scheduler_count"] == 2


def test_fixed_base_checkpoint_resumes_only_with_its_resolved_config(tmp_path: Path):
    fixed_path = tmp_path / "fixed_base_epoch_8.pth"
    _save(fixed_path, _checkpoint(fixed_base=True))
    assert (
        run_tool._validate_checkpoint_contract(
            fixed_path,
            _config(fixed_base=True),
            allow_mismatch=False,
            require_resume_state=True,
        )
        is not None
    )

    with pytest.raises(ValueError, match="config sha256"):
        run_tool._validate_checkpoint_contract(
            fixed_path,
            _config(),
            allow_mismatch=False,
            require_resume_state=True,
        )

    sequential_path = tmp_path / "sequential_epoch_8.pth"
    _save(sequential_path, _checkpoint())
    with pytest.raises(ValueError, match="config sha256"):
        run_tool._validate_checkpoint_contract(
            sequential_path,
            _config(fixed_base=True),
            allow_mismatch=False,
            require_resume_state=True,
        )


def test_wrong_c0_or_control_head_cannot_resume_candidate(tmp_path: Path):
    checkpoint = _checkpoint()
    checkpoint["meta"]["sealed_training_contract"]["head_type"] = "ROSARefinementHead"
    path = tmp_path / "wrong_head.pth"
    _save(path, checkpoint)

    with pytest.raises(ValueError, match="head type differs"):
        run_tool._validate_checkpoint_contract(
            path,
            _config(),
            allow_mismatch=False,
            require_resume_state=True,
        )

    control = _config(role="control")
    with pytest.raises(ValueError, match="config|role|head"):
        run_tool._validate_checkpoint_contract(
            path,
            control,
            allow_mismatch=False,
            require_resume_state=True,
        )


def test_missing_native_state_fails_even_if_payload_digest_is_rewritten(
    tmp_path: Path,
):
    checkpoint = _checkpoint()
    checkpoint["state_dict"].pop("img_backbone.weight")
    checkpoint["meta"]["sealed_payload_contract"] = checkpoint_payload_contract(
        checkpoint
    )
    path = tmp_path / "missing_native.pth"
    _save(path, checkpoint)

    with pytest.raises(ValueError, match="omits or changes native/sidecar"):
        run_tool._validate_checkpoint_contract(
            path,
            _config(),
            allow_mismatch=False,
            require_resume_state=True,
        )


def test_incomplete_optimizer_state_cannot_resume_after_epoch_zero(tmp_path: Path):
    checkpoint = _checkpoint()
    checkpoint["optimizer"]["state"].pop(2)
    checkpoint["meta"]["sealed_payload_contract"] = checkpoint_payload_contract(
        checkpoint
    )
    path = tmp_path / "missing_optimizer_parameter.pth"
    _save(path, checkpoint)

    with pytest.raises(ValueError, match="optimizer state is incomplete"):
        run_tool._validate_checkpoint_contract(
            path,
            _config(),
            allow_mismatch=False,
            require_resume_state=True,
        )


def test_wrong_horizon_or_resolved_config_identity_is_rejected(tmp_path: Path):
    checkpoint = _checkpoint()
    checkpoint["meta"]["sealed_training_contract"]["scheduler_config_sha256"] = "2" * 64
    path = tmp_path / "twenty_epoch_contract.pth"
    _save(path, checkpoint)
    with pytest.raises(ValueError, match="scheduler config"):
        run_tool._validate_checkpoint_contract(
            path,
            _config(),
            allow_mismatch=False,
            require_resume_state=True,
        )

    checkpoint = _checkpoint()
    path = tmp_path / "wrong_config.pth"
    _save(path, checkpoint)
    config = _config()
    config["train_dataloader"] = {"batch_size": 3}
    with pytest.raises(ValueError, match="config sha256"):
        run_tool._validate_checkpoint_contract(
            path,
            config,
            allow_mismatch=False,
            require_resume_state=True,
        )


def test_sealed_pair_requires_per_epoch_source_for_epoch8_milestone():
    config = _config()
    config.default_hooks.checkpoint.interval = 2

    with pytest.raises(ValueError, match="per-epoch checkpoints"):
        run_tool._sealed_training_contract(config)


@pytest.mark.parametrize(
    ("field", "mutation", "message"),
    (
        (
            "runtime_identity",
            lambda value: {**value, "torch_version": "0.0.0"},
            "runtime identity",
        ),
        (
            "source_identity",
            lambda value: {
                **value,
                "rosa": {**value["rosa"], "head": "f" * 40},
            },
            "source identity",
        ),
    ),
)
def test_source_and_runtime_identity_are_exact(
    tmp_path: Path,
    field: str,
    mutation,
    message: str,
):
    checkpoint = _checkpoint()
    contract = checkpoint["meta"]["sealed_training_contract"]
    contract[field] = mutation(contract[field])
    path = tmp_path / f"wrong_{field}.pth"
    _save(path, checkpoint)

    with pytest.raises(ValueError, match=message):
        run_tool._validate_checkpoint_contract(
            path,
            _config(),
            allow_mismatch=False,
            require_resume_state=True,
        )


def test_best_weight_checkpoint_is_evaluable_but_not_resumable(tmp_path: Path):
    path = tmp_path / "best_fisheye_bev_mAP_epoch_12.pth"
    _save(path, _checkpoint(training_state=False))

    assert (
        run_tool._validate_checkpoint_contract(
            path,
            _config(),
            allow_mismatch=False,
        )
        is not None
    )
    with pytest.raises(ValueError, match="missing optimizer state"):
        run_tool._validate_checkpoint_contract(
            path,
            _config(),
            allow_mismatch=False,
            require_resume_state=True,
        )


class _StateModel(torch.nn.Module):
    def __init__(self, state: OrderedDict) -> None:
        super().__init__()
        self._state = state

    def state_dict(self, *args, **kwargs):
        del args, kwargs
        return self._state


class _StateOwner:
    def __init__(self, state) -> None:
        self._state = state

    def state_dict(self):
        return self._state


def test_runtime_preload_guard_rejects_wrong_scheduler_definition():
    checkpoint = _checkpoint()
    expected = checkpoint["meta"]["sealed_payload_contract"]
    fresh_optimizer = copy.deepcopy(_optimizer_state())
    fresh_optimizer["state"] = {}
    for group in fresh_optimizer["param_groups"]:
        group["lr"] = 2e-4 / 3.0
    fresh_optimizer["base_param_settings"]["lr"] = 2e-4 / 3.0
    runner = SimpleNamespace(
        model=_StateModel(_model_state()),
        optim_wrapper=_StateOwner(fresh_optimizer),
        param_schedulers=[
            _StateOwner(state) for state in _scheduler_state(horizon=2000, progress=0)
        ],
    )

    with pytest.raises(ValueError, match="scheduler definition differs"):
        validate_runner_resume_contract(runner, expected)


def test_sparse_epoch8_milestone_is_one_hardlinked_checkpoint(tmp_path: Path):
    source = tmp_path / "epoch_8.pth"
    source.write_bytes(b"sealed epoch eight")
    runner = SimpleNamespace(epoch=7, work_dir=str(tmp_path))
    hook = ROSASparseMilestoneCheckpointHook((8,))

    hook.after_train_epoch(runner)
    target = tmp_path / "milestone_epoch_8.pth"
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino == source.stat().st_ino
    hook.after_train_epoch(runner)
    assert target.stat().st_ino == source.stat().st_ino
