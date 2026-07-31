from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mmengine")
from mmengine.config import Config


_SPEC = importlib.util.spec_from_file_location(
    "rosa_run_tool",
    Path(__file__).parents[1] / "tools" / "run.py",
)
assert _SPEC is not None and _SPEC.loader is not None
run_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_tool)


def _config(profile: str | None) -> Config:
    head = (
        {"type": "Sparse4DHead"}
        if profile is None
        else {
            "type": "ROSASparse4DHead",
            "rosa_cfg": {"profile": profile},
        }
    )
    return Config({"model": {"head": head}})


def _save_checkpoint(
    path: Path,
    *,
    has_rosa: bool,
    metadata: dict | None = None,
) -> None:
    prefix = "head.canonical_sixth_block.rosa" if has_rosa else "head.decoder"
    torch.save(
        {
            "state_dict": {f"{prefix}.weight": torch.ones(1)},
            "meta": metadata or {},
        },
        path,
    )


def test_checkpoint_contract_accepts_matching_release(tmp_path: Path):
    checkpoint = tmp_path / "release.pth"
    _save_checkpoint(
        checkpoint,
        has_rosa=True,
        metadata={
            "architecture": "ROSA R18 Main126",
            "projection_contract": "legacy_front_only",
        },
    )

    run_tool._validate_checkpoint_contract(
        checkpoint,
        _config("published"),
        allow_mismatch=False,
    )


def test_checkpoint_contract_rejects_projection_and_topology(tmp_path: Path):
    checkpoint = tmp_path / "release.pth"
    _save_checkpoint(
        checkpoint,
        has_rosa=True,
        metadata={
            "architecture": "ROSA R18 Main126",
            "projection_contract": "legacy_front_only",
        },
    )

    with pytest.raises(
        ValueError,
        match="projection.*published checkpoint.*compact",
    ):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("balanced"),
            allow_mismatch=False,
        )


def test_checkpoint_contract_rejects_box_coordinate_geometry(tmp_path: Path):
    checkpoint = tmp_path / "wrong_z_convention.pth"
    source_config = run_tool._effective_rosa_config(_config("balanced"))
    source_mapping = {
        **source_config.__dict__,
        "anchor_z_is_bottom": True,
    }
    _save_checkpoint(
        checkpoint,
        has_rosa=True,
        metadata={
            "projection_contract": "full_fov",
            "rosa_config": source_mapping,
        },
    )

    with pytest.raises(ValueError, match="box-coordinate geometry"):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("balanced"),
            allow_mismatch=False,
        )


def test_checkpoint_contract_fails_closed_without_metadata(tmp_path: Path):
    checkpoint = tmp_path / "unknown_rosa.pth"
    _save_checkpoint(checkpoint, has_rosa=True)

    with pytest.raises(ValueError, match="lacks projection_contract"):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("published"),
            allow_mismatch=False,
        )


def test_checkpoint_contract_detects_rosa_presence(tmp_path: Path):
    checkpoint = tmp_path / "baseline.pth"
    _save_checkpoint(checkpoint, has_rosa=False)

    with pytest.raises(ValueError, match="checkpoint ROSA=False"):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("published"),
            allow_mismatch=False,
        )


def test_checkpoint_contract_override_warns(tmp_path: Path):
    checkpoint = tmp_path / "unknown_rosa.pth"
    _save_checkpoint(checkpoint, has_rosa=True)

    with pytest.warns(UserWarning, match="contract mismatch"):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("published"),
            allow_mismatch=True,
        )


def test_checkpoint_hook_records_effective_contract():
    metadata = run_tool._checkpoint_contract_metadata(_config("fast"))
    assert metadata is not None
    checkpoint = {"meta": {"epoch": 2}}
    run_tool._CheckpointContractHook(metadata).before_save_checkpoint(
        SimpleNamespace(),
        checkpoint,
    )

    assert checkpoint["meta"]["epoch"] == 2
    assert checkpoint["meta"]["projection_contract"] == "full_fov"
    assert checkpoint["meta"]["refinement_profile"] == "fast"
    assert checkpoint["meta"]["rosa_config"]["architecture_profile"] == "compact"


class _FakeRefiner:
    def __init__(self) -> None:
        self._compiled_call_impl = None
        self.compile_calls = []

    def compile(self, **kwargs) -> None:
        self.compile_calls.append(kwargs)
        self._compiled_call_impl = object()


def test_fast_compile_supports_alias_and_is_idempotent():
    refiner = _FakeRefiner()
    runner = SimpleNamespace(
        model=SimpleNamespace(
            head=SimpleNamespace(rosa_refiner=refiner),
        )
    )
    config = _config("compact_v2_fast")

    assert run_tool._compile_fast_inference(runner, config)
    assert run_tool._compile_fast_inference(runner, config)
    assert refiner.compile_calls == [
        {"mode": "reduce-overhead", "fullgraph": False}
    ]


def test_non_fast_profile_is_not_compiled():
    runner = SimpleNamespace(model=SimpleNamespace())
    assert not run_tool._compile_fast_inference(runner, _config("balanced"))
