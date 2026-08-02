from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
from collections import OrderedDict
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("mmengine")
from mmengine.config import Config
from mmengine.logging.history_buffer import HistoryBuffer

from rosa import ROSARefiner

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


def _native_deploy_config(
    *,
    head_type: str = "ROSAOfficialResidualNativeBaselineHead",
    deformable_type: str = (
        "ROSAInferenceReuseExactHomogeneous4x4FullFOV" "DeformableFeatureAggregation"
    ),
) -> Config:
    return Config(
        {
            "model": {
                "head": {
                    "type": head_type,
                    "deformable_model": {
                        "type": deformable_type,
                        "projection_type": "ocam_fisheye",
                        "kps_generator": {
                            "type": "ROSANormalizedKeyPointsGenerator",
                            "anchor_z_is_bottom": False,
                        },
                    },
                }
            }
        }
    )


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


def _factorized_refiner_state(
    *,
    profile: str = "balanced",
    namespace: str = "modern",
    module_prefix: bool = False,
) -> OrderedDict:
    effective = run_tool._effective_rosa_config(_config(profile))
    with torch.random.fork_rng(devices=[]):
        refiner_state = ROSARefiner(effective).state_dict()
    prefix = {
        "modern": "head.rosa_refiner.",
        "legacy": "head.canonical_sixth_block.",
    }[namespace]
    if module_prefix:
        prefix = "module." + prefix
    return OrderedDict(
        (prefix + key, torch.zeros_like(value)) for key, value in refiner_state.items()
    )


def _save_factorized_checkpoint(
    path: Path,
    *,
    profile: str = "balanced",
    metadata: dict | None = None,
    state_dict: OrderedDict | None = None,
) -> None:
    torch.save(
        {
            "state_dict": (
                _factorized_refiner_state(profile=profile)
                if state_dict is None
                else state_dict
            ),
            "meta": (
                run_tool._checkpoint_contract_metadata(_config(profile))
                if metadata is None
                else metadata
            ),
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


def test_checkpoint_contract_accepts_matching_factorized_compact(tmp_path: Path):
    checkpoint = tmp_path / "factorized_v3.pth"
    rng_before = torch.random.get_rng_state().clone()
    _save_factorized_checkpoint(checkpoint)

    run_tool._validate_checkpoint_contract(
        checkpoint,
        _config("balanced"),
        allow_mismatch=False,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    ("namespace", "module_prefix"),
    (("legacy", False), ("modern", True)),
)
def test_checkpoint_contract_accepts_loader_compatible_refiner_prefixes(
    tmp_path: Path,
    namespace: str,
    module_prefix: bool,
):
    checkpoint = tmp_path / f"factorized_{namespace}_{module_prefix}.pth"
    _save_factorized_checkpoint(
        checkpoint,
        state_dict=_factorized_refiner_state(
            namespace=namespace,
            module_prefix=module_prefix,
        ),
    )

    run_tool._validate_checkpoint_contract(
        checkpoint,
        _config("balanced"),
        allow_mismatch=False,
    )


def test_checkpoint_contract_rejects_dense_compact_v2(tmp_path: Path):
    checkpoint = tmp_path / "dense_compact_v2.pth"
    metadata = run_tool._checkpoint_contract_metadata(_config("balanced"))
    assert metadata is not None
    metadata["rosa_contract_version"] = 2
    metadata.pop("rfl_topology_contract")
    _save_factorized_checkpoint(checkpoint, metadata=metadata)

    with pytest.raises(
        ValueError,
        match="factorized compact RFL requires checkpoint contract v3",
    ):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("balanced"),
            allow_mismatch=False,
        )


def test_checkpoint_contract_rejects_wrong_v3_topology_even_with_override(
    tmp_path: Path,
):
    checkpoint = tmp_path / "wrong_v3_topology.pth"
    metadata = run_tool._checkpoint_contract_metadata(_config("balanced"))
    assert metadata is not None
    metadata["rfl_topology_contract"] = "rosa.rfl.dense_outer_product.v2"
    _save_factorized_checkpoint(checkpoint, metadata=metadata)

    with pytest.raises(ValueError, match="topology integrity.*RFL topology"):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("balanced"),
            allow_mismatch=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_rfl", "missing keys.*ray_proj.weight"),
        ("wrong_rfl_shape", "wrong shapes.*output_proj.weight"),
        ("partial_refiner", "missing keys.*candidate_head.1.bias"),
        ("extra_refiner", "unexpected keys.*unowned.weight"),
        ("external_extra", "unrecognized ROSA state keys.*rosa_unowned.weight"),
    ),
)
def test_checkpoint_contract_rejects_invalid_factorized_refiner_schema(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    checkpoint = tmp_path / f"{mutation}.pth"
    state = _factorized_refiner_state()
    if mutation == "missing_rfl":
        state.pop("head.rosa_refiner.rosa.rfl.ray_proj.weight")
    elif mutation == "wrong_rfl_shape":
        state["head.rosa_refiner.rosa.rfl.output_proj.weight"] = torch.zeros(256, 144)
    elif mutation == "partial_refiner":
        state.pop("head.rosa_refiner.candidate_head.1.bias")
    elif mutation == "extra_refiner":
        state["head.rosa_refiner.unowned.weight"] = torch.ones(1)
    else:
        state["head.rosa_unowned.weight"] = torch.ones(1)
    _save_factorized_checkpoint(checkpoint, state_dict=state)

    with pytest.raises(ValueError, match=message):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("balanced"),
            allow_mismatch=True,
        )


def test_checkpoint_contract_rejects_mixed_legacy_and_modern_refiner_state(
    tmp_path: Path,
):
    checkpoint = tmp_path / "mixed_prefixes.pth"
    state = _factorized_refiner_state()
    state["head.canonical_sixth_block.level_head.weight"] = state[
        "head.rosa_refiner.level_head.weight"
    ].clone()
    _save_factorized_checkpoint(checkpoint, state_dict=state)

    with pytest.raises(ValueError, match="mixes modern and legacy"):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("balanced"),
            allow_mismatch=False,
        )


def test_checkpoint_contract_reads_mmengine_history_without_unsafe_pickle(
    tmp_path: Path,
):
    checkpoint = tmp_path / "mmengine_checkpoint.pth"
    torch.save(
        {
            "state_dict": {"head.canonical_sixth_block.rosa.weight": torch.ones(1)},
            "meta": {
                "architecture": "ROSA R18 Main126",
                "projection_contract": "legacy_front_only",
            },
            "message_hub": {
                "loss": HistoryBuffer(
                    log_history=[np.float64(1.25)],
                    count_history=[np.int64(1)],
                )
            },
        },
        checkpoint,
    )

    run_tool._validate_checkpoint_contract(
        checkpoint,
        _config("published"),
        allow_mismatch=False,
    )


def test_checkpoint_contract_rejects_projection_and_topology(tmp_path: Path):
    checkpoint = tmp_path / "release.pth"
    _save_factorized_checkpoint(
        checkpoint,
        metadata={
            "architecture": "ROSA R18 Main126",
            "projection_contract": "legacy_front_only",
        },
    )

    with pytest.raises(ValueError) as error:
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _config("balanced"),
            allow_mismatch=False,
        )
    message = str(error.value)
    assert "published checkpoint cannot initialize compact" in message
    assert "projection 'legacy_front_only' != 'full_fov'" in message


def test_checkpoint_contract_rejects_box_coordinate_geometry(tmp_path: Path):
    checkpoint = tmp_path / "wrong_z_convention.pth"
    source_config = run_tool._effective_rosa_config(_config("balanced"))
    source_mapping = {
        **source_config.__dict__,
        "anchor_z_is_bottom": True,
    }
    metadata = run_tool._checkpoint_contract_metadata(_config("balanced"))
    assert metadata is not None
    metadata["rosa_config"] = source_mapping
    _save_factorized_checkpoint(checkpoint, metadata=metadata)

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


def test_deploy_contract_rejects_incomplete_forged_payload(tmp_path: Path):
    checkpoint = tmp_path / "forged_deploy.pth"
    torch.save(
        {
            "state_dict": {"head.decoder.weight": torch.ones(1)},
            "meta": {
                "deploy_contract": run_tool._OBJECT_DEPTH_DEPLOY_CONTRACT,
                "native_deploy_behavior": run_tool._C73_DEPLOY_BEHAVIOR,
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="metadata key set changed"):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _native_deploy_config(),
            allow_mismatch=False,
        )


def test_checkpoint_contract_rejects_unknown_deploy_behavior(tmp_path: Path):
    checkpoint = tmp_path / "unknown_deploy.pth"
    torch.save(
        {
            "state_dict": {"head.decoder.weight": torch.ones(1)},
            "meta": {
                "deploy_contract": run_tool._OBJECT_DEPTH_DEPLOY_CONTRACT,
                "native_deploy_behavior": "unknown.behavior",
            },
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="native_deploy_behavior"):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _native_deploy_config(),
            allow_mismatch=False,
        )


def test_checkpoint_contract_rejects_behavior_without_deploy_contract(
    tmp_path: Path,
):
    checkpoint = tmp_path / "missing_contract.pth"
    torch.save(
        {
            "state_dict": {"head.decoder.weight": torch.ones(1)},
            "meta": {"native_deploy_behavior": run_tool._C73_DEPLOY_BEHAVIOR},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="missing its deploy contract"):
        run_tool._validate_checkpoint_contract(
            checkpoint,
            _native_deploy_config(),
            allow_mismatch=False,
        )


def test_checkpoint_hook_records_effective_contract():
    metadata = run_tool._checkpoint_contract_metadata(_config("fast"))
    assert metadata is not None
    checkpoint = {"meta": {"epoch": 2}}
    hook = run_tool._CheckpointContractHook(metadata)
    hook.before_save_checkpoint(
        SimpleNamespace(),
        checkpoint,
    )

    assert checkpoint["meta"]["epoch"] == 2
    assert checkpoint["meta"]["rosa_contract_version"] == 3
    assert checkpoint["meta"]["rfl_topology_contract"] == (
        "rosa.rfl.signed_gated_cp16.v3"
    )
    assert checkpoint["meta"]["projection_contract"] == "full_fov"
    assert checkpoint["meta"]["refinement_profile"] == "fast"
    assert checkpoint["meta"]["rosa_config"]["architecture_profile"] == "compact"
    restored = pickle.loads(pickle.dumps(hook))
    assert restored.metadata == hook.metadata


class _FakeRefiner:
    def __init__(self) -> None:
        self._compiled_call_impl = None
        self.compile_calls = []

    def compile(self, **kwargs) -> None:
        self.compile_calls.append(kwargs)
        self._compiled_call_impl = object()


def test_fast_compile_is_idempotent():
    refiner = _FakeRefiner()
    runner = SimpleNamespace(
        model=SimpleNamespace(
            head=SimpleNamespace(rosa_refiner=refiner),
        )
    )
    config = _config("fast")

    assert run_tool._compile_fast_inference(runner, config)
    assert run_tool._compile_fast_inference(runner, config)
    assert refiner.compile_calls == [{"mode": "reduce-overhead", "fullgraph": False}]


def test_retired_dense_compact_v2_fails_before_compile():
    runner = SimpleNamespace(model=SimpleNamespace())
    with pytest.raises(ValueError, match="retired dense compact v2 topology"):
        run_tool._compile_fast_inference(runner, _config("compact_v2_fast"))


def test_non_fast_profile_is_not_compiled():
    runner = SimpleNamespace(model=SimpleNamespace())
    assert not run_tool._compile_fast_inference(runner, _config("balanced"))


def test_evaluate_enables_trusted_mmengine_checkpoint_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.touch()
    config = Config(
        {
            "test_evaluator": {"type": "ToyMetric"},
            "model": {"head": {"type": "Sparse4DHead"}},
        }
    )
    args = SimpleNamespace(
        checkpoint=checkpoint,
        allow_checkpoint_contract_mismatch=False,
        dump=None,
    )
    calls = []

    class FakeRunner:
        @classmethod
        def from_cfg(cls, received_config):
            assert run_tool.os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
            calls.append(("build", received_config.load_from))
            return cls()

        def load_or_resume(self):
            calls.append(("load", None))

        def test(self):
            calls.append(("test", None))

    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.setattr(
        run_tool,
        "_validate_checkpoint_contract",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(run_tool, "Runner", FakeRunner)
    monkeypatch.setattr(
        run_tool,
        "_compile_fast_inference",
        lambda *args, **kwargs: False,
    )

    run_tool.evaluate(args, config)

    assert calls == [
        ("build", str(checkpoint.resolve())),
        ("load", None),
        ("test", None),
    ]


@pytest.mark.parametrize("resume_mode", ("explicit", "auto"))
def test_train_resume_rejects_partial_factorized_state_before_runner_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume_mode: str,
):
    checkpoint = tmp_path / "partial.pth"
    _save_checkpoint(
        checkpoint,
        has_rosa=True,
        metadata=run_tool._checkpoint_contract_metadata(_config("balanced")),
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    if resume_mode == "auto":
        (work_dir / "last_checkpoint").write_text(
            str(checkpoint.resolve()),
            encoding="utf-8",
        )
        resume = "auto"
    else:
        resume = str(checkpoint)

    class MustNotBuild:
        @classmethod
        def from_cfg(cls, _config):
            raise AssertionError("invalid resume reached Runner construction")

    config = _config("balanced")
    config.work_dir = str(work_dir)
    snapshot_factory = run_tool._private_checkpoint_snapshot
    observed = {}

    @contextmanager
    def observe_snapshot(path):
        with snapshot_factory(path) as snapshot:
            observed["snapshot"] = snapshot
            yield snapshot

    monkeypatch.setattr(run_tool, "_private_checkpoint_snapshot", observe_snapshot)
    monkeypatch.setattr(run_tool, "Runner", MustNotBuild)
    with pytest.raises(ValueError, match="topology integrity.*missing keys"):
        run_tool.train(
            SimpleNamespace(resume=resume, amp=False),
            config,
        )
    snapshot = observed["snapshot"]
    assert not snapshot.path.exists()
    assert not snapshot.private_directory.exists()


@pytest.mark.parametrize("resume_mode", ("explicit", "auto"))
def test_train_resume_validates_and_pins_checkpoint_before_runner_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume_mode: str,
):
    checkpoint = tmp_path / "valid.pth"
    _save_factorized_checkpoint(checkpoint)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    if resume_mode == "auto":
        (work_dir / "last_checkpoint").write_text(
            str(checkpoint.resolve()),
            encoding="utf-8",
        )
        resume = "auto"
    else:
        resume = str(checkpoint)

    events = []
    validate = run_tool._validate_checkpoint_contract
    snapshot_factory = run_tool._private_checkpoint_snapshot
    observed_snapshot = {}

    @contextmanager
    def observe_snapshot(path):
        with snapshot_factory(path) as snapshot:
            observed_snapshot["value"] = snapshot
            yield snapshot

    def observed_validate(*args, **kwargs):
        events.append(("validate", args[0], kwargs["allow_mismatch"]))
        return validate(*args, **kwargs)

    training_work_dir = work_dir

    class FakeRunner:
        rank = 1
        work_dir = str(training_work_dir)

        @classmethod
        def from_cfg(cls, received_config):
            snapshot = observed_snapshot["value"]
            assert events == [("validate", snapshot.path, False)]
            assert received_config.resume is True
            assert received_config.load_from == str(snapshot.path)
            assert snapshot.path.is_file()
            assert snapshot.source_path == checkpoint.resolve()
            events.append(("build", received_config.load_from))
            return cls()

        def register_hook(self, _hook, *, priority):
            events.append(("hook", priority))

        def train(self):
            assert observed_snapshot["value"].path.is_file()
            events.append(("train", None))

    config = _config("balanced")
    config.work_dir = str(work_dir)
    monkeypatch.setattr(run_tool, "_private_checkpoint_snapshot", observe_snapshot)
    monkeypatch.setattr(run_tool, "_validate_checkpoint_contract", observed_validate)
    monkeypatch.setattr(run_tool, "Runner", FakeRunner)

    run_tool.train(
        SimpleNamespace(resume=resume, amp=False),
        config,
    )

    snapshot = observed_snapshot["value"]
    assert events == [
        ("validate", snapshot.path, False),
        ("build", str(snapshot.path)),
        ("hook", "VERY_LOW"),
        ("train", None),
    ]
    assert not snapshot.path.exists()
    assert not snapshot.private_directory.exists()


@pytest.mark.parametrize("raise_inside", (False, True))
def test_private_checkpoint_snapshot_is_exact_unlinked_and_always_cleans_up(
    tmp_path: Path,
    raise_inside: bool,
):
    checkpoint = tmp_path / "valid.pth"
    _save_factorized_checkpoint(checkpoint)
    expected_bytes = checkpoint.read_bytes()
    observed = {}

    def exercise() -> None:
        with run_tool._private_checkpoint_snapshot(checkpoint) as snapshot:
            observed["snapshot"] = snapshot
            assert snapshot.source_path == checkpoint.resolve()
            assert snapshot.byte_count == len(expected_bytes)
            assert snapshot.sha256 == hashlib.sha256(expected_bytes).hexdigest()
            assert snapshot.path.read_bytes() == expected_bytes
            assert snapshot.path.is_file()
            assert os.readlink(snapshot.path).endswith(" (deleted)")
            assert snapshot.private_directory.is_dir()
            assert snapshot.private_directory.stat().st_mode & 0o077 == 0
            if raise_inside:
                raise RuntimeError("stop inside snapshot")

    if raise_inside:
        with pytest.raises(RuntimeError, match="stop inside snapshot"):
            exercise()
    else:
        exercise()

    snapshot = observed["snapshot"]
    assert not snapshot.path.exists()
    assert not snapshot.private_directory.exists()


@pytest.mark.parametrize("resume_mode", ("explicit", "auto"))
@pytest.mark.parametrize("attack", ("rename_replace", "in_place_overwrite"))
def test_train_resume_loads_verified_snapshot_after_source_path_attack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume_mode: str,
    attack: str,
):
    checkpoint = tmp_path / "valid.pth"
    _save_factorized_checkpoint(checkpoint)
    original_inode = checkpoint.stat().st_ino
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    if resume_mode == "auto":
        (work_dir / "last_checkpoint").write_text(
            str(checkpoint.resolve()),
            encoding="utf-8",
        )
        resume = "auto"
    else:
        resume = str(checkpoint)

    snapshot_factory = run_tool._private_checkpoint_snapshot
    validate = run_tool._validate_checkpoint_contract
    observed = {}

    @contextmanager
    def observe_snapshot(path):
        with snapshot_factory(path) as snapshot:
            observed["snapshot"] = snapshot
            yield snapshot

    def validate_then_attack(
        snapshot_path,
        config,
        *,
        allow_mismatch,
        require_resume_state=False,
    ):
        validate(
            snapshot_path,
            config,
            allow_mismatch=allow_mismatch,
            require_resume_state=require_resume_state,
        )
        if attack == "rename_replace":
            replacement = tmp_path / "replacement.pth"
            _save_checkpoint(replacement, has_rosa=False)
            os.replace(replacement, checkpoint)
            assert checkpoint.stat().st_ino != original_inode
        else:
            _save_checkpoint(checkpoint, has_rosa=False)
            assert checkpoint.stat().st_ino == original_inode

    training_work_dir = work_dir

    class FakeRunner:
        rank = 1
        work_dir = str(training_work_dir)

        @classmethod
        def from_cfg(cls, received_config):
            snapshot = observed["snapshot"]
            assert received_config.resume is True
            assert received_config.load_from == str(snapshot.path)
            instance = cls()
            instance.load_from = Path(received_config.load_from)
            return instance

        def register_hook(self, _hook, *, priority):
            assert priority == "VERY_LOW"

        def train(self):
            from mmengine.runner.checkpoint import CheckpointLoader

            loaded = CheckpointLoader.load_checkpoint(
                str(self.load_from),
                map_location="cpu",
                logger=None,
            )
            state = run_tool._checkpoint_state_dict(loaded)
            assert state is not None
            assert (
                len([key for key in state if key.startswith("head.rosa_refiner.")])
                == 40
            )
            assert "head.rosa_refiner.rosa.rfl.ray_proj.weight" in state
            observed["loaded"] = True

    config = _config("balanced")
    config.work_dir = str(work_dir)
    monkeypatch.setattr(run_tool, "_private_checkpoint_snapshot", observe_snapshot)
    monkeypatch.setattr(run_tool, "_validate_checkpoint_contract", validate_then_attack)
    monkeypatch.setattr(run_tool, "Runner", FakeRunner)

    run_tool.train(SimpleNamespace(resume=resume, amp=False), config)

    attacked = run_tool._load_checkpoint_without_executing_code(checkpoint)
    assert set(attacked["state_dict"]) == {"head.decoder.weight"}
    assert observed["loaded"] is True
    snapshot = observed["snapshot"]
    assert not snapshot.path.exists()
    assert not snapshot.private_directory.exists()


def test_train_auto_without_checkpoint_pins_a_fresh_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    events = []
    training_work_dir = work_dir

    class FakeRunner:
        rank = 1
        work_dir = str(training_work_dir)

        @classmethod
        def from_cfg(cls, received_config):
            assert received_config.resume is False
            assert received_config.load_from is None
            events.append("build")
            return cls()

        def register_hook(self, _hook, *, priority):
            assert priority == "VERY_LOW"

        def train(self):
            events.append("train")

    def must_not_validate(*_args, **_kwargs):
        raise AssertionError("missing auto checkpoint reached validation")

    config = _config("balanced")
    config.work_dir = str(work_dir)
    monkeypatch.setattr(run_tool, "_validate_checkpoint_contract", must_not_validate)
    monkeypatch.setattr(run_tool, "Runner", FakeRunner)

    run_tool.train(SimpleNamespace(resume="auto", amp=False), config)

    assert events == ["build", "train"]


def _object_depth_source_state() -> OrderedDict:
    state = OrderedDict(
        (f"native.layer_{index}", torch.tensor([float(index)]))
        for index in range(run_tool._OBJECT_DEPTH_DEPLOY_STATE_KEYS)
    )
    for level in range(3):
        state[f"depth_branch.depth_layers.{level}.weight"] = torch.ones(1, 256, 1, 1)
        state[f"depth_branch.depth_layers.{level}.bias"] = torch.ones(1)
    state._metadata = OrderedDict(
        [
            ("", {"version": 1}),
            ("native", {"version": 1}),
            ("depth_branch", {"version": 1}),
            ("depth_branch.depth_layers", {"version": 1}),
            ("depth_branch.depth_layers.0", {"version": 1}),
            ("depth_branch.depth_layers.1", {"version": 1}),
            ("depth_branch.depth_layers.2", {"version": 1}),
        ]
    )
    return state


def _allow_synthetic_object_depth_state(
    monkeypatch: pytest.MonkeyPatch,
    state: OrderedDict,
) -> None:
    depth_keys = set(run_tool._OBJECT_DEPTH_TRAIN_ONLY_KEYS)
    deploy_state = OrderedDict(
        (key, value) for key, value in state.items() if key not in depth_keys
    )
    monkeypatch.setattr(
        run_tool,
        "_OBJECT_DEPTH_DEPLOY_STATE_ELEMENTS",
        sum(tensor.numel() for tensor in deploy_state.values()),
    )
    monkeypatch.setattr(
        run_tool,
        "_OBJECT_DEPTH_SOURCE_STRUCTURE_SHA256",
        run_tool._state_structure_sha256(state),
    )
    monkeypatch.setattr(
        run_tool,
        "_OBJECT_DEPTH_DEPLOY_STRUCTURE_SHA256",
        run_tool._state_structure_sha256(deploy_state),
    )
    source_metadata = state._metadata
    depth_metadata_keys = set(run_tool._OBJECT_DEPTH_TRAIN_ONLY_METADATA_KEYS)
    deploy_metadata = OrderedDict(
        (key, value)
        for key, value in source_metadata.items()
        if key not in depth_metadata_keys
    )
    monkeypatch.setattr(
        run_tool,
        "_OBJECT_DEPTH_SOURCE_METADATA_KEYS",
        len(source_metadata),
    )
    monkeypatch.setattr(
        run_tool,
        "_OBJECT_DEPTH_DEPLOY_METADATA_KEYS",
        len(deploy_metadata),
    )
    monkeypatch.setattr(
        run_tool,
        "_OBJECT_DEPTH_SOURCE_METADATA_SHA256",
        run_tool._state_metadata_sha256(source_metadata),
    )
    monkeypatch.setattr(
        run_tool,
        "_OBJECT_DEPTH_DEPLOY_METADATA_SHA256",
        run_tool._state_metadata_sha256(deploy_metadata),
    )


def _approve_synthetic_source(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
) -> None:
    monkeypatch.setattr(
        run_tool,
        "_C73_APPROVED_SOURCE_SHA256",
        frozenset({run_tool._sha256_file(source)}),
    )


def test_export_deploy_strips_only_train_depth_and_training_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "epoch_8.pth"
    output = tmp_path / "deploy.pth"
    state = _object_depth_source_state()
    _allow_synthetic_object_depth_state(monkeypatch, state)
    torch.save(
        {
            "meta": {"epoch": 8},
            "state_dict": state,
            "optimizer": {"large_train_only": torch.ones(4096)},
            "param_schedulers": [{"last_epoch": 8}],
        },
        source,
    )
    _approve_synthetic_source(monkeypatch, source)
    result = run_tool.export_deploy(SimpleNamespace(checkpoint=source, output=output))

    exported = torch.load(output, map_location="cpu", weights_only=True)
    assert set(exported) == {"meta", "state_dict"}
    assert list(exported["state_dict"]) == [
        f"native.layer_{index}"
        for index in range(run_tool._OBJECT_DEPTH_DEPLOY_STATE_KEYS)
    ]
    assert exported["meta"]["stripped_state_keys"] == list(
        run_tool._OBJECT_DEPTH_TRAIN_ONLY_KEYS
    )
    assert exported["meta"]["stripped_elements"] == 771
    assert exported["meta"]["native_deploy_behavior"] == (run_tool._C73_DEPLOY_BEHAVIOR)
    assert exported["meta"]["native_deploy_target_head_type"] == (
        "ROSAOfficialResidualNativeBaselineHead"
    )
    assert exported["meta"]["optimizer_state_included"] is False
    assert exported["meta"]["scheduler_state_included"] is False
    assert result["state_keys"] == run_tool._OBJECT_DEPTH_DEPLOY_STATE_KEYS
    assert result["output_sha256"] == run_tool._sha256_file(output)
    assert source.exists()
    run_tool._validate_checkpoint_contract(
        output,
        _native_deploy_config(),
        allow_mismatch=False,
    )
    with pytest.raises(ValueError, match="requires head"):
        run_tool._validate_checkpoint_contract(
            output,
            _native_deploy_config(head_type="ROSANativeBaselineHead"),
            allow_mismatch=False,
        )
    with pytest.raises(ValueError, match="exact homogeneous"):
        run_tool._validate_checkpoint_contract(
            output,
            _native_deploy_config(
                deformable_type=(
                    "ROSAMetricGatedAffine3x3FullFOV" "DeformableFeatureAggregation"
                )
            ),
            allow_mismatch=False,
        )


def test_deploy_validator_rejects_removed_behavior_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "epoch_8.pth"
    output = tmp_path / "deploy.pth"
    tampered = tmp_path / "tampered.pth"
    state = _object_depth_source_state()
    _allow_synthetic_object_depth_state(monkeypatch, state)
    torch.save({"state_dict": state}, source)
    _approve_synthetic_source(monkeypatch, source)
    run_tool.export_deploy(SimpleNamespace(checkpoint=source, output=output))
    payload = torch.load(output, map_location="cpu", weights_only=True)
    payload["meta"].pop("native_deploy_behavior")
    torch.save(payload, tampered)

    with pytest.raises(ValueError, match="metadata key set changed"):
        run_tool._validate_checkpoint_contract(
            tampered,
            _native_deploy_config(),
            allow_mismatch=False,
        )


def test_export_deploy_fails_closed_on_changed_depth_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "changed.pth"
    state = _object_depth_source_state()
    state["depth_branch.depth_layers.2.weight"] = torch.ones(1, 255, 1, 1)
    _allow_synthetic_object_depth_state(monkeypatch, state)
    torch.save({"state_dict": state}, source)
    _approve_synthetic_source(monkeypatch, source)

    with pytest.raises(ValueError, match="shape changed"):
        run_tool.export_deploy(
            SimpleNamespace(
                checkpoint=source,
                output=tmp_path / "must_not_exist.pth",
            )
        )
    assert not (tmp_path / "must_not_exist.pth").exists()


def test_export_deploy_rejects_retained_state_structure_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "mutated_native.pth"
    expected_state = _object_depth_source_state()
    _allow_synthetic_object_depth_state(monkeypatch, expected_state)
    mutated_state = expected_state.copy()
    mutated_state._metadata = expected_state._metadata.copy()
    mutated_state["native.renamed_layer"] = mutated_state.pop("native.layer_0")
    torch.save({"state_dict": mutated_state}, source)
    _approve_synthetic_source(monkeypatch, source)

    with pytest.raises(ValueError, match="source state structure changed"):
        run_tool.export_deploy(
            SimpleNamespace(
                checkpoint=source,
                output=tmp_path / "must_not_exist.pth",
            )
        )
    assert not (tmp_path / "must_not_exist.pth").exists()


def test_export_deploy_rejects_module_metadata_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "mutated_metadata.pth"
    expected_state = _object_depth_source_state()
    _allow_synthetic_object_depth_state(monkeypatch, expected_state)
    mutated_state = expected_state.copy()
    mutated_state._metadata = OrderedDict(
        (key, dict(value)) for key, value in expected_state._metadata.items()
    )
    mutated_state._metadata["native"]["version"] = 999
    torch.save({"state_dict": mutated_state}, source)
    _approve_synthetic_source(monkeypatch, source)

    with pytest.raises(ValueError, match="source metadata changed"):
        run_tool.export_deploy(
            SimpleNamespace(
                checkpoint=source,
                output=tmp_path / "must_not_exist.pth",
            )
        )
    assert not (tmp_path / "must_not_exist.pth").exists()


def test_export_deploy_rejects_unapproved_same_schema_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "same_schema_but_unapproved.pth"
    state = _object_depth_source_state()
    _allow_synthetic_object_depth_state(monkeypatch, state)
    torch.save({"state_dict": state}, source)

    with pytest.raises(ValueError, match="not an approved C73"):
        run_tool.export_deploy(
            SimpleNamespace(
                checkpoint=source,
                output=tmp_path / "must_not_exist.pth",
            )
        )
    assert not (tmp_path / "must_not_exist.pth").exists()


def test_export_deploy_never_overwrites_existing_output(tmp_path: Path):
    source = tmp_path / "source.pth"
    output = tmp_path / "existing.pth"
    torch.save({"state_dict": _object_depth_source_state()}, source)
    output.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError):
        run_tool.export_deploy(SimpleNamespace(checkpoint=source, output=output))
    assert output.read_bytes() == b"keep-me"


def test_approved_source_is_deserialized_from_its_hashed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source.pth"
    approved_bytes = b"approved checkpoint snapshot"
    changed_bytes = b"path changed after snapshot"
    source.write_bytes(approved_bytes)
    approved_sha256 = hashlib.sha256(approved_bytes).hexdigest()
    monkeypatch.setattr(
        run_tool,
        "_C73_APPROVED_SOURCE_SHA256",
        frozenset({approved_sha256}),
    )

    def load_snapshot(snapshot):
        source.write_bytes(changed_bytes)
        assert snapshot.read() == approved_bytes
        return {"state_dict": "captured"}

    monkeypatch.setattr(
        run_tool,
        "_load_checkpoint_without_executing_code",
        load_snapshot,
    )

    checkpoint, observed_sha256, byte_count = run_tool._load_approved_c73_checkpoint(
        source
    )

    assert checkpoint == {"state_dict": "captured"}
    assert observed_sha256 == approved_sha256
    assert byte_count == len(approved_bytes)
    assert source.read_bytes() == changed_bytes


def test_unapproved_source_is_rejected_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source.pth"
    source.write_bytes(b"unapproved checkpoint")
    monkeypatch.setattr(
        run_tool,
        "_C73_APPROVED_SOURCE_SHA256",
        frozenset({hashlib.sha256(b"approved").hexdigest()}),
    )

    def must_not_deserialize(_snapshot):
        raise AssertionError("unapproved bytes reached deserialization")

    monkeypatch.setattr(
        run_tool,
        "_load_checkpoint_without_executing_code",
        must_not_deserialize,
    )

    with pytest.raises(ValueError, match="not an approved C73"):
        run_tool._load_approved_c73_checkpoint(source)
