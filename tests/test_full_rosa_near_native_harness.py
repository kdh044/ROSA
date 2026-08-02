"""CPU-only contracts for the disarmed full-ROSA speed harness."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/verify_full_rosa_near_native.py"
SPEC = importlib.util.spec_from_file_location("full_rosa_speed_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def test_module_import_and_disarmed_main_are_stdlib_only_subprocess() -> None:
    script = r"""
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import tempfile

forbidden = {'torch', 'numpy', 'mmengine', 'mmcv', 'mmdet', 'mmdet3d', 'rosa'}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in forbidden:
            raise AssertionError('forbidden accelerator import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
spec = importlib.util.spec_from_file_location('isolated_full_rosa_tool', __TOOL_PATH__)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def forbidden_armed_call(*args, **kwargs):
    raise AssertionError('armed-only helper reached from dry plan')

module._load_armed_verifier = forbidden_armed_call
module._acquire_physical_gpu_lock = forbidden_armed_call
module._nvidia_smi_snapshot = forbidden_armed_call

with tempfile.TemporaryDirectory() as raw_temp:
    temp = Path(raw_temp)
    checkpoint = temp / 'checkpoint.pth'
    checkpoint.write_bytes(b'checkpoint')
    config = temp / 'native.py'
    config.write_text('model = {}\n', encoding='utf-8')
    annotation = temp / 'val.pkl'
    annotation.write_bytes(b'annotation')
    detector = temp / 'detector'
    official = temp / 'official'
    official.mkdir()
    (official / 'dataset.py').write_text('VALUE = 1\n', encoding='utf-8')

    for relative in sorted(set(module.HOST_SOURCE_FILES + module.HOST_DAF_SOURCE_FILES)):
        path = detector / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + '\n', encoding='utf-8')
    daf = detector / 'projects/mmdet3d_plugin/ops' / module.DAF_EXTENSION_BASENAME
    daf.write_bytes(b'daf-binary')

    source_revision = '1' * 40
    host_revision = '2' * 40
    official_revision = '3' * 40
    module.DEFAULT_NATIVE_CONFIG = config
    module.DEFAULT_MAIN126_ANNOTATION = annotation
    module.EXPECTED_MAIN126_ANNOTATION_SHA256 = module.file_sha256(annotation)
    module.DEFAULT_DAF_EXTENSION_SHA256 = module.file_sha256(daf)
    module.EXPECTED_DETECTOR_HOST_REVISION = host_revision
    module.EXPECTED_OFFICIAL_REVISION = official_revision
    module.BLESSED_DAF_BUILD_INPUT_SHA256 = {
        relative: module.file_sha256(detector / relative)
        for relative in (
            'projects/mmdet3d_plugin/ops/setup.py',
            'projects/mmdet3d_plugin/ops/src/deformable_aggregation.cpp',
            'projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu',
        )
    }
    wrapper = 'projects/mmdet3d_plugin/ops/deformable_aggregation.py'
    module.BLESSED_DAF_WRAPPER_SHA256 = module.file_sha256(detector / wrapper)
    module.EXPECTED_OFFICIAL_SOURCE_SHA256 = module._source_contract(
        official, suffixes=('.py',)
    )['sha256']

    revisions = {
        module.ROOT.resolve(): source_revision,
        detector.resolve(): host_revision,
        official.resolve(): official_revision,
    }
    def snapshot(root):
        resolved = Path(root).resolve()
        return {
            'root': str(resolved),
            'revision': revisions[resolved],
            'tree': '4' * 40,
            'status': '',
        }
    module._git_snapshot = snapshot

    for profile, candidate in (
        ('fast', module.FAST_ARM),
        ('balanced', module.BALANCED_ARM),
    ):
        output = temp / ('not-created-' + profile)
        argv = [
            '--checkpoint', str(checkpoint),
            '--expected-checkpoint-sha256', module.file_sha256(checkpoint),
            '--expected-source-revision', source_revision,
            '--native-config', str(config),
            '--detector-host-root', str(detector),
            '--expected-detector-host-revision', host_revision,
            '--official-root', str(official),
            '--expected-official-revision', official_revision,
            '--expected-daf-extension-sha256', module.file_sha256(daf),
            '--output-dir', str(output),
            '--candidate-profile', profile,
        ]
        args = module.build_parser().parse_args(argv)
        plan = module.prepare_plan(args)
        assert set(plan['arms']) == {module.NATIVE_ARM, candidate}
        assert set(plan['comparisons']) == {candidate}
        assert plan['candidate_profile'] == profile
        assert module.main(argv) == 0
        assert not output.exists()

assert not any(name.split('.', 1)[0] in forbidden for name in sys.modules)
print('stdlib-only-ok')
"""
    script = script.replace("__TOOL_PATH__", repr(str(TOOL_PATH)))
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "stdlib-only-ok" in result.stdout
    assert '"cuda_queried": false' in result.stdout


@pytest.mark.parametrize(
    ("candidate", "iterations"),
    ((tool.FAST_ARM, 60), (tool.BALANCED_ARM, 300)),
)
def test_pair_schedule_is_balanced_pccp_cppc_and_replays_one_batch(
    candidate: str, iterations: int
) -> None:
    schedule = tool.build_alternating_schedule(
        iterations_per_arm=iterations,
        phase="measured",
        candidate_arm=candidate,
        batch_count=2,
    )
    assert len(schedule) == iterations * 2
    assert sum(row["arm"] == tool.NATIVE_ARM for row in schedule) == iterations
    assert sum(row["arm"] == candidate for row in schedule) == iterations

    grouped = {}
    for row in schedule:
        grouped.setdefault(row["pair_index"], []).append(row)
    assert sorted(grouped) == list(range(iterations))
    for pair_index, rows in grouped.items():
        assert {row["arm"] for row in rows} == {tool.NATIVE_ARM, candidate}
        assert {row["batch_index"] for row in rows} == {pair_index % 2}
        assert len({row["round_index"] for row in rows}) == 1
        assert len({row["pattern"] for row in rows}) == 1
    assert (
        tuple(
            next(row["pattern"] for row in schedule if row["round_index"] == index)
            for index in range(6)
        )
        == tool.PHYSICAL_ROUND_ORDER
    )


@pytest.mark.parametrize(
    ("profile", "candidate"),
    (("fast", tool.FAST_ARM), ("balanced", tool.BALANCED_ARM)),
)
def test_candidate_profile_selects_exactly_one_pair(profile, candidate) -> None:
    arms = tool.selected_arms(profile)

    assert arms == (tool.NATIVE_ARM, candidate)
    assert tool.selected_candidate(profile) == candidate
    assert set(tool._arms_plan(arms)) == {tool.NATIVE_ARM, candidate}


@pytest.mark.parametrize(
    ("state", "idle", "stream", "message"),
    (
        ("DISARMED", tool.IDLE_RESERVED_TOKEN, tool.CURRENT_STREAM_TOKEN, "ARMED"),
        ("ARMED", "wrong", tool.CURRENT_STREAM_TOKEN, "idle/reserved"),
        ("ARMED", tool.IDLE_RESERVED_TOKEN, "wrong", "current-stream"),
    ),
)
def test_execution_gate_fails_before_any_armed_import(
    monkeypatch, state, idle, stream, message
) -> None:
    monkeypatch.setattr(
        tool,
        "_load_armed_verifier",
        lambda: (_ for _ in ()).throw(AssertionError("armed import reached")),
    )
    args = SimpleNamespace(
        execute=True,
        execution_state=state,
        idle_reserved_token=idle,
        current_stream_token=stream,
    )
    with pytest.raises(tool.ProtocolError, match=message):
        tool.validate_execution_gate(args)


@pytest.mark.parametrize(
    "value",
    (
        "GPU-E79E0304-D240-431B-F05F-7F3A67933B97",
        "gpu-e79e0304-d240-431b-f05f-7f3a67933b97",
        b"GPU-e79e0304-d240-431b-f05f-7f3a67933b97",
    ),
)
def test_gpu_uuid_normalization_matches_nvidia_smi_selector_case(value) -> None:
    assert tool._normalize_gpu_uuid(value) == (
        "GPU-e79e0304-d240-431b-f05f-7f3a67933b97"
    )


def test_gpu_selection_requires_canonical_lower_uuid(monkeypatch) -> None:
    args = SimpleNamespace(execute=True, device="cuda:0")
    canonical = "GPU-e79e0304-d240-431b-f05f-7f3a67933b97"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", canonical)
    assert tool._gpu_selection_contract(args)["physical_uuid"] == canonical

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", canonical.upper())
    with pytest.raises(tool.ProtocolError, match="lower-case UUID suffix"):
        tool._gpu_selection_contract(args)


def test_nvidia_smi_snapshot_uses_canonical_lower_selector(monkeypatch) -> None:
    canonical = "GPU-e79e0304-d240-431b-f05f-7f3a67933b97"
    commands = []

    def run(command, **kwargs):
        del kwargs
        commands.append(command)
        if any(value.startswith("--query-gpu=") for value in command):
            row = (
                "2026/08/02 20:09:39.000, 0, " + canonical
                + ", NVIDIA GPU, 580.65.06, P0, 40.0, 450.0, 210, 405"
            )
        else:
            row = ""
        return SimpleNamespace(stdout=row)

    monkeypatch.setattr(tool.subprocess, "run", run)
    snapshot = tool._nvidia_smi_snapshot(canonical.upper())

    assert snapshot["selected_gpu_uuid"] == canonical
    assert len(commands) == 2
    assert all(command[1] == f"--id={canonical}" for command in commands)


def test_current_stream_guard_uses_ids_without_real_torch() -> None:
    class FakeCuda:
        @staticmethod
        def current_stream(device):
            del device
            return SimpleNamespace(cuda_stream=17)

        @staticmethod
        def default_stream(device):
            del device
            return SimpleNamespace(cuda_stream=23)

    with pytest.raises(tool.ProtocolError, match="not the default"):
        tool._require_current_default_stream(SimpleNamespace(cuda=FakeCuda()), "cuda:0")


def test_armed_execution_rejects_preimported_accelerator_module(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(tool.ProtocolError, match="standard-library-only"):
        tool._reject_preimported_armed_modules()


def test_lock_write_failure_closes_and_removes_owned_sentinel(monkeypatch) -> None:
    gpu_uuid = f"GPU-UNITTEST-WRITE-{id(object())}"
    canonical = tool._normalize_gpu_uuid(gpu_uuid)
    lock = Path("/tmp") / f".full_rosa_near_native_{canonical}.lock"

    def fail_write(descriptor, payload):
        del descriptor, payload
        raise OSError("injected write failure")

    monkeypatch.setattr(tool.os, "write", fail_write)
    try:
        with pytest.raises(OSError, match="injected write failure"):
            tool._acquire_physical_gpu_lock(gpu_uuid)
        assert not lock.exists()
    finally:
        lock.unlink(missing_ok=True)


def test_lock_release_does_not_unlink_a_replaced_path() -> None:
    gpu_uuid = f"GPU-UNITTEST-REPLACE-{id(object())}"
    descriptor, lock = tool._acquire_physical_gpu_lock(gpu_uuid)
    lock.unlink()
    lock.write_text("replacement\n", encoding="utf-8")
    try:
        tool._release_physical_gpu_lock(descriptor, lock)
        assert lock.read_text(encoding="utf-8") == "replacement\n"
    finally:
        lock.unlink(missing_ok=True)


def test_atomic_publication_removes_new_marker_when_directory_fsync_fails(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "COMPLETE.json"
    monkeypatch.setattr(
        tool,
        "_fsync_directory",
        lambda directory: (_ for _ in ()).throw(OSError(f"fsync {directory}")),
    )

    with pytest.raises(OSError, match="fsync"):
        tool._atomic_json(output, {"status": "COMPLETE"})
    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()


@pytest.mark.parametrize(
    ("candidate", "expected_compiles"),
    ((tool.FAST_ARM, 1), (tool.BALANCED_ARM, 0)),
)
def test_memory_measurement_builds_only_one_model_at_a_time(
    monkeypatch, candidate, expected_compiles
) -> None:
    build_calls = []
    compile_calls = []
    reset_calls = []

    class Model:
        pass

    def build_models(verifier, configs, state_dict, arms, device):
        del verifier, configs, state_dict, device
        build_calls.append(tuple(arms))
        assert len(arms) == 1
        arm = arms[0]
        return {arm: Model()}, {arm: {"arm": arm}}, {arm: {"loaded": True}}

    class Cuda:
        @staticmethod
        def current_stream(device):
            del device
            return SimpleNamespace(cuda_stream=7)

        @staticmethod
        def default_stream(device):
            del device
            return SimpleNamespace(cuda_stream=7)

    class Verifier:
        probes = 0

        @classmethod
        def _memory_probe(cls, model, source, device):
            del model, source, device
            cls.probes += 1
            return {
                "peak_allocated_bytes": 100 + cls.probes,
                "peak_reserved_bytes": 200 + cls.probes,
                "peak_allocated_increment_bytes": 10 + cls.probes,
                "peak_reserved_increment_bytes": 20 + cls.probes,
            }

    class RunTool:
        @staticmethod
        def _compile_fast_inference(model, config):
            compile_calls.append((model, config))
            return True

    monkeypatch.setattr(tool, "_build_models", build_models)
    monkeypatch.setattr(
        tool,
        "_preflight_outputs",
        lambda verifier, torch, models, source, device: {
            arm: {"raw_fingerprint": arm} for arm in models
        },
    )
    monkeypatch.setattr(
        tool,
        "_stabilize_cuda_allocator",
        lambda torch, device: {"allocated_bytes": 0, "reserved_bytes": 0},
    )
    monkeypatch.setattr(
        tool, "_reset_compiler_cache", lambda torch: reset_calls.append(torch)
    )

    configs = {tool.NATIVE_ARM: object(), candidate: object()}
    fake_torch = SimpleNamespace(cuda=Cuda())
    memory, contracts, loads = tool._measure_isolated_memory(
        Verifier,
        fake_torch,
        RunTool,
        configs,
        {},
        [object(), object()],
        "cuda:0",
        candidate,
    )

    assert build_calls == [(tool.NATIVE_ARM,), (candidate,)]
    assert len(compile_calls) == expected_compiles
    assert len(reset_calls) == expected_compiles
    assert set(contracts) == set(loads) == {tool.NATIVE_ARM, candidate}
    assert memory["contract"]["one_benchmark_model_resident_per_probe"] is True
    for arm in (tool.NATIVE_ARM, candidate):
        assert memory[arm]["resident_benchmark_models"] == [arm]
        assert memory[arm]["peer_model_resident"] is False
        assert len(memory[arm]["probe_records"]) == 3


def test_fast_timing_rejects_compiler_counter_changes() -> None:
    counters = {"stats": {"unique_graphs": 1}}

    class Cuda:
        @staticmethod
        def current_stream(device):
            del device
            return SimpleNamespace(cuda_stream=11)

        @staticmethod
        def default_stream(device):
            del device
            return SimpleNamespace(cuda_stream=11)

    class Verifier:
        C0_C32_FULL_DETECTOR_SCOPE = "full"
        PARENT_ARM = "parent"
        CANDIDATE_ARM = "candidate"

        @staticmethod
        def _run_scope(**kwargs):
            del kwargs
            return {"paired_bootstrap": {}}

    fake_torch = SimpleNamespace(
        cuda=Cuda(),
        _dynamo=SimpleNamespace(utils=SimpleNamespace(counters=counters)),
    )
    models = {tool.NATIVE_ARM: object(), tool.FAST_ARM: object()}
    result = tool._run_pair_protocol(
        Verifier, fake_torch, models, [], [], "cuda:0", tool.FAST_ARM
    )
    assert result["compiler_guard"]["applicable"] is True
    assert result["compiler_guard"]["compilation_or_recompile_observed"] is False

    def recompile(**kwargs):
        del kwargs
        counters["stats"]["unique_graphs"] += 1
        return {"paired_bootstrap": {}}

    Verifier._run_scope = staticmethod(recompile)
    with pytest.raises(tool.ProtocolError, match="compiler/recompile"):
        tool._run_pair_protocol(
            Verifier, fake_torch, models, [], [], "cuda:0", tool.FAST_ARM
        )


def test_balanced_timing_does_not_require_compiler_counters() -> None:
    class Cuda:
        @staticmethod
        def current_stream(device):
            del device
            return SimpleNamespace(cuda_stream=13)

        @staticmethod
        def default_stream(device):
            del device
            return SimpleNamespace(cuda_stream=13)

    verifier = SimpleNamespace(
        C0_C32_FULL_DETECTOR_SCOPE="full",
        PARENT_ARM="parent",
        CANDIDATE_ARM="candidate",
        _run_scope=lambda **kwargs: {"paired_bootstrap": {}},
    )
    result = tool._run_pair_protocol(
        verifier,
        SimpleNamespace(cuda=Cuda()),
        {tool.NATIVE_ARM: object(), tool.BALANCED_ARM: object()},
        [],
        [],
        "cuda:0",
        tool.BALANCED_ARM,
    )
    assert result["compiler_guard"]["applicable"] is False


@dataclass(frozen=True)
class _SemanticConfig:
    refine_topk: int
    num_refinement_steps: int
    projection_contract: str
    max_candidate_range_m: float
    enable_rfl: bool

    @classmethod
    def from_mapping(cls, value):
        return cls(**dict(value))


class _SemanticRunTool:
    target = _SemanticConfig(50, 1, "full_fov", 68.0, True)

    @classmethod
    def _effective_rosa_config(cls, config):
        del config
        return cls.target

    @staticmethod
    def _checkpoint_state_dict(checkpoint):
        return checkpoint["state_dict"]

    @staticmethod
    def _state_dict_has_rosa(state_dict):
        del state_dict
        return True


def _semantic_checkpoint(**source_overrides):
    source = {
        "refine_topk": 100,
        "num_refinement_steps": 2,
        "projection_contract": "full_fov",
        "max_candidate_range_m": 68.0,
        "enable_rfl": True,
    }
    source.update(source_overrides)
    return {
        "meta": {
            "rosa_config": source,
            "refinement_profile": "balanced",
            "rosa_contract_version": 2,
            "projection_contract": "full_fov",
        },
        "state_dict": {"head.rosa_refiner.weight": "sealed"},
    }


def test_checkpoint_semantics_allows_only_the_preregistered_fast_budget_delta() -> None:
    record = tool._validate_checkpoint_semantics(
        _SemanticRunTool,
        _semantic_checkpoint(),
        object(),
        tool.FAST_ARM,
    )

    assert record["validated_against_arm"] == tool.FAST_ARM
    assert record["source_refinement_profile"] == "balanced"
    assert record["allowed_runtime_delta"] == tool.FAST_CHECKPOINT_ALLOWED_DELTAS
    assert record["observed_runtime_delta"] == tool.FAST_CHECKPOINT_ALLOWED_DELTAS
    assert record["all_other_effective_fields_exact"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda checkpoint: checkpoint["meta"].pop("rosa_config"), "rosa_config"),
        (
            lambda checkpoint: checkpoint["meta"]["rosa_config"].update(
                max_candidate_range_m=67.0
            ),
            "delta is not allowed",
        ),
        (
            lambda checkpoint: checkpoint["meta"].update(
                projection_contract="legacy_front_only"
            ),
            "projection contract",
        ),
        (
            lambda checkpoint: checkpoint["meta"].update(
                refinement_profile="published"
            ),
            "not balanced",
        ),
    ),
)
def test_checkpoint_semantics_fails_closed_on_missing_or_unrelated_metadata(
    mutation, message
) -> None:
    checkpoint = _semantic_checkpoint()
    mutation(checkpoint)
    with pytest.raises(tool.ProtocolError, match=message):
        tool._validate_checkpoint_semantics(
            _SemanticRunTool, checkpoint, object(), tool.FAST_ARM
        )


def test_compiled_parity_requires_raw_exact_output_for_every_batch() -> None:
    eager = {
        "0": {"raw_fingerprint": "batch-zero"},
        "1": {"raw_fingerprint": "batch-one"},
    }
    compiled = {
        "0": {"raw_fingerprint": "batch-zero"},
        "1": {"raw_fingerprint": "batch-one"},
    }
    parity = tool._require_all_batch_compiled_parity(eager, compiled, batch_count=2)
    assert set(parity) == {"0", "1"}
    assert all(record["raw_exact"] for record in parity.values())

    compiled["1"] = {"raw_fingerprint": "different"}
    with pytest.raises(tool.ProtocolError, match="captured batch 1"):
        tool._require_all_batch_compiled_parity(eager, compiled, batch_count=2)

    with pytest.raises(tool.ProtocolError, match="every captured batch"):
        tool._require_all_batch_compiled_parity(
            eager, {"0": compiled["0"]}, batch_count=2
        )
    with pytest.raises(tool.ProtocolError, match="lacks raw fingerprint"):
        tool._require_all_batch_compiled_parity(
            {"0": {}, "1": eager["1"]},
            {"0": {}, "1": eager["1"]},
            batch_count=2,
        )


@pytest.mark.parametrize("profile", ("fast", "balanced"))
def test_candidate_config_changes_only_the_preregistered_head_fields(profile) -> None:
    native = {
        "model": {
            "backbone": {"depth": 18},
            "head": {"type": "ROSANativeBaselineHead", "rosa_cfg": None},
        },
        "work_dir": "unchanged",
        "seed": 3407,
    }

    candidate = tool._expected_candidate_config(native, profile)

    assert native["model"]["head"] == {
        "type": "ROSANativeBaselineHead",
        "rosa_cfg": None,
    }
    assert candidate["model"]["backbone"] == native["model"]["backbone"]
    assert candidate["work_dir"] == native["work_dir"]
    assert candidate["seed"] == native["seed"]
    assert candidate["model"]["head"] == {
        "type": "ROSARefinementHead",
        "rosa_cfg": {"profile": profile},
    }


def test_native_checkpoint_filter_accepts_only_rosa_sidecar_state() -> None:
    tool._validate_native_checkpoint_exclusions(
        ["head.rosa_refiner.confidence.weight", "head.rosa_refiner.yaw.bias"]
    )
    with pytest.raises(tool.ProtocolError, match="non-ROSA"):
        tool._validate_native_checkpoint_exclusions(
            ["head.rosa_refiner.yaw.bias", "head.decoder.weight"]
        )


@pytest.mark.parametrize("profile", ("fast", "balanced"))
def test_effective_dataset_contract_pins_main126_annotation_and_batch_size(
    tmp_path, profile
) -> None:
    annotation = tmp_path / "val.pkl"
    annotation.write_bytes(b"frozen")
    dataset = SimpleNamespace(
        type="Fisheye3DODSparse4DDataset",
        ann_file=str(annotation),
    )
    configs = {
        arm: SimpleNamespace(
            test_dataloader=SimpleNamespace(batch_size=4, dataset=dataset)
        )
        for arm in tool.selected_arms(profile)
    }

    tool._validate_effective_dataset_configs(configs, annotation)

    configs[tool.selected_candidate(profile)].test_dataloader.batch_size = 1
    with pytest.raises(tool.ProtocolError, match="batch size"):
        tool._validate_effective_dataset_configs(configs, annotation)


def test_near_native_gate_maps_physical_arms_and_freezes_thresholds() -> None:
    captured = {}

    class Verifier:
        PARENT_ARM = "logical_parent"
        CANDIDATE_ARM = "logical_candidate"

        @staticmethod
        def evaluate_c0_c32_near_native_gate(paired, memory):
            captured["paired"] = paired
            captured["memory"] = memory
            return {
                "passed": True,
                "speed_claim_eligible": True,
                "decision": "NEAR_NATIVE_SUPPORTED",
                "checks": {
                    "median_ratio_ci95_upper_at_most_1_10": {
                        "threshold_max_inclusive": 1.10,
                        "passed": True,
                    },
                    "p95_ratio_at_most_1_10": {
                        "threshold_max_inclusive": 1.10,
                        "passed": True,
                    },
                    "peak_allocated_bytes_regression_at_most_2_percent": {
                        "threshold_max_inclusive": 1.02,
                        "passed": True,
                    },
                    "peak_reserved_bytes_regression_at_most_2_percent": {
                        "threshold_max_inclusive": 1.02,
                        "passed": True,
                    },
                },
            }

    paired = {"metrics": "sealed"}
    memory = {
        tool.NATIVE_ARM: {"peak_allocated_bytes": 100},
        tool.FAST_ARM: {"peak_allocated_bytes": 101},
    }

    tool._evaluate_near_native_gate(
        Verifier,
        {"paired_bootstrap": paired},
        memory,
        tool.FAST_ARM,
    )

    assert captured["paired"] is paired
    assert captured["memory"] == {
        "logical_parent": memory[tool.NATIVE_ARM],
        "logical_candidate": memory[tool.FAST_ARM],
    }


def test_plan_json_shape_is_serializable_without_accelerator_types() -> None:
    arms = tool._arms_plan(tool.selected_arms("balanced"))
    encoded = json.dumps(arms, sort_keys=True)
    assert "ROSARefinementHead" in encoded
    assert set(arms) == {tool.NATIVE_ARM, tool.BALANCED_ARM}
