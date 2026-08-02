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
        ('fast_eager', module.FAST_EAGER_ARM),
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
        assert plan['schema'].endswith('.v2')
        assert plan['protocol'].endswith('-v2')
        compiler_guard = plan['measurement']['compiler_guard']
        if profile == 'fast':
            assert compiler_guard['applicable'] is True
            assert 'raw-exact' in compiler_guard['requirement']
        else:
            assert compiler_guard['applicable'] is False
            assert 'reason' in compiler_guard
        if profile == 'fast_eager':
            assert plan['candidate_model_profile'] == 'fast'
            assert plan['candidate_execution_mode'] == 'eager'
            assert plan['arms'][candidate]['compile'] is False
            assert plan['arms'][candidate]['compile_forbidden'] is True
            assert plan['arms'][candidate]['config_derivation']['only_changes'][
                'model.head.rosa_cfg'
            ] == {'profile': 'fast'}
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
    (
        (tool.FAST_ARM, 60),
        (tool.FAST_EAGER_ARM, 300),
        (tool.BALANCED_ARM, 300),
    ),
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
    ("profile", "candidate", "model_profile", "execution_mode"),
    (
        ("fast", tool.FAST_ARM, "fast", "compiled"),
        ("fast_eager", tool.FAST_EAGER_ARM, "fast", "eager"),
        ("balanced", tool.BALANCED_ARM, "balanced", "eager"),
    ),
)
def test_candidate_profile_selects_exactly_one_pair(
    profile, candidate, model_profile, execution_mode
) -> None:
    arms = tool.selected_arms(profile)

    assert arms == (tool.NATIVE_ARM, candidate)
    assert tool.selected_candidate(profile) == candidate
    assert tool.candidate_model_profile(candidate) == model_profile
    assert tool.candidate_execution_mode(candidate) == execution_mode
    assert set(tool._arms_plan(arms)) == {tool.NATIVE_ARM, candidate}


def test_schedule_rejects_an_unregistered_candidate_arm() -> None:
    with pytest.raises(tool.ProtocolError, match="candidate_arm"):
        tool.build_alternating_schedule(
            iterations_per_arm=60,
            phase="measured",
            candidate_arm="rosa_unknown",
            batch_count=2,
        )


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
                "2026/08/02 20:09:39.000, 0, "
                + canonical
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


def test_logical_device_identity_accepts_torch_cuuid_wrapper() -> None:
    TorchCUuuid = type(
        "_CUuuid",
        (),
        {
            "__module__": "torch._C",
            "__str__": lambda self: "E79E0304-D240-431B-F05F-7F3A67933B97",
        },
    )

    class FakeCuda:
        @staticmethod
        def get_device_properties(device):
            del device
            return SimpleNamespace(uuid=TorchCUuuid(), name="NVIDIA GPU")

    device = SimpleNamespace(index=0)
    fake_torch = SimpleNamespace(
        cuda=FakeCuda(), _C=SimpleNamespace(_CUuuid=TorchCUuuid)
    )
    identity = tool._logical_device_identity(fake_torch, device)

    assert identity["gpu_uuid"] == "GPU-e79e0304-d240-431b-f05f-7f3a67933b97"


def test_logical_device_identity_rejects_arbitrary_stringable_uuid() -> None:
    class StringableUuid:
        def __str__(self):
            return "e79e0304-d240-431b-f05f-7f3a67933b97"

    class FakeCuda:
        @staticmethod
        def get_device_properties(device):
            del device
            return SimpleNamespace(uuid=StringableUuid(), name="NVIDIA GPU")

    with pytest.raises(tool.ProtocolError, match="did not expose a GPU UUID"):
        tool._logical_device_identity(
            SimpleNamespace(cuda=FakeCuda()), SimpleNamespace(index=0)
        )


def test_logical_device_identity_rejects_same_named_cuuid_impostor() -> None:
    ExpectedTorchCUuuid = type("_CUuuid", (), {"__module__": "torch._C"})
    ImpostorTorchCUuuid = type(
        "_CUuuid",
        (),
        {
            "__module__": "torch._C",
            "__str__": lambda self: "e79e0304-d240-431b-f05f-7f3a67933b97",
        },
    )

    class FakeCuda:
        @staticmethod
        def get_device_properties(device):
            del device
            return SimpleNamespace(uuid=ImpostorTorchCUuuid(), name="NVIDIA GPU")

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(), _C=SimpleNamespace(_CUuuid=ExpectedTorchCUuuid)
    )
    with pytest.raises(tool.ProtocolError, match="did not expose a GPU UUID"):
        tool._logical_device_identity(fake_torch, SimpleNamespace(index=0))


def test_shared_native_state_projection_preserves_ordered_mapping_type() -> None:
    from collections import OrderedDict

    native = OrderedDict((("backbone.weight", object()), ("head.weight", object())))
    candidate_values = {
        "head.weight": object(),
        "rosa.weight": object(),
        "backbone.weight": object(),
    }

    projected = tool._project_shared_native_state(native, candidate_values)

    assert type(projected) is OrderedDict
    assert tuple(projected) == tuple(native)
    assert all(projected[name] is candidate_values[name] for name in native)


def test_shared_native_state_projection_rejects_missing_key() -> None:
    with pytest.raises(tool.ProtocolError, match="misses native state key"):
        tool._project_shared_native_state({"weight": object()}, {})


def test_allocator_stabilization_clears_cublas_workspace_before_cache(
    monkeypatch,
) -> None:
    calls = []

    class FakeCuda:
        @staticmethod
        def synchronize(device):
            calls.append(("synchronize", device))

        @staticmethod
        def empty_cache():
            calls.append(("empty_cache", None))

        @staticmethod
        def memory_allocated(device):
            calls.append(("memory_allocated", device))
            return 0

        @staticmethod
        def memory_reserved(device):
            calls.append(("memory_reserved", device))
            return 0

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        _C=SimpleNamespace(
            _cuda_clearCublasWorkspaces=lambda: calls.append(("clear_cublas", None))
        ),
    )
    monkeypatch.setattr(tool.gc, "collect", lambda: calls.append(("gc", None)))

    assert tool._stabilize_cuda_allocator(fake_torch, "cuda:0") == {
        "allocated_bytes": 0,
        "reserved_bytes": 0,
    }
    assert calls[:5] == [
        ("synchronize", "cuda:0"),
        ("gc", None),
        ("clear_cublas", None),
        ("empty_cache", None),
        ("synchronize", "cuda:0"),
    ]


def test_allocator_stabilization_requires_documented_cublas_cleanup() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: None), _C=SimpleNamespace()
    )

    with pytest.raises(tool.ProtocolError, match="cuBLAS cleanup API"):
        tool._stabilize_cuda_allocator(fake_torch, "cuda:0")


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


def test_fast_eager_compile_state_attestation_is_fail_closed() -> None:
    refiner = SimpleNamespace(_compiled_call_impl=None)
    model = SimpleNamespace(head=SimpleNamespace(rosa_refiner=refiner))

    record = tool._attest_candidate_compile_state(
        model,
        tool.FAST_EAGER_ARM,
        expected_compiled=False,
        phase="before timing",
    )
    assert record["model_profile"] == "fast"
    assert record["execution_mode"] == "eager"
    assert record["observed_compiled"] is False

    refiner._compiled_call_impl = object()
    with pytest.raises(tool.ProtocolError, match="must be uncompiled"):
        tool._attest_candidate_compile_state(
            model,
            tool.FAST_EAGER_ARM,
            expected_compiled=False,
            phase="after timing",
        )
    with pytest.raises(tool.ProtocolError, match="forbids compiled"):
        tool._attest_candidate_compile_state(
            model,
            tool.FAST_EAGER_ARM,
            expected_compiled=True,
            phase="invalid contract",
        )


def test_compile_state_attestation_rejects_missing_or_wrong_fast_state() -> None:
    missing = SimpleNamespace(head=SimpleNamespace(rosa_refiner=SimpleNamespace()))
    with pytest.raises(tool.ProtocolError, match="does not expose"):
        tool._attest_candidate_compile_state(
            missing,
            tool.FAST_EAGER_ARM,
            expected_compiled=False,
            phase="missing attribute",
        )

    uncompiled = SimpleNamespace(
        head=SimpleNamespace(rosa_refiner=SimpleNamespace(_compiled_call_impl=None))
    )
    with pytest.raises(tool.ProtocolError, match="must be compiled"):
        tool._attest_candidate_compile_state(
            uncompiled,
            tool.FAST_ARM,
            expected_compiled=True,
            phase="after compilation",
        )


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
    (
        (tool.FAST_ARM, 1),
        (tool.FAST_EAGER_ARM, 0),
        (tool.BALANCED_ARM, 0),
    ),
)
def test_memory_measurement_builds_only_one_model_at_a_time(
    monkeypatch, candidate, expected_compiles
) -> None:
    build_calls = []
    compile_calls = []
    reset_calls = []
    lifecycle = []

    class Model:
        def __init__(self):
            self.head = SimpleNamespace(
                rosa_refiner=SimpleNamespace(_compiled_call_impl=None)
            )

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
            model.model.head.rosa_refiner._compiled_call_impl = object()
            return True

    monkeypatch.setattr(tool, "_build_models", build_models)
    monkeypatch.setattr(
        tool,
        "_preflight_outputs",
        lambda verifier, torch, models, source, device: {
            arm: {"raw_fingerprint": arm} for arm in models
        },
    )

    def stabilize(torch, device):
        del torch, device
        lifecycle.append("stabilize")
        return {"allocated_bytes": 0, "reserved_bytes": 0}

    monkeypatch.setattr(tool, "_stabilize_cuda_allocator", stabilize)
    monkeypatch.setattr(tool.gc, "collect", lambda: lifecycle.append("gc"))
    monkeypatch.setattr(
        tool,
        "_reset_compiler_cache",
        lambda torch: (reset_calls.append(torch), lifecycle.append("reset")),
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
    expected_lifecycle = [
        "stabilize",
        "stabilize",
        "gc",
        "stabilize",
        "stabilize",
        "gc",
    ]
    if expected_compiles:
        expected_lifecycle.append("reset")
    expected_lifecycle.append("stabilize")
    assert lifecycle == expected_lifecycle
    assert set(contracts) == set(loads) == {tool.NATIVE_ARM, candidate}
    assert memory["contract"]["one_benchmark_model_resident_per_probe"] is True
    assert memory["contract"]["between_arm_cleanup"] == [
        {
            "phase": "model_release",
            "steps": ["models.clear", "gc.collect"],
        },
        {
            "phase": "compiled_arm_only",
            "steps": ["torch._dynamo.reset"],
        },
        {
            "phase": "allocator_stabilization",
            "steps": [
                "torch.cuda.synchronize",
                "gc.collect",
                "torch._C._cuda_clearCublasWorkspaces",
                "torch.cuda.empty_cache",
                "torch.cuda.synchronize",
            ],
        },
    ]
    for arm in (tool.NATIVE_ARM, candidate):
        assert memory[arm]["resident_benchmark_models"] == [arm]
        assert memory[arm]["peer_model_resident"] is False
        assert len(memory[arm]["probe_records"]) == 3
    assert memory[candidate]["execution_mode"] == tool.candidate_execution_mode(
        candidate
    )
    assert all(
        record["passed"] for record in memory[candidate]["compile_state_attestations"]
    )
    assert all(
        record["observed_compiled"] is (candidate == tool.FAST_ARM)
        for record in memory[candidate]["compile_state_attestations"][1:]
    )


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


@pytest.mark.parametrize(
    ("candidate", "reason"),
    (
        (tool.FAST_EAGER_ARM, "uncompiled fast reference"),
        (tool.BALANCED_ARM, "balanced candidate"),
    ),
)
def test_eager_timing_does_not_require_compiler_counters(candidate, reason) -> None:
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
        {tool.NATIVE_ARM: object(), candidate: object()},
        [],
        [],
        "cuda:0",
        candidate,
    )
    assert result["compiler_guard"]["applicable"] is False
    assert reason in result["compiler_guard"]["reason"]
    assert result["candidate_execution_mode"] == "eager"


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


@pytest.mark.parametrize("arm", (tool.FAST_ARM, tool.FAST_EAGER_ARM))
def test_checkpoint_semantics_allows_only_the_preregistered_fast_budget_delta(
    arm,
) -> None:
    record = tool._validate_checkpoint_semantics(
        _SemanticRunTool,
        _semantic_checkpoint(),
        object(),
        arm,
    )

    assert record["validated_against_arm"] == arm
    assert record["source_refinement_profile"] == "balanced"
    assert record["allowed_runtime_delta"] == tool.FAST_CHECKPOINT_ALLOWED_DELTAS
    assert record["observed_runtime_delta"] == tool.FAST_CHECKPOINT_ALLOWED_DELTAS
    assert record["runtime_model_profile"] == "fast"
    assert record["runtime_execution_mode"] == tool.candidate_execution_mode(arm)
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
@pytest.mark.parametrize("arm", (tool.FAST_ARM, tool.FAST_EAGER_ARM))
def test_checkpoint_semantics_fails_closed_on_missing_or_unrelated_metadata(
    mutation, message, arm
) -> None:
    checkpoint = _semantic_checkpoint()
    mutation(checkpoint)
    with pytest.raises(tool.ProtocolError, match=message):
        tool._validate_checkpoint_semantics(_SemanticRunTool, checkpoint, object(), arm)


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


def test_fast_eager_derives_the_exact_fast_model_config_without_name_leak() -> None:
    native = {
        "model": {
            "backbone": {"depth": 18},
            "head": {"type": "ROSANativeBaselineHead", "rosa_cfg": None},
        },
        "seed": 3407,
    }
    arms = tool._arms_plan(tool.selected_arms("fast_eager"))
    derivation = arms[tool.FAST_EAGER_ARM]["config_derivation"]["only_changes"]
    expected = tool._expected_candidate_config(
        native, tool.candidate_model_profile(tool.FAST_EAGER_ARM)
    )

    assert tool.candidate_model_profile(tool.FAST_EAGER_ARM) == "fast"
    assert derivation["model.head.rosa_cfg"] == {"profile": "fast"}
    assert "fast_eager" not in json.dumps(derivation, sort_keys=True)
    assert expected["model"]["head"] == {
        "type": "ROSARefinementHead",
        "rosa_cfg": {"profile": "fast"},
    }


def test_derive_configs_materializes_fast_eager_as_fast() -> None:
    class FakeConfig:
        def __init__(self):
            self.model = SimpleNamespace(
                backbone=SimpleNamespace(depth=18),
                head=SimpleNamespace(type="ROSANativeBaselineHead", rosa_cfg=None),
            )
            self.seed = 3407

        def to_dict(self):
            return {
                "model": {
                    "backbone": {"depth": self.model.backbone.depth},
                    "head": {
                        "type": self.model.head.type,
                        "rosa_cfg": self.model.head.rosa_cfg,
                    },
                },
                "seed": self.seed,
            }

    verifier = SimpleNamespace(
        Config=SimpleNamespace(
            fromfile=lambda path, import_custom_modules: FakeConfig()
        )
    )
    configs = tool._derive_configs(
        verifier,
        Path("unused-native-config.py"),
        tool.selected_arms("fast_eager"),
    )

    assert configs[tool.NATIVE_ARM].model.head.type == "ROSANativeBaselineHead"
    assert configs[tool.FAST_EAGER_ARM].model.head.type == "ROSARefinementHead"
    assert configs[tool.FAST_EAGER_ARM].model.head.rosa_cfg == {"profile": "fast"}


def test_native_checkpoint_filter_accepts_only_rosa_sidecar_state() -> None:
    tool._validate_native_checkpoint_exclusions(
        ["head.rosa_refiner.confidence.weight", "head.rosa_refiner.yaw.bias"]
    )
    with pytest.raises(tool.ProtocolError, match="non-ROSA"):
        tool._validate_native_checkpoint_exclusions(
            ["head.rosa_refiner.yaw.bias", "head.decoder.weight"]
        )


@pytest.mark.parametrize("profile", ("fast", "fast_eager", "balanced"))
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
    arms = tool._arms_plan(tool.selected_arms("fast_eager"))
    encoded = json.dumps(arms, sort_keys=True)
    assert "ROSARefinementHead" in encoded
    assert '"model_profile": "fast"' in encoded
    assert '"execution_mode": "eager"' in encoded
    assert set(arms) == {tool.NATIVE_ARM, tool.FAST_EAGER_ARM}
