"""CPU-only contracts for the disarmed full-ROSA speed harness."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

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
    calibration = temp / 'repeatability.json'
    calibration.write_bytes(b'calibration')
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
    module.DEFAULT_C32_NATIVE_CONFIG = config
    module.DEFAULT_MAIN126_ANNOTATION = annotation
    module.EXPECTED_MAIN126_ANNOTATION_SHA256 = module.file_sha256(annotation)
    module.DEFAULT_COMPILED_BOUNDED_CALIBRATION_ARTIFACT = calibration
    module.EXPECTED_COMPILED_BOUNDED_CALIBRATION_SHA256 = (
        module.file_sha256(calibration)
    )
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
        ('fast_compiled_bounded', module.FAST_COMPILED_BOUNDED_ARM),
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
        assert plan['detector_pair'] == 'c0'
        assert plan['detector_pair_contract']['native_head_type'] == (
            'ROSANativeBaselineHead'
        )
        assert plan['detector_pair_contract']['candidate_head_type'] == (
            'ROSARefinementHead'
        )
        assert plan['candidate_profile'] == profile
        assert plan['schema'].endswith('.v4')
        assert plan['protocol'].endswith('-v4')
        compiler_guard = plan['measurement']['compiler_guard']
        if profile in ('fast', 'fast_compiled_bounded'):
            assert compiler_guard['applicable'] is True
            assert 'raw-exact' in compiler_guard['requirement']
        else:
            assert compiler_guard['applicable'] is False
            assert 'reason' in compiler_guard
        if profile == 'fast':
            assert plan['measurement']['candidate_compile']['kwargs'] == {
                'mode': 'reduce-overhead',
                'fullgraph': False,
            }
        if profile == 'fast_eager':
            assert plan['candidate_model_profile'] == 'fast'
            assert plan['candidate_execution_mode'] == 'eager'
            assert plan['arms'][candidate]['compile'] is False
            assert plan['arms'][candidate]['compile_forbidden'] is True
            assert plan['arms'][candidate]['config_derivation']['only_changes'][
                'model.head.rosa_cfg'
            ] == {'profile': 'fast'}
        if profile == 'fast_compiled_bounded':
            parity = plan['output_contract']['compiled_eager_parity']
            compile_contract = plan['measurement']['candidate_compile']
            assert compile_contract['route'] == (
                'benchmark-local head.rosa_refiner.compile'
            )
            assert compile_contract['kwargs'] == {
                'mode': 'max-autotune-no-cudagraphs',
                'fullgraph': False,
            }
            assert parity['mode'] == (
                'query_aligned_fixed_absolute_tolerance_v3'
            )
            assert parity['alignment_key'] == 'query_indices'
            assert parity['exact_fields_after_alignment'] == [
                'labels_3d',
                'instance_ids',
            ]
            assert parity['bounded_float_fields'] == [
                'scores_3d',
                'cls_scores',
                'bboxes_3d',
            ]
            assert parity['absolute_thresholds'] == {
                'scores_3d': 3.5e-6,
                'cls_scores': 4e-6,
                'bboxes_3d': 0.003,
            }
            assert parity['query_contract'] == {
                'maximum_count': 300,
                'count_matches_decoder_output': True,
                'unique': True,
                'inclusive_range': [0, 299],
                'set_exact_across_all_three_comparisons': True,
            }
            assert parity['rank_inversion_policy'] == {
                'maximum_reference_score_gap': 7e-6,
                'larger_gap_inversion_forbidden': True,
            }
            assert parity['required_comparisons'] == [
                'eager_reference_a_vs_eager_reference_b',
                'eager_reference_a_vs_compiled',
                'eager_reference_b_vs_compiled',
            ]
            assert parity['threshold_basis']['fixed_calibration_screen'] == (
                module.COMPILED_BOUNDED_CALIBRATION_SCREEN
            )
            assert parity['threshold_basis'][
                'dynamic_candidate_calibration_forbidden'
            ] is True
            calibration_contract = plan['compiled_parity_calibration']
            assert calibration_contract['applicable'] is True
            assert calibration_contract['artifact']['path'] == str(calibration)
            assert calibration_contract['artifact']['sha256'] == (
                module.file_sha256(calibration)
            )
            assert plan['promotion_contract'][
                'official_full_metric_parity_required_before_promotion'
            ] is True
        assert module.main(argv) == 0
        assert not output.exists()

    c32_output = temp / 'not-created-c32-fast-eager'
    c32_argv = [
        '--checkpoint', str(checkpoint),
        '--expected-checkpoint-sha256', module.file_sha256(checkpoint),
        '--expected-source-revision', source_revision,
        '--detector-pair', 'c32',
        '--detector-host-root', str(detector),
        '--expected-detector-host-revision', host_revision,
        '--official-root', str(official),
        '--expected-official-revision', official_revision,
        '--expected-daf-extension-sha256', module.file_sha256(daf),
        '--output-dir', str(c32_output),
        '--candidate-profile', 'fast_eager',
    ]
    c32_args = module.build_parser().parse_args(c32_argv)
    c32_plan = module.prepare_plan(c32_args)
    assert set(c32_plan['arms']) == {
        module.C32_NATIVE_ARM,
        module.FAST_EAGER_ARM,
    }
    assert c32_plan['detector_pair'] == 'c32'
    assert c32_plan['protocol'] == module.C32_PROTOCOL
    assert c32_plan['native_config']['path'] == str(config.resolve())
    assert c32_plan['detector_pair_contract']['native_head_type'] == (
        'ROSAOfficialResidualNativeBaselineHead'
    )
    assert c32_plan['detector_pair_contract']['candidate_head_type'] == (
        'ROSAOfficialResidualRefinementHead'
    )
    c32_derivation = c32_plan['arms'][module.FAST_EAGER_ARM][
        'config_derivation'
    ]
    assert c32_derivation['base'] == module.C32_NATIVE_ARM
    assert c32_derivation['only_changes'] == {
        'model.head.type': 'ROSAOfficialResidualRefinementHead',
        'model.head.rosa_cfg': {'profile': 'fast'},
    }
    assert module.main(c32_argv) == 0
    assert not c32_output.exists()

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
        (tool.FAST_COMPILED_BOUNDED_ARM, 60),
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
        (
            "fast_compiled_bounded",
            tool.FAST_COMPILED_BOUNDED_ARM,
            "fast",
            "compiled",
        ),
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


def test_c32_pair_uses_explicit_native_arm_and_exact_head_labels() -> None:
    arms = tool.selected_arms("fast_eager", "c32")
    records = tool._arms_plan(arms, "c32")
    schedule = tool.build_alternating_schedule(
        iterations_per_arm=60,
        phase="measured",
        candidate_arm=tool.FAST_EAGER_ARM,
        batch_count=2,
        detector_pair="c32",
    )

    assert arms == (tool.C32_NATIVE_ARM, tool.FAST_EAGER_ARM)
    assert records[tool.C32_NATIVE_ARM]["head_type"] == (
        "ROSAOfficialResidualNativeBaselineHead"
    )
    assert records[tool.FAST_EAGER_ARM]["head_type"] == (
        "ROSAOfficialResidualRefinementHead"
    )
    assert records[tool.FAST_EAGER_ARM]["config_derivation"]["only_changes"] == {
        "model.head.type": "ROSAOfficialResidualRefinementHead",
        "model.head.rosa_cfg": {"profile": "fast"},
    }
    assert {row["arm"] for row in schedule} == {
        tool.C32_NATIVE_ARM,
        tool.FAST_EAGER_ARM,
    }
    assert sum(row["arm"] == tool.C32_NATIVE_ARM for row in schedule) == 60


@pytest.mark.parametrize("detector_pair", ("C32", "official", ""))
def test_detector_pair_selection_fails_closed(detector_pair) -> None:
    with pytest.raises(tool.ProtocolError, match="detector pair"):
        tool.detector_pair_contract(detector_pair)


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


def test_inductor_cleanup_releases_only_new_compile_task_modules(
    monkeypatch,
) -> None:
    prefix = tool.INDUCTOR_COMPILE_TASK_MODULE_PREFIX
    baseline_name = prefix + "same_hash_test_module"
    created_name = baseline_name
    baseline_module = ModuleType(baseline_name)
    alternate_baseline_module = ModuleType(baseline_name)
    created_module = ModuleType(created_name)

    class FakePyCodeCache:
        modules = [baseline_module, alternate_baseline_module]
        modules_no_attr = {"/baseline.py": baseline_module}

    codecache_module = ModuleType("torch._inductor.codecache")
    codecache_module.PyCodeCache = FakePyCodeCache
    monkeypatch.setitem(sys.modules, "torch._inductor.codecache", codecache_module)
    monkeypatch.setitem(sys.modules, baseline_name, baseline_module)
    baseline_snapshot = tool._inductor_compile_task_module_snapshot()
    FakePyCodeCache.modules.extend((created_module, created_module))
    FakePyCodeCache.modules_no_attr["/created.py"] = created_module
    monkeypatch.setitem(sys.modules, created_name, created_module)
    observed_snapshot = tool._inductor_compile_task_module_snapshot()

    record = tool._release_new_inductor_compile_modules(
        baseline_snapshot,
        observed_snapshot,
    )

    assert sys.modules[baseline_name] is baseline_module
    assert FakePyCodeCache.modules == [baseline_module, alternate_baseline_module]
    assert FakePyCodeCache.modules_no_attr == {"/baseline.py": baseline_module}
    assert record["created_count"] == 1
    assert record["removed_module_names"] == [created_name]
    assert record["tracked_list_references_removed"] == 2
    assert record["tracked_dict_entries_removed"] == 1
    assert record["sys_module_entries_removed"] == 1
    assert record["baseline_sys_module_entries_restored"] == 1
    assert record["passed"] is True


def test_inductor_module_snapshot_rejects_malformed_registry(monkeypatch) -> None:
    class FakePyCodeCache:
        modules = [object()]
        modules_no_attr = {}

    codecache_module = ModuleType("torch._inductor.codecache")
    codecache_module.PyCodeCache = FakePyCodeCache
    monkeypatch.setitem(sys.modules, "torch._inductor.codecache", codecache_module)

    with pytest.raises(tool.ProtocolError, match="registry is malformed"):
        tool._inductor_compile_task_module_snapshot()


def test_inductor_cleanup_rejects_missing_baseline_sys_entry(monkeypatch) -> None:
    entry = (tool.INDUCTOR_COMPILE_TASK_MODULE_PREFIX + "baseline", 1)
    baseline = tool._InductorCompileTaskSnapshot(frozenset(), frozenset({entry}))
    observed = tool._InductorCompileTaskSnapshot(frozenset(), frozenset())
    monkeypatch.setattr(
        tool,
        "_inductor_compile_task_module_snapshot",
        lambda: observed,
    )

    with pytest.raises(tool.ProtocolError, match="pre-existing sys.modules entry"):
        tool._release_new_inductor_compile_modules(baseline, observed)


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


def test_compiled_arms_use_distinct_frozen_compile_routes() -> None:
    class Refiner:
        def __init__(self):
            self._compiled_call_impl = None
            self.calls = []

        def compile(self, **kwargs):
            self.calls.append(kwargs)
            self._compiled_call_impl = object()

    class RunTool:
        calls = []

        @classmethod
        def _compile_fast_inference(cls, runner, config):
            cls.calls.append((runner, config))
            runner.model.head.rosa_refiner.compile(
                mode="reduce-overhead", fullgraph=False
            )
            return True

    raw_refiner = Refiner()
    raw_model = SimpleNamespace(head=SimpleNamespace(rosa_refiner=raw_refiner))
    raw = tool._compile_candidate_refiner(
        RunTool, raw_model, "raw-config", tool.FAST_ARM
    )
    assert len(RunTool.calls) == 1
    assert raw_refiner.calls == [{"mode": "reduce-overhead", "fullgraph": False}]
    assert raw["observed_route"] == "frozen tools.run._compile_fast_inference"
    assert raw["observed_kwargs"] == {
        "mode": "reduce-overhead",
        "fullgraph": False,
    }

    bounded_refiner = Refiner()
    bounded_model = SimpleNamespace(head=SimpleNamespace(rosa_refiner=bounded_refiner))
    bounded = tool._compile_candidate_refiner(
        RunTool,
        bounded_model,
        "bounded-config",
        tool.FAST_COMPILED_BOUNDED_ARM,
    )
    assert len(RunTool.calls) == 1
    assert bounded_refiner.calls == [
        {"mode": "max-autotune-no-cudagraphs", "fullgraph": False}
    ]
    assert bounded["observed_route"] == ("benchmark-local head.rosa_refiner.compile")
    assert bounded["observed_kwargs"] == {
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": False,
    }
    assert (
        tool.candidate_compile_contract(tool.FAST_COMPILED_BOUNDED_ARM)[
            "production_fast_route_unchanged"
        ]
        is True
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
        (tool.FAST_COMPILED_BOUNDED_ARM, 1),
        (tool.FAST_EAGER_ARM, 0),
        (tool.BALANCED_ARM, 0),
    ),
)
@pytest.mark.parametrize("native_arm", (tool.NATIVE_ARM, tool.C32_NATIVE_ARM))
def test_memory_measurement_builds_only_one_model_at_a_time(
    monkeypatch, candidate, expected_compiles, native_arm
) -> None:
    build_calls = []
    production_compile_calls = []
    benchmark_compile_calls = []
    reset_calls = []
    lifecycle = []

    class Refiner:
        def __init__(self):
            self._compiled_call_impl = None

        def compile(self, **kwargs):
            benchmark_compile_calls.append(kwargs)
            self._compiled_call_impl = object()

    class Model:
        def __init__(self):
            self.head = SimpleNamespace(rosa_refiner=Refiner())

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
            production_compile_calls.append((model, config))
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
    empty_snapshot = tool._InductorCompileTaskSnapshot(frozenset(), frozenset())
    created_snapshot = tool._InductorCompileTaskSnapshot(
        frozenset({1}),
        frozenset({(tool.INDUCTOR_COMPILE_TASK_MODULE_PREFIX + "candidate", 1)}),
    )
    module_snapshots = iter(
        (
            empty_snapshot,
            created_snapshot,
        )
    )
    monkeypatch.setattr(
        tool,
        "_inductor_compile_task_module_snapshot",
        lambda: next(module_snapshots),
    )
    monkeypatch.setattr(
        tool,
        "_release_new_inductor_compile_modules",
        lambda baseline, observed: (
            lifecycle.append("release"),
            {
                "baseline_count": len(baseline.cache_ids),
                "created_count": len(observed.cache_ids - baseline.cache_ids),
                "passed": True,
            },
        )[1],
    )

    configs = {native_arm: object(), candidate: object()}
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
        native_arm=native_arm,
    )

    assert build_calls == [(native_arm,), (candidate,)]
    assert len(production_compile_calls) == (1 if candidate == tool.FAST_ARM else 0)
    assert benchmark_compile_calls == (
        [{"mode": "max-autotune-no-cudagraphs", "fullgraph": False}]
        if candidate == tool.FAST_COMPILED_BOUNDED_ARM
        else []
    )
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
        expected_lifecycle.extend(("reset", "release", "gc"))
    expected_lifecycle.append("stabilize")
    assert lifecycle == expected_lifecycle
    assert set(contracts) == set(loads) == {native_arm, candidate}
    assert memory["contract"]["one_benchmark_model_resident_per_probe"] is True
    assert memory["contract"]["between_arm_cleanup"] == [
        {
            "phase": "model_release",
            "steps": ["models.clear", "gc.collect"],
        },
        {
            "phase": "compiled_arm_only",
            "steps": [
                "torch._dynamo.reset",
                "release newly created torch._inductor compile-task modules",
                "gc.collect",
            ],
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
    for arm in (native_arm, candidate):
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
        record["observed_compiled"] is (candidate in tool.COMPILED_ARMS)
        for record in memory[candidate]["compile_state_attestations"][1:]
    )
    invocation = memory[candidate]["compile_invocation_attestation"]
    if candidate in tool.COMPILED_ARMS:
        assert memory[candidate]["compiler_cache_cleanup_attestation"] == {
            "baseline_count": 0,
            "created_count": 1,
            "passed": True,
        }
        assert invocation["passed"] is True
        assert (
            invocation["observed_kwargs"]
            == tool.candidate_compile_contract(candidate)["kwargs"]
        )
    else:
        assert memory[candidate]["compiler_cache_cleanup_attestation"] is None
        assert invocation is None


def test_memory_cleanup_runs_when_compilation_raises(monkeypatch) -> None:
    events = []

    class Model:
        head = SimpleNamespace(rosa_refiner=SimpleNamespace(_compiled_call_impl=None))

    def build_models(verifier, configs, state_dict, arms, device):
        del verifier, configs, state_dict, device
        arm = tuple(arms)[0]
        return {arm: Model()}, {arm: {}}, {arm: {}}

    class Verifier:
        @staticmethod
        def _memory_probe(model, source, device):
            del model, source, device
            return {
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 1,
                "peak_allocated_increment_bytes": 1,
                "peak_reserved_increment_bytes": 1,
            }

    class Cuda:
        current_stream = staticmethod(lambda device: SimpleNamespace(cuda_stream=7))
        default_stream = staticmethod(lambda device: SimpleNamespace(cuda_stream=7))

    empty = tool._InductorCompileTaskSnapshot(frozenset(), frozenset())
    partial = tool._InductorCompileTaskSnapshot(
        frozenset({1}),
        frozenset({(tool.INDUCTOR_COMPILE_TASK_MODULE_PREFIX + "partial", 1)}),
    )
    snapshots = iter((empty, partial))

    def compile_raises(*args):
        del args
        events.append("compile")
        raise RuntimeError("compile failed")

    monkeypatch.setattr(tool, "_build_models", build_models)
    monkeypatch.setattr(
        tool,
        "_preflight_outputs",
        lambda verifier, torch, models, source, device: {arm: {} for arm in models},
    )
    monkeypatch.setattr(
        tool,
        "_stabilize_cuda_allocator",
        lambda torch, device: {"allocated_bytes": 0, "reserved_bytes": 0},
    )
    monkeypatch.setattr(
        tool,
        "_inductor_compile_task_module_snapshot",
        lambda: (events.append("snapshot"), next(snapshots))[1],
    )
    monkeypatch.setattr(tool, "_compile_candidate_refiner", compile_raises)
    monkeypatch.setattr(
        tool,
        "_reset_compiler_cache",
        lambda torch: events.append("reset"),
    )
    monkeypatch.setattr(
        tool,
        "_release_new_inductor_compile_modules",
        lambda baseline, observed: events.append("release") or {"passed": True},
    )
    monkeypatch.setattr(tool.gc, "collect", lambda: events.append("gc"))

    configs = {tool.NATIVE_ARM: object(), tool.FAST_COMPILED_BOUNDED_ARM: object()}
    with pytest.raises(RuntimeError, match="compile failed"):
        tool._measure_isolated_memory(
            Verifier,
            SimpleNamespace(cuda=Cuda()),
            object(),
            configs,
            {},
            [object()],
            "cuda:0",
            tool.FAST_COMPILED_BOUNDED_ARM,
        )

    assert events == [
        "gc",
        "snapshot",
        "compile",
        "snapshot",
        "gc",
        "reset",
        "release",
        "gc",
    ]


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
@pytest.mark.parametrize("native_arm", (tool.NATIVE_ARM, tool.C32_NATIVE_ARM))
def test_eager_timing_does_not_require_compiler_counters(
    candidate, reason, native_arm
) -> None:
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
        {native_arm: object(), candidate: object()},
        [],
        [],
        "cuda:0",
        candidate,
        native_arm,
    )
    assert result["compiler_guard"]["applicable"] is False
    assert reason in result["compiler_guard"]["reason"]
    assert result["candidate_execution_mode"] == "eager"
    assert result["detector_pair"] == tool._detector_pair_from_native_arm(native_arm)
    assert result["arm_aliases"]["parent"] == native_arm


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

    @property
    def rfl_topology_contract(self):
        return "rosa.rfl.signed_gated_cp16.v3"


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
            "rosa_contract_version": 3,
            "rfl_topology_contract": "rosa.rfl.signed_gated_cp16.v3",
            "projection_contract": "full_fov",
        },
        "state_dict": {"head.rosa_refiner.weight": "sealed"},
    }


@pytest.mark.parametrize(
    "arm",
    (
        tool.FAST_ARM,
        tool.FAST_COMPILED_BOUNDED_ARM,
        tool.FAST_EAGER_ARM,
    ),
)
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
    assert record["rosa_contract_version"] == 3
    assert record["rfl_topology_contract"] == ("rosa.rfl.signed_gated_cp16.v3")
    assert record["allowed_runtime_delta"] == tool.FAST_CHECKPOINT_ALLOWED_DELTAS
    assert record["observed_runtime_delta"] == tool.FAST_CHECKPOINT_ALLOWED_DELTAS
    assert record["runtime_model_profile"] == "fast"
    assert record["runtime_execution_mode"] == tool.candidate_execution_mode(arm)
    assert record["all_other_effective_fields_exact"] is True


def _c32_semantic_config(*, head_type=None, profile="fast"):
    return SimpleNamespace(
        model=SimpleNamespace(
            head=SimpleNamespace(
                type=(
                    "ROSAOfficialResidualRefinementHead"
                    if head_type is None
                    else head_type
                ),
                rosa_cfg={"profile": profile},
            )
        )
    )


def test_c32_checkpoint_semantics_records_exact_detector_pair() -> None:
    record = tool._validate_checkpoint_semantics(
        _SemanticRunTool,
        _semantic_checkpoint(),
        _c32_semantic_config(),
        tool.FAST_EAGER_ARM,
        detector_pair="c32",
    )

    assert record["detector_pair"] == "c32"
    assert record["candidate_head_type"] == ("ROSAOfficialResidualRefinementHead")
    assert record["checkpoint_has_rosa_state"] is True
    assert record["all_other_effective_fields_exact"] is True


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (
            _c32_semantic_config(head_type="ROSARefinementHead"),
            "exact C32 ROSA candidate head",
        ),
        (_c32_semantic_config(profile="balanced"), "exact ROSA profile selector"),
    ),
)
def test_c32_checkpoint_semantics_rejects_wrong_head_or_profile(
    config, message
) -> None:
    with pytest.raises(tool.ProtocolError, match=message):
        tool._validate_checkpoint_semantics(
            _SemanticRunTool,
            _semantic_checkpoint(),
            config,
            tool.FAST_EAGER_ARM,
            detector_pair="c32",
        )


def test_c32_checkpoint_semantics_rejects_missing_sidecar_state() -> None:
    class MissingSidecarRunTool(_SemanticRunTool):
        @staticmethod
        def _state_dict_has_rosa(state_dict):
            del state_dict
            return False

    with pytest.raises(tool.ProtocolError, match="must contain the ROSA sidecar state"):
        tool._validate_checkpoint_semantics(
            MissingSidecarRunTool,
            _semantic_checkpoint(),
            _c32_semantic_config(),
            tool.FAST_EAGER_ARM,
            detector_pair="c32",
        )


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
        (
            lambda checkpoint: checkpoint["meta"].update(rosa_contract_version=2),
            "contract version",
        ),
        (
            lambda checkpoint: checkpoint["meta"].pop("rfl_topology_contract"),
            "RFL topology contract",
        ),
    ),
)
@pytest.mark.parametrize(
    "arm",
    (
        tool.FAST_ARM,
        tool.FAST_COMPILED_BOUNDED_ARM,
        tool.FAST_EAGER_ARM,
    ),
)
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


def test_original_fast_raw_exact_gate_does_not_accept_bounded_box_delta() -> None:
    eager = {
        "0": {"raw_fingerprint": "eager-box-bits"},
        "1": {"raw_fingerprint": "same"},
    }
    compiled = {
        "0": {"raw_fingerprint": "compiled-box-bits-with-1e-6-delta"},
        "1": {"raw_fingerprint": "same"},
    }

    with pytest.raises(tool.ProtocolError, match="not raw-exact"):
        tool._require_all_batch_compiled_parity(eager, compiled, batch_count=2)
    assert tool.compiled_parity_contract(tool.FAST_ARM) == {
        "applicable": True,
        "mode": "raw_exact_all_output_fields",
        "all_output_values_raw_exact": True,
    }


def _test_raw_fingerprint(values):
    return [
        (
            str(value.dtype),
            tuple(value.shape),
            tuple(value.stride()),
            bytes(value.contiguous().view(torch.uint8).reshape(-1).tolist()),
        )
        for value in values
    ]


def _bounded_parity_fixture(*, box_dtype=torch.float32, count=50):
    eager_records, compiled_records, eager_payloads, compiled_payloads = {}, {}, {}, {}
    # A prefix of this permutation stays unique and inside the real decoder's
    # 300-query domain, including the full 300-output production case.
    query_indices = (torch.arange(count) * 299) % 300
    scores = torch.linspace(0.8, 0.3, count)
    if count >= 2:
        scores[:2] = torch.tensor([0.900000, 0.899996])
    for batch_index in range(2):
        key = str(batch_index)
        record = {
            "sample_count": 1,
            "detections_per_sample": [count],
            "finite_schema": {"sealed": True},
            "stream": {"current_stream": 7, "default_stream": 7},
        }
        eager_records[key], compiled_records[key] = record, dict(record)
        eager_sample = {
            "bboxes_3d": torch.arange(count * 10, dtype=box_dtype).reshape(count, 10)
            / 10
            + batch_index,
            "scores_3d": scores.clone(),
            "cls_scores": scores.clone() + 0.02,
            "labels_3d": torch.arange(count) % 6,
            "query_indices": query_indices.clone(),
            "instance_ids": query_indices.clone() + 1000,
        }
        compiled_sample = {name: value.clone() for name, value in eager_sample.items()}
        compiled_sample["bboxes_3d"].add_(1.0e-3)
        compiled_sample["cls_scores"].add_(
            tool.COMPILED_BOUNDED_SCORE_MAX_ABS["cls_scores"] / 4
        )
        near_tie_swap = torch.arange(count)
        if count >= 2:
            compiled_sample["scores_3d"][0].sub_(3.0e-6)
            compiled_sample["scores_3d"][1].add_(3.0e-6)
            near_tie_swap[:2] = torch.tensor([1, 0])
        compiled_sample = {
            name: value[near_tie_swap].clone()
            for name, value in compiled_sample.items()
        }
        eager_payloads[key], compiled_payloads[key] = (eager_sample,), (
            compiled_sample,
        )
    return (
        SimpleNamespace(raw_fingerprint=_test_raw_fingerprint),
        eager_records,
        compiled_records,
        eager_payloads,
        compiled_payloads,
    )


def test_compiled_bounded_parity_is_explicit_and_records_error_distribution() -> None:
    fixture = _bounded_parity_fixture()
    parity = tool._require_all_batch_compiled_bounded_parity(
        fixture[0],
        torch,
        *fixture[1:],
        batch_count=2,
    )

    assert set(parity) == {"0", "1"}
    for record in parity.values():
        assert record["passed"] is True
        assert record["query_alignment"]["query_set_exact"] is True
        assert set(record["exact_fields_after_alignment"]) == set(
            tool.COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS
        )
        assert record["rank_displacement"]["changed_query_count"] == 2
        assert record["rank_displacement"]["max"] == 1.0
        error = record["bounded_float_fields"]["bboxes_3d"]
        assert set(error["absolute_error"]) == {
            "count",
            "max",
            "mean",
            "p99_nearest_rank",
        }
        assert error["absolute_error"]["count"] == 500
        assert error["absolute_error"]["max"] <= 0.003
        assert record["rank_displacement"]["inversion_count"] == 1
        assert (
            record["rank_displacement"]["maximum_inversion_score_gap"]
            <= tool.COMPILED_BOUNDED_NUMERICAL_TIE_MAX_GAP
        )
        for field in tool.COMPILED_BOUNDED_SCORE_FIELDS:
            assert (
                record["bounded_float_fields"][field]["absolute_error"]["max"]
                <= tool.COMPILED_BOUNDED_SCORE_MAX_ABS[field]
            )
        assert record["promotion_status"] == "PENDING_OFFICIAL_FULL_METRIC_PARITY"


def test_compiled_parity_calibration_artifact_is_hash_bound(
    tmp_path, monkeypatch
) -> None:
    artifact = tmp_path / "repeatability.json"
    artifact.write_bytes(b"unexpected")
    monkeypatch.setattr(tool, "DEFAULT_COMPILED_BOUNDED_CALIBRATION_ARTIFACT", artifact)
    monkeypatch.setattr(tool, "EXPECTED_COMPILED_BOUNDED_CALIBRATION_SHA256", "0" * 64)
    with pytest.raises(tool.ProtocolError, match="artifact hash drifted"):
        tool.compiled_parity_calibration_contract(tool.FAST_COMPILED_BOUNDED_ARM)


@pytest.mark.parametrize("count", [0, 7, 300])
def test_compiled_bounded_parity_accepts_variable_decoder_count(count) -> None:
    fixture = _bounded_parity_fixture(count=count)
    parity = tool._require_all_batch_compiled_bounded_parity(
        fixture[0], torch, *fixture[1:], batch_count=2
    )
    assert all(
        record["bounded_float_fields"]["bboxes_3d"]["absolute_error"]["count"]
        == count * tool.COMPILED_BOUNDED_BOX_DIM
        for record in parity.values()
    )


def test_compiled_bounded_parity_rejects_invalid_decoder_count() -> None:
    fixture = _bounded_parity_fixture()
    for records in fixture[1:3]:
        records["0"]["detections_per_sample"] = [49]
    with pytest.raises(tool.ProtocolError, match="does not match the decoder record"):
        tool._require_all_batch_compiled_bounded_parity(
            fixture[0], torch, *fixture[1:], batch_count=2
        )

    over_limit = _bounded_parity_fixture()
    for records in over_limit[1:3]:
        records["0"]["detections_per_sample"] = [301]
    with pytest.raises(tool.ProtocolError, match=r"must be in \[0, 300\]"):
        tool._require_all_batch_compiled_bounded_parity(
            over_limit[0], torch, *over_limit[1:], batch_count=2
        )


@pytest.mark.parametrize("field", tool.COMPILED_BOUNDED_EXACT_ASSOCIATION_FIELDS)
def test_compiled_bounded_parity_rejects_query_aligned_association_change(
    field,
) -> None:
    fixture = _bounded_parity_fixture()
    compiled_payload = fixture[4]
    compiled_payload["1"][0][field][0].add_(1)

    with pytest.raises(tool.ProtocolError, match=f"exact field {field}"):
        tool._require_all_batch_compiled_bounded_parity(
            fixture[0], torch, *fixture[1:], batch_count=2
        )


def test_compiled_bounded_parity_rejects_query_set_or_uniqueness_change() -> None:
    changed = _bounded_parity_fixture()
    changed[4]["0"][0]["query_indices"][2] = 1
    with pytest.raises(tool.ProtocolError, match="changed the query set"):
        tool._require_all_batch_compiled_bounded_parity(
            changed[0], torch, *changed[1:], batch_count=2
        )

    duplicate = _bounded_parity_fixture()
    duplicate[4]["0"][0]["query_indices"][0] = duplicate[4]["0"][0]["query_indices"][1]
    with pytest.raises(tool.ProtocolError, match="must be unique"):
        tool._require_all_batch_compiled_bounded_parity(
            duplicate[0], torch, *duplicate[1:], batch_count=2
        )

    out_of_range = _bounded_parity_fixture()
    out_of_range[4]["0"][0]["query_indices"][0] = 300
    with pytest.raises(tool.ProtocolError, match=r"must be in \[0, 299\]"):
        tool._require_all_batch_compiled_bounded_parity(
            out_of_range[0], torch, *out_of_range[1:], batch_count=2
        )


def test_compiled_bounded_parity_rejects_non_tie_rank_inversion() -> None:
    fixture = _bounded_parity_fixture()
    reference = fixture[3]["0"][0]
    candidate = fixture[4]["0"][0]
    reference["scores_3d"][:2] = torch.tensor([0.900000, 0.899992])
    query_order = torch.argsort(candidate["query_indices"])
    for field, value in candidate.items():
        candidate[field] = value[query_order].clone()
    candidate["scores_3d"][:2] = torch.tensor([0.899995, 0.899997])
    score_order = torch.argsort(candidate["scores_3d"], descending=True, stable=True)
    for field, value in candidate.items():
        candidate[field] = value[score_order].clone()
    with pytest.raises(tool.ProtocolError, match="non-tied score gap"):
        tool._require_all_batch_compiled_bounded_parity(
            fixture[0], torch, *fixture[1:], batch_count=2
        )


def test_compiled_bounded_parity_rejects_unsorted_decoder_scores() -> None:
    fixture = _bounded_parity_fixture()
    scores = fixture[4]["0"][0]["scores_3d"]
    scores[[10, 11]] = scores[[11, 10]]
    with pytest.raises(tool.ProtocolError, match="must be non-increasing"):
        tool._require_all_batch_compiled_bounded_parity(
            fixture[0], torch, *fixture[1:], batch_count=2
        )


@pytest.mark.parametrize("field", tool.COMPILED_BOUNDED_SCORE_FIELDS)
def test_compiled_bounded_parity_rejects_score_absolute_excess(field) -> None:
    fixture = _bounded_parity_fixture()
    fixture[4]["0"][0][field][0].add_(1.0e-4)
    with pytest.raises(tool.ProtocolError, match=f"exceeded {field}"):
        tool._require_all_batch_compiled_bounded_parity(
            fixture[0], torch, *fixture[1:], batch_count=2
        )


def test_compiled_bounded_parity_rejects_component_box_absolute_excess() -> None:
    fixture = _bounded_parity_fixture(box_dtype=torch.float64)
    fixture[4]["0"][0]["bboxes_3d"][0, 6].add_(0.01)
    with pytest.raises(tool.ProtocolError, match="exceeded decoded-box"):
        tool._require_all_batch_compiled_bounded_parity(
            fixture[0], torch, *fixture[1:], batch_count=2
        )


def test_compiled_bounded_parity_rejects_schema_or_stream_drift() -> None:
    fixture = _bounded_parity_fixture()
    fixture[2]["1"]["stream"] = {"current_stream": 9, "default_stream": 9}
    with pytest.raises(tool.ProtocolError, match="schema/stream field stream"):
        tool._require_all_batch_compiled_bounded_parity(
            fixture[0], torch, *fixture[1:], batch_count=2
        )


def test_parity_payload_is_an_owned_detached_cpu_copy() -> None:
    class Boxes:
        def __init__(self, tensor):
            self.tensor = tensor

    source_fields = {
        "bboxes_3d": Boxes(torch.randn(2, 10, requires_grad=True)),
        "scores_3d": torch.rand(2, requires_grad=True),
        "cls_scores": torch.rand(2, requires_grad=True),
        "labels_3d": torch.tensor([1, 2], dtype=torch.long),
        "query_indices": torch.tensor([7, 4], dtype=torch.long),
        "instance_ids": torch.tensor([70, 40], dtype=torch.long),
    }
    predictions = SimpleNamespace(**source_fields)
    output = [SimpleNamespace(pred_instances_3d=predictions)]

    def prediction_tensor(sample, name):
        value = getattr(sample.pred_instances_3d, name)
        return value.tensor if name == "bboxes_3d" else value

    payload = tool._capture_detached_cpu_parity_payload(
        SimpleNamespace(_prediction_tensor=prediction_tensor), torch, output
    )

    assert len(payload) == 1
    for name, copied in payload[0].items():
        source = source_fields[name]
        source = source.tensor if name == "bboxes_3d" else source
        assert copied.device.type == "cpu"
        assert copied.requires_grad is False
        assert copied.grad_fn is None
        assert copied._base is None
        assert (
            copied.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
        assert torch.equal(copied, source.detach())


def test_actual_frozen_decoder_result_exposes_all_six_parity_fields() -> None:
    from rosa.integrations.sparse4d_scores import decode_frozen_jointscore_rank

    class IdentityDecoder:
        selection_score_mode = "legacy"
        sorted = True
        num_output = 3
        score_threshold = None

        @staticmethod
        def decode_box(boxes):
            return boxes

    classification = torch.randn(1, 3, 6)
    boxes = torch.randn(1, 3, 11)
    labels = torch.tensor([[0, 2, 5]], dtype=torch.long)
    selected_probability = torch.tensor([[0.8, 0.7, 0.6]])
    joint_score = torch.tensor([[0.6, 0.8, 0.7]])
    instance_ids = torch.tensor([[100, 101, 102]], dtype=torch.long)
    decoded = decode_frozen_jointscore_rank(
        IdentityDecoder(),
        [classification],
        [boxes],
        labels,
        selected_probability,
        joint_score,
        instance_id=instance_ids,
        output_idx=0,
        validate_values=True,
        pack_host_transfers=False,
    )

    assert len(decoded) == 1
    assert set(decoded[0]) >= {
        "boxes_3d",
        "scores_3d",
        "cls_scores",
        "labels_3d",
        "query_indices",
        "instance_ids",
    }


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


@pytest.mark.parametrize("profile", ("fast", "balanced"))
def test_c32_candidate_config_changes_only_exact_head_and_profile(profile) -> None:
    native = {
        "model": {
            "backbone": {"depth": 18},
            "head": {
                "type": "ROSAOfficialResidualNativeBaselineHead",
                "rosa_cfg": None,
            },
        },
        "test_dataloader": {"batch_size": 4, "dataset": "Main126"},
        "param_scheduler": [{"end": 30}],
        "seed": 3407,
    }

    candidate = tool._expected_candidate_config(native, profile, "c32")

    assert native["model"]["head"]["type"] == ("ROSAOfficialResidualNativeBaselineHead")
    assert candidate["model"]["head"] == {
        "type": "ROSAOfficialResidualRefinementHead",
        "rosa_cfg": {"profile": profile},
    }
    candidate_without_delta = json.loads(json.dumps(candidate))
    candidate_without_delta["model"]["head"] = native["model"]["head"]
    assert candidate_without_delta == native


@pytest.mark.parametrize(
    ("head_type", "rosa_cfg", "message"),
    (
        ("ROSANativeBaselineHead", None, "exact C32 head"),
        (
            "ROSAOfficialResidualNativeBaselineHead",
            {"profile": "fast"},
            "sidecar settings",
        ),
    ),
)
def test_c32_candidate_derivation_rejects_wrong_native_config(
    head_type, rosa_cfg, message
) -> None:
    native = {
        "model": {
            "head": {
                "type": head_type,
                "rosa_cfg": rosa_cfg,
            }
        }
    }
    with pytest.raises(tool.ProtocolError, match=message):
        tool._expected_candidate_config(native, "fast", "c32")


def test_frozen_c32_runtime_config_is_the_exact_pair_parent() -> None:
    from mmengine import Config

    native = Config.fromfile(
        tool.DEFAULT_C32_NATIVE_CONFIG,
        import_custom_modules=False,
    ).to_dict()
    candidate = tool._expected_candidate_config(native, "fast", "c32")

    assert native["model"]["head"]["type"] == ("ROSAOfficialResidualNativeBaselineHead")
    assert native["model"]["img_backbone"]["depth"] == 18
    assert native["test_dataloader"]["batch_size"] == 4
    assert native["test_dataloader"]["dataset"]["type"] == (
        "Fisheye3DODSparse4DDataset"
    )
    assert native["model"]["head"]["rosa_cfg"] is None
    candidate["model"]["head"] = native["model"]["head"]
    assert candidate == native


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


def test_derive_configs_materializes_c32_fast_eager_from_exact_native() -> None:
    class FakeConfig:
        def __init__(self):
            self.model = SimpleNamespace(
                backbone=SimpleNamespace(depth=18),
                head=SimpleNamespace(
                    type="ROSAOfficialResidualNativeBaselineHead",
                    rosa_cfg=None,
                ),
            )
            self.schedule = {"epochs": 30}

        def to_dict(self):
            return {
                "model": {
                    "backbone": {"depth": self.model.backbone.depth},
                    "head": {
                        "type": self.model.head.type,
                        "rosa_cfg": self.model.head.rosa_cfg,
                    },
                },
                "schedule": self.schedule,
            }

    verifier = SimpleNamespace(
        Config=SimpleNamespace(
            fromfile=lambda path, import_custom_modules: FakeConfig()
        )
    )
    configs = tool._derive_configs(
        verifier,
        Path("unused-c32-native-config.py"),
        tool.selected_arms("fast_eager", "c32"),
        "c32",
    )

    assert configs[tool.C32_NATIVE_ARM].model.head.type == (
        "ROSAOfficialResidualNativeBaselineHead"
    )
    assert configs[tool.FAST_EAGER_ARM].model.head.type == (
        "ROSAOfficialResidualRefinementHead"
    )
    assert configs[tool.FAST_EAGER_ARM].model.head.rosa_cfg == {"profile": "fast"}
    assert (
        configs[tool.C32_NATIVE_ARM].schedule == configs[tool.FAST_EAGER_ARM].schedule
    )


def test_native_checkpoint_filter_accepts_only_rosa_sidecar_state() -> None:
    tool._validate_native_checkpoint_exclusions(
        ["head.rosa_refiner.confidence.weight", "head.rosa_refiner.yaw.bias"]
    )
    with pytest.raises(tool.ProtocolError, match="non-ROSA"):
        tool._validate_native_checkpoint_exclusions(
            ["head.rosa_refiner.yaw.bias", "head.decoder.weight"]
        )


@pytest.mark.parametrize(
    "profile", ("fast", "fast_compiled_bounded", "fast_eager", "balanced")
)
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

    c0_gate = tool._evaluate_near_native_gate(
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
    assert c0_gate["detector_pair"] == "c0"
    assert c0_gate["parent_arm"] == tool.NATIVE_ARM

    c32_memory = {
        tool.C32_NATIVE_ARM: {"peak_allocated_bytes": 100},
        tool.FAST_ARM: {"peak_allocated_bytes": 101},
    }
    c32_gate = tool._evaluate_near_native_gate(
        Verifier,
        {"paired_bootstrap": paired},
        c32_memory,
        tool.FAST_ARM,
        tool.C32_NATIVE_ARM,
    )
    assert captured["memory"] == {
        "logical_parent": c32_memory[tool.C32_NATIVE_ARM],
        "logical_candidate": c32_memory[tool.FAST_ARM],
    }
    assert c32_gate["detector_pair"] == "c32"
    assert c32_gate["parent_arm"] == tool.C32_NATIVE_ARM


def test_plan_json_shape_is_serializable_without_accelerator_types() -> None:
    arms = tool._arms_plan(tool.selected_arms("fast_eager"))
    encoded = json.dumps(arms, sort_keys=True)
    assert "ROSARefinementHead" in encoded
    assert '"model_profile": "fast"' in encoded
    assert '"execution_mode": "eager"' in encoded
    assert set(arms) == {tool.NATIVE_ARM, tool.FAST_EAGER_ARM}
