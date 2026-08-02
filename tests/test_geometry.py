import math
import sys

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import rosa.observation as observation_module
from rosa.geometry import (
    _ocam_project_points_with_raw_camera_xyz,
    ocam_project_points,
    ocam_projection_scales,
)
from rosa.observation import _camera_xyz, project_and_describe
from rosa.refiner import ROSAConfig
from rosa.types import RayFeatureRelation


def _metas():
    intrinsic = torch.eye(3).reshape(1, 1, 3, 3)
    intrinsic[..., 0, 2] = 400.0
    intrinsic[..., 1, 2] = 400.0
    polynomial = torch.zeros(1, 1, 5)
    polynomial[..., 1] = -150.0
    return {
        "lidar2cam": torch.eye(4).reshape(1, 1, 4, 4),
        "ocam_intrinsic": intrinsic,
        "ocam_pol": polynomial,
        "image_wh": torch.full((1, 1, 2), 800.0),
        "ocam_lens_fov_deg": 220.0,
    }


def _ray_at_incidence(degrees):
    angle = math.radians(degrees)
    # OCam runtime convention: forward is -Z.
    return torch.tensor(
        (math.sin(angle), 0.0, -math.cos(angle)),
        dtype=torch.float32,
    ).reshape(1, 1, 1, 3)


def _raw_equal(left, right):
    integer_dtype = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }[left.dtype]
    return torch.equal(
        left.contiguous().view(integer_dtype),
        right.contiguous().view(integer_dtype),
    )


def _assert_observation_raw_equal(actual, expected):
    assert type(actual) is type(expected) is tuple
    assert len(actual) == len(expected) == 4
    for candidate, parent in zip(actual, expected):
        assert type(candidate) is type(parent)
        if type(candidate) is RayFeatureRelation:
            assert _raw_equal(candidate.ray, parent.ray)
            assert torch.equal(candidate.valid, parent.valid)
        elif candidate.is_floating_point():
            assert candidate.dtype == parent.dtype
            assert _raw_equal(candidate, parent)
        else:
            assert torch.equal(candidate, parent)


def test_single_transform_candidate_stays_disabled_by_default():
    assert observation_module._ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE is False


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
@pytest.mark.parametrize("noncontiguous", (False, True))
def test_single_transform_candidate_preserves_projection_and_raw_xyz(
    dtype,
    noncontiguous,
):
    metas = {
        name: value.to(dtype=dtype) if torch.is_tensor(value) else value
        for name, value in _metas().items()
    }
    values = torch.tensor(
        (
            (0.0, -0.0, -1.0),
            (0.5, -0.25, -2.0),
            (float("nan"), 0.0, -1.0),
            (float("inf"), float("-inf"), 0.0),
            (9999.0, 0.0, -1.0),
            (10000.0, 0.0, -1.0),
        ),
        dtype=dtype,
    ).reshape(1, 1, 6, 3)
    if noncontiguous:
        backing = torch.empty(1, 1, 6, 6, dtype=dtype)
        backing[..., ::2] = values
        points = backing[..., ::2]
        assert not points.is_contiguous()
    else:
        points = values

    with torch.no_grad():
        expected_projection = ocam_project_points(
            points,
            metas,
            metas["image_wh"],
            contract="full_fov",
            inference_storage_reuse=True,
        )
        actual_projection, raw_camera_xyz = (
            _ocam_project_points_with_raw_camera_xyz(
                points,
                metas,
                metas["image_wh"],
                contract="full_fov",
                inference_storage_reuse=True,
            )
        )
        expected_camera_xyz = _camera_xyz(points, metas["lidar2cam"])

    assert _raw_equal(actual_projection, expected_projection)
    assert _raw_equal(raw_camera_xyz, expected_camera_xyz)
    assert raw_camera_xyz.shape == expected_camera_xyz.shape
    assert raw_camera_xyz.stride() == expected_camera_xyz.stride()
    assert raw_camera_xyz.dtype == expected_camera_xyz.dtype
    assert raw_camera_xyz._base is None


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
@pytest.mark.parametrize(
    ("projection_contract", "compact_relation"),
    (
        ("legacy_front_only", False),
        ("legacy_front_only", True),
        ("full_fov", False),
        ("full_fov", True),
    ),
)
def test_single_transform_candidate_is_raw_exact_end_to_end(
    dtype,
    projection_contract,
    compact_relation,
    monkeypatch,
):
    metas = {
        name: value.to(dtype=dtype) if torch.is_tensor(value) else value
        for name, value in _metas().items()
    }
    keypoints = torch.tensor(
        [[[[0.0, -0.0, -1.0], [0.5, 0.0, -2.0],
           [float("nan"), 0.0, -1.0], [10000.0, 0.0, -1.0]]]],
        dtype=dtype,
    )
    anchor = torch.zeros(1, 1, 11, dtype=dtype)
    anchor[..., :3] = torch.tensor((0.25, 0.0, -1.0), dtype=dtype)
    anchor[..., 3:6] = torch.tensor(
        (1.5, 3.0, 1.5),
        dtype=dtype,
    ).log()
    anchor[..., 7] = 1.0

    with torch.no_grad():
        expected = project_and_describe(
            keypoints,
            anchor,
            metas,
            projection_contract,
            compact_relation=compact_relation,
        )
        monkeypatch.setattr(
            observation_module,
            "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
            True,
        )
        actual = project_and_describe(
            keypoints,
            anchor,
            metas,
            projection_contract,
            compact_relation=compact_relation,
        )

    _assert_observation_raw_equal(actual, expected)


def test_eager_single_transform_candidate_executes_one_camera_transform(monkeypatch):
    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, -0.0, -1.0], [0.5, 0.0, -2.0]]]],
    )
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 7] = 1.0
    transform_codes = {
        "projection": observation_module._geometry._camera_xyz.__code__,
        "descriptor": observation_module._camera_xyz.__code__,
    }

    def invoke(enabled):
        calls = {name: 0 for name in transform_codes}

        def profile(frame, event, _arg):
            if event == "call":
                for name, code in transform_codes.items():
                    if frame.f_code is code:
                        calls[name] += 1
            return profile

        monkeypatch.setattr(
            observation_module,
            "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
            enabled,
        )
        previous_profile = sys.getprofile()
        sys.setprofile(profile)
        try:
            with torch.no_grad():
                output = project_and_describe(
                    keypoints,
                    anchor,
                    metas,
                    "full_fov",
                )
        finally:
            sys.setprofile(previous_profile)
        return output, calls

    expected, released_calls = invoke(False)
    actual, candidate_calls = invoke(True)

    assert released_calls == {"projection": 1, "descriptor": 1}
    assert candidate_calls == {"projection": 1, "descriptor": 0}
    _assert_observation_raw_equal(actual, expected)


def test_single_transform_candidate_fails_closed_for_extension_surfaces(
    monkeypatch,
):
    metas = _metas()
    keypoints = torch.zeros(1, 1, 1, 3)
    monkeypatch.setattr(
        observation_module,
        "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
        True,
    )

    assert not observation_module._can_reuse_camera_transform(
        keypoints,
        metas,
    )
    with torch.no_grad():
        assert observation_module._can_reuse_camera_transform(
            keypoints,
            metas,
        )
        custom_mapping = type("CustomMetadata", (dict,), {})(metas)
        assert not observation_module._can_reuse_camera_transform(
            keypoints,
            custom_mapping,
        )
        released_geometry_camera_xyz = (
            observation_module._geometry._camera_xyz
        )
        monkeypatch.setattr(
            observation_module._geometry,
            "_camera_xyz",
            lambda points, lidar2cam: released_geometry_camera_xyz(
                points,
                lidar2cam,
            ),
        )
        assert not observation_module._can_reuse_camera_transform(
            keypoints,
            metas,
        )
        monkeypatch.setattr(
            observation_module._geometry,
            "_camera_xyz",
            released_geometry_camera_xyz,
        )
        released_camera_xyz = observation_module._camera_xyz
        monkeypatch.setattr(
            observation_module,
            "_camera_xyz",
            lambda points, lidar2cam: released_camera_xyz(points, lidar2cam),
        )
        assert not observation_module._can_reuse_camera_transform(
            keypoints,
            metas,
        )


def test_single_transform_candidate_preserves_callable_hooks(monkeypatch):
    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, -0.0, -1.0], [0.5, 0.0, -2.0]]]],
    )
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 7] = 1.0
    surfaces = (
        (observation_module, "_camera_xyz", 1),
        (observation_module, "ocam_project_points", 1),
        (observation_module, "_ocam_project_points_with_raw_camera_xyz", 0),
        (observation_module._geometry, "_camera_xyz", 1),
        (observation_module._geometry, "_ocam_project_points_impl", 1),
    )

    for owner, name, calls_per_fallback in surfaces:
        original = getattr(owner, name)
        calls = []

        def hooked(*args, _original=original, **kwargs):
            calls.append(None)
            return _original(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(owner, name, hooked)
            patch.setattr(
                observation_module,
                "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
                False,
            )
            with torch.no_grad():
                expected = project_and_describe(
                    keypoints,
                    anchor,
                    metas,
                    "full_fov",
                )
            assert len(calls) == calls_per_fallback

            patch.setattr(
                observation_module,
                "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
                True,
            )
            with torch.no_grad():
                actual = project_and_describe(
                    keypoints,
                    anchor,
                    metas,
                    "full_fov",
                )
            assert len(calls) == 2 * calls_per_fallback

        _assert_observation_raw_equal(actual, expected)


def test_single_transform_candidate_preserves_torch_override_modes(monkeypatch):
    class RecordingFunctionMode(TorchFunctionMode):
        def __init__(self):
            self.calls = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.calls.append(func)
            return func(*args, **(kwargs or {}))

    class RecordingDispatchMode(TorchDispatchMode):
        def __init__(self):
            self.calls = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.calls.append(func)
            return func(*args, **(kwargs or {}))

    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, -0.0, -1.0], [0.5, 0.0, -2.0]]]],
    )
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 7] = 1.0

    def invoke(enabled, mode_type):
        mode = mode_type()
        monkeypatch.setattr(
            observation_module,
            "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
            enabled,
        )
        with torch.no_grad(), mode:
            assert not observation_module._can_reuse_camera_transform(
                keypoints,
                metas,
            )
            output = project_and_describe(
                keypoints,
                anchor,
                metas,
                "full_fov",
            )
        return output, mode.calls

    for mode_type in (RecordingFunctionMode, RecordingDispatchMode):
        expected, expected_calls = invoke(False, mode_type)
        actual, actual_calls = invoke(True, mode_type)
        assert actual_calls == expected_calls
        _assert_observation_raw_equal(actual, expected)


def test_dispatch_mode_guard_fails_closed_without_infra_api(monkeypatch):
    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, -0.0, -1.0], [0.5, 0.0, -2.0]]]],
    )
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 7] = 1.0

    monkeypatch.setattr(
        observation_module,
        "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
        False,
    )
    with torch.no_grad():
        expected = project_and_describe(
            keypoints,
            anchor,
            metas,
            "full_fov",
        )

    monkeypatch.setattr(
        observation_module,
        "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
        True,
    )
    monkeypatch.delattr(
        observation_module._python_dispatch,
        "_is_in_non_infra_torch_dispatch_mode",
        raising=False,
    )
    with torch.no_grad():
        assert not observation_module._can_reuse_camera_transform(
            keypoints,
            metas,
        )
        actual = project_and_describe(
            keypoints,
            anchor,
            metas,
            "full_fov",
        )

    _assert_observation_raw_equal(actual, expected)


def test_single_transform_candidate_preserves_tensor_subclass_dispatch(monkeypatch):
    class TensorSubclass(torch.Tensor):
        pass

    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, -0.0, -1.0], [0.5, 0.0, -2.0]]]],
    ).as_subclass(TensorSubclass)
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 7] = 1.0

    monkeypatch.setattr(
        observation_module,
        "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
        False,
    )
    with torch.no_grad():
        expected = project_and_describe(
            keypoints,
            anchor,
            metas,
            "full_fov",
        )
    monkeypatch.setattr(
        observation_module,
        "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
        True,
    )
    with torch.no_grad():
        assert not observation_module._can_reuse_camera_transform(
            keypoints,
            metas,
        )
        actual = project_and_describe(
            keypoints,
            anchor,
            metas,
            "full_fov",
        )

    _assert_observation_raw_equal(actual, expected)


def test_single_transform_candidate_preserves_metadata_container_hooks(monkeypatch):
    class RecordingMetadata(dict):
        def __init__(self, values):
            super().__init__(values)
            self.calls = []

        def __contains__(self, key):
            self.calls.append(("contains", key))
            return super().__contains__(key)

        def __getitem__(self, key):
            self.calls.append(("getitem", key))
            return super().__getitem__(key)

        def get(self, key, default=None):
            self.calls.append(("get", key))
            return super().get(key, default)

    keypoints = torch.tensor(
        [[[[0.0, -0.0, -1.0], [0.5, 0.0, -2.0]]]],
    )
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 7] = 1.0

    def invoke(enabled):
        metadata = RecordingMetadata(_metas())
        monkeypatch.setattr(
            observation_module,
            "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
            enabled,
        )
        with torch.no_grad():
            output = project_and_describe(
                keypoints,
                anchor,
                metadata,
                "full_fov",
            )
        return output, metadata.calls

    expected, expected_calls = invoke(False)
    actual, actual_calls = invoke(True)

    assert actual_calls == expected_calls
    _assert_observation_raw_equal(actual, expected)


def test_single_transform_candidate_preserves_named_tensor_exception(monkeypatch):
    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, -0.0, -1.0], [0.5, 0.0, -2.0]]]],
    ).refine_names("batch", "query", "point", "xyz")
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 7] = 1.0
    errors = []

    for enabled in (False, True):
        monkeypatch.setattr(
            observation_module,
            "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
            enabled,
        )
        with torch.no_grad(), pytest.raises(RuntimeError) as captured:
            project_and_describe(
                keypoints,
                anchor,
                metas,
                "full_fov",
            )
        errors.append(captured.value)

    assert type(errors[0]) is type(errors[1])
    assert errors[0].args == errors[1].args


def test_single_transform_candidate_training_fallback_is_raw_exact(monkeypatch):
    def invoke(enabled):
        metas = {
            name: (
                value.double().detach().clone().requires_grad_(True)
                if torch.is_tensor(value)
                else value
            )
            for name, value in _metas().items()
        }
        keypoints = torch.tensor(
            [[[[0.2, -0.1, -2.0], [0.5, 0.25, -3.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        anchor = torch.zeros(1, 1, 11, dtype=torch.float64)
        anchor[..., :3] = torch.tensor((0.1, -0.2, -2.5))
        anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5)).log()
        anchor[..., 7] = 1.0
        anchor.requires_grad_()
        monkeypatch.setattr(
            observation_module,
            "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
            enabled,
        )
        output = project_and_describe(
            keypoints,
            anchor,
            metas,
            "full_fov",
        )
        loss = output[0].sum() + output[1].sum() + output[2].sum()
        leaves = (
            keypoints,
            anchor,
            metas["lidar2cam"],
            metas["image_wh"],
            metas["ocam_intrinsic"],
            metas["ocam_pol"],
        )
        gradients = torch.autograd.grad(loss, leaves)
        return output, gradients

    expected, expected_gradients = invoke(False)
    actual, actual_gradients = invoke(True)

    _assert_observation_raw_equal(actual, expected)
    for candidate, parent in zip(actual_gradients, expected_gradients):
        assert _raw_equal(candidate, parent)


def test_single_transform_candidate_does_not_expand_conditioned_input_api(
    monkeypatch,
):
    metas = _metas()
    conditioned = _ray_at_incidence(45.0)[:, None]
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 7] = 1.0
    monkeypatch.setattr(
        observation_module,
        "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
        True,
    )

    with torch.no_grad(), pytest.raises(
        ValueError,
        match=r"points must be \[B,N,P,3\]",
    ):
        project_and_describe(
            conditioned,
            anchor,
            metas,
            "full_fov",
        )


@pytest.mark.parametrize("candidate_enabled", (False, True))
def test_no_grad_descriptor_remains_fullgraph_compilable(
    candidate_enabled,
    monkeypatch,
):
    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, 0.0, -1.0], [0.5, 0.0, -1.0]]]],
    )
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5)).log()
    anchor[..., 7] = 1.0
    monkeypatch.setattr(
        observation_module,
        "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
        candidate_enabled,
    )

    def describe(points, boxes, lidar2cam, intrinsic, polynomial, image_wh):
        metadata = {
            "lidar2cam": lidar2cam,
            "ocam_intrinsic": intrinsic,
            "ocam_pol": polynomial,
            "image_wh": image_wh,
        }
        return project_and_describe(
            points,
            boxes,
            metadata,
            "full_fov",
        )

    inputs = (
        keypoints,
        anchor,
        metas["lidar2cam"],
        metas["ocam_intrinsic"],
        metas["ocam_pol"],
        metas["image_wh"],
    )
    compiled = torch.compile(describe, backend="eager", fullgraph=True)
    with torch.no_grad():
        expected = describe(*inputs)
        actual = compiled(*inputs)

    for candidate, parent in zip(actual, expected):
        assert candidate.dtype == parent.dtype
        if candidate.is_floating_point():
            assert _raw_equal(candidate, parent)
        else:
            assert torch.equal(candidate, parent)


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
@pytest.mark.parametrize("compact_relation", (False, True))
def test_compiled_single_transform_candidate_is_raw_exact_for_special_values(
    dtype,
    compact_relation,
    monkeypatch,
):
    metas = {
        name: value.to(dtype=dtype) if torch.is_tensor(value) else value
        for name, value in _metas().items()
    }
    values = torch.tensor(
        [
            [
                [
                    [0.0, -0.0, -1.0],
                    [0.5, -0.25, -2.0],
                    [float("nan"), 0.0, -1.0],
                    [float("inf"), float("-inf"), 0.0],
                    [9999.0, 0.0, -1.0],
                    [10000.0, 0.0, -1.0],
                ]
            ]
        ],
        dtype=dtype,
    )
    backing = torch.empty(1, 1, 6, 6, dtype=dtype)
    backing[..., ::2] = values
    keypoints = backing[..., ::2]
    assert not keypoints.is_contiguous()
    anchor = torch.zeros(1, 1, 11, dtype=dtype)
    anchor[..., :3] = torch.tensor((0.25, -0.0, -1.0), dtype=dtype)
    anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5), dtype=dtype).log()
    anchor[..., 7] = 1.0

    def describe(points, boxes, lidar2cam, intrinsic, polynomial, image_wh):
        metadata = {
            "lidar2cam": lidar2cam,
            "ocam_intrinsic": intrinsic,
            "ocam_pol": polynomial,
            "image_wh": image_wh,
        }
        return project_and_describe(
            points,
            boxes,
            metadata,
            "full_fov",
            compact_relation=compact_relation,
        )

    inputs = (
        keypoints,
        anchor,
        metas["lidar2cam"],
        metas["ocam_intrinsic"],
        metas["ocam_pol"],
        metas["image_wh"],
    )
    monkeypatch.setattr(
        observation_module,
        "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
        False,
    )
    with torch.no_grad():
        expected = describe(*inputs)
    monkeypatch.setattr(
        observation_module,
        "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
        True,
    )

    torch._dynamo.reset()
    try:
        compiled = torch.compile(describe, backend="eager", fullgraph=True)
        with torch.no_grad():
            actual = compiled(*inputs)
    finally:
        torch._dynamo.reset()

    _assert_observation_raw_equal(actual, expected)


def test_compiled_candidate_graph_contains_one_camera_transform(monkeypatch):
    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, -0.0, -1.0], [0.5, 0.0, -2.0]]]],
    )
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5)).log()
    anchor[..., 7] = 1.0

    def describe(points, boxes, lidar2cam, intrinsic, polynomial, image_wh):
        metadata = {
            "lidar2cam": lidar2cam,
            "ocam_intrinsic": intrinsic,
            "ocam_pol": polynomial,
            "image_wh": image_wh,
        }
        return project_and_describe(
            points,
            boxes,
            metadata,
            "full_fov",
            compact_relation=True,
        )

    inputs = (
        keypoints,
        anchor,
        metas["lidar2cam"],
        metas["ocam_intrinsic"],
        metas["ocam_pol"],
        metas["image_wh"],
    )

    def capture(enabled):
        graphs = []

        def eager_backend(graph_module, _example_inputs):
            graphs.append(graph_module)
            return graph_module.forward

        monkeypatch.setattr(
            observation_module,
            "_ENABLE_SINGLE_CAMERA_TRANSFORM_CANDIDATE",
            enabled,
        )
        torch._dynamo.reset()
        compiled = torch.compile(
            describe,
            backend=eager_backend,
            fullgraph=True,
        )
        with torch.no_grad():
            output = compiled(*inputs)
        assert len(graphs) == 1
        return graphs[0], output

    try:
        released_graph, expected = capture(False)
        candidate_graph, actual = capture(True)
    finally:
        torch._dynamo.reset()

    def camera_transform_count(graph_module):
        return sum(
            node.op == "call_function" and node.target is torch.matmul
            for node in graph_module.graph.nodes
        )

    assert camera_transform_count(released_graph) == 2
    assert camera_transform_count(candidate_graph) == 1
    assert not any(
        node.op == "call_method" and node.target == "copy_"
        for node in candidate_graph.graph.nodes
    )
    _assert_observation_raw_equal(actual, expected)


def test_legacy_contract_reproduces_front_plane_clip():
    metas = _metas()
    ray_100 = _ray_at_incidence(100.0)
    legacy = ocam_project_points(
        ray_100,
        metas,
        metas["image_wh"],
        contract="legacy_front_only",
    )
    corrected = ocam_project_points(
        ray_100,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(legacy, torch.full_like(legacy, 2.0))
    assert bool(((corrected >= 0) & (corrected < 1)).all())


def test_full_fov_still_rejects_ray_outside_declared_lens():
    metas = _metas()
    ray_115 = _ray_at_incidence(115.0)
    projected = ocam_project_points(
        ray_115,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(projected, torch.full_like(projected, 2.0))


def test_published_checkpoint_contract_is_explicit_default():
    assert ROSAConfig().projection_contract == "legacy_front_only"


def test_corrected_profiles_use_the_complete_declared_fisheye_fov():
    assert (
        ROSAConfig.from_mapping({"profile": "balanced"}).projection_contract
        == "full_fov"
    )
    assert (
        ROSAConfig.from_mapping({"profile": "fast"}).projection_contract == "full_fov"
    )


def test_full_fov_rejects_the_undefined_camera_origin():
    metas = _metas()
    projected = ocam_project_points(
        torch.zeros(1, 1, 1, 3),
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(projected, torch.full_like(projected, 2.0))


def test_full_fov_rejects_unstable_or_nonfinite_camera_positions():
    metas = _metas()
    points = torch.tensor(
        [
            [
                [5e-4, 0.0, 0.0],
                [float("nan"), 0.0, -1.0],
                [float("inf"), 0.0, -1.0],
                [1e5, 0.0, -1.0],
            ]
        ],
        dtype=torch.float32,
    ).unsqueeze(1)
    projected = ocam_project_points(
        points,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(projected, torch.full_like(projected, 2.0))


def test_projection_supports_camera_conditioned_keypoints():
    metas = _metas()
    shared = _ray_at_incidence(45.0)
    conditioned = shared[:, None]
    expected = ocam_project_points(
        shared,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    actual = ocam_project_points(
        conditioned,
        metas,
        metas["image_wh"],
        contract="full_fov",
    )
    assert torch.equal(actual, expected)


def test_full_fov_projection_scales_include_anisotropic_ocam_affine():
    metas = {
        name: value.double() if isinstance(value, torch.Tensor) else value
        for name, value in _metas().items()
    }
    intrinsic = metas["ocam_intrinsic"].clone()
    intrinsic[..., 0, 0] = 2.0
    intrinsic[..., 0, 1] = 0.4
    intrinsic[..., 1, 0] = -0.3
    intrinsic[..., 1, 1] = 0.7
    metas["ocam_intrinsic"] = intrinsic

    incidence = 0.7
    azimuth = 0.4
    ray = torch.tensor(
        (
            math.sin(incidence) * math.cos(azimuth),
            math.sin(incidence) * math.sin(azimuth),
            -math.cos(incidence),
        ),
        dtype=torch.float64,
    )
    radial_direction = torch.tensor(
        (
            math.cos(incidence) * math.cos(azimuth),
            math.cos(incidence) * math.sin(azimuth),
            math.sin(incidence),
        ),
        dtype=torch.float64,
    )
    tangential_direction = torch.tensor(
        (-math.sin(azimuth), math.cos(azimuth), 0.0),
        dtype=torch.float64,
    )
    shaped_ray = ray.reshape(1, 1, 1, 1, 3)

    legacy = ocam_projection_scales(
        shaped_ray,
        metas,
        contract="legacy_front_only",
    )
    identity_metas = dict(metas)
    identity_metas["ocam_intrinsic"] = torch.eye(
        3,
        dtype=torch.float64,
    ).reshape(1, 1, 3, 3)
    legacy_identity = ocam_projection_scales(
        shaped_ray,
        identity_metas,
        contract="legacy_front_only",
    )
    assert torch.equal(legacy[0], legacy_identity[0])
    assert torch.equal(legacy[1], legacy_identity[1])

    corrected = ocam_projection_scales(
        shaped_ray,
        metas,
        contract="full_fov",
    )
    epsilon = 1e-6
    for scale, direction in zip(
        corrected,
        (radial_direction, tangential_direction),
    ):
        positive = (ray * math.cos(epsilon) + direction * math.sin(epsilon)).reshape(
            1, 1, 1, 3
        )
        negative = (ray * math.cos(epsilon) - direction * math.sin(epsilon)).reshape(
            1, 1, 1, 3
        )
        positive_pixel = (
            ocam_project_points(
                positive,
                metas,
                metas["image_wh"],
                contract="full_fov",
            )[0, 0, 0, 0]
            * metas["image_wh"][0, 0]
        )
        negative_pixel = (
            ocam_project_points(
                negative,
                metas,
                metas["image_wh"],
                contract="full_fov",
            )[0, 0, 0, 0]
            * metas["image_wh"][0, 0]
        )
        finite_difference = torch.linalg.vector_norm(
            positive_pixel - negative_pixel
        ) / (2.0 * epsilon)
        torch.testing.assert_close(
            scale.squeeze(),
            finite_difference,
            rtol=1e-7,
            atol=1e-7,
        )


def test_full_fov_boundary_is_zero_on_declared_angular_cone():
    metas = {
        name: value.double() if isinstance(value, torch.Tensor) else value
        for name, value in _metas().items()
    }
    metas["ocam_lens_fov_deg"] = 180.0
    ray_on_cone = torch.tensor(
        (1.0, 0.0, 0.0),
        dtype=torch.float64,
    ).reshape(1, 1, 1, 3)
    anchor = torch.zeros(1, 1, 8, dtype=torch.float64)
    anchor[..., 7] = 1.0

    uv, geometry, _, valid = project_and_describe(
        ray_on_cone,
        anchor,
        metas,
        "full_fov",
    )

    assert bool(valid.item())
    torch.testing.assert_close(uv, torch.full_like(uv, 0.5))
    assert torch.equal(geometry[..., 5], torch.zeros_like(geometry[..., 5]))


def test_legacy_boundary_remains_the_rectangular_image_margin():
    metas = _metas()
    metas["ocam_lens_fov_deg"] = 90.0
    ray_near_cone = _ray_at_incidence(44.0)
    anchor = torch.zeros(1, 1, 8)
    anchor[..., 7] = 1.0

    uv, geometry, _, valid = project_and_describe(
        ray_near_cone,
        anchor,
        metas,
        "legacy_front_only",
    )
    expected = (
        torch.minimum(
            torch.minimum(uv[..., 0], 1.0 - uv[..., 0]),
            torch.minimum(uv[..., 1], 1.0 - uv[..., 1]),
        )
        .mul(2.0)
        .clamp(0.0, 1.0)
    )

    assert bool(valid.item())
    assert torch.equal(geometry[..., 5], expected)
    assert bool((geometry[..., 5] > 0.5).all())


def test_bottom_z_descriptor_uses_the_geometric_box_center():
    metas = _metas()
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 0] = 1.0
    anchor[..., 2] = -1.0
    anchor[..., 7] = 1.0
    keypoint = torch.tensor([[[[1.0, 0.0, -0.5]]]])
    _, bottom_geometry, _, _ = project_and_describe(
        keypoint,
        anchor,
        metas,
        "full_fov",
        anchor_z_is_bottom=True,
    )
    _, centered_geometry, _, _ = project_and_describe(
        keypoint,
        anchor,
        metas,
        "legacy_front_only",
        anchor_z_is_bottom=False,
    )
    torch.testing.assert_close(
        bottom_geometry[..., -3:],
        torch.zeros_like(bottom_geometry[..., -3:]),
    )
    torch.testing.assert_close(
        centered_geometry[..., -1],
        torch.full_like(centered_geometry[..., -1], 0.5),
    )


def test_project_and_describe_eval_storage_reuse_is_raw_exact():
    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, 0.0, -1.0], [0.5, 0.0, -1.0], [float("nan"), 0.0, -1.0]]]]
    )
    anchor = torch.zeros(1, 1, 11)
    anchor[..., :3] = torch.tensor((0.25, 0.0, -1.0))
    anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5)).log()
    anchor[..., 7] = 1.0

    expected = project_and_describe(
        keypoints,
        anchor,
        metas,
        "full_fov",
    )
    with torch.no_grad():
        actual = project_and_describe(
            keypoints,
            anchor,
            metas,
            "full_fov",
        )

    for candidate, parent in zip(actual, expected):
        if type(candidate) is RayFeatureRelation:
            assert type(parent) is RayFeatureRelation
            assert torch.equal(candidate.ray, parent.ray)
            assert torch.equal(candidate.valid, parent.valid)
            continue
        if candidate.is_floating_point():
            integer_dtype = {
                torch.float16: torch.int16,
                torch.bfloat16: torch.int16,
                torch.float32: torch.int32,
                torch.float64: torch.int64,
            }[candidate.dtype]
            assert torch.equal(
                candidate.contiguous().view(integer_dtype),
                parent.contiguous().view(integer_dtype),
            )
        else:
            assert torch.equal(candidate, parent)


def test_compact_rfl_relation_preserves_legacy_values_and_owns_storage():
    metas = _metas()
    keypoints = torch.tensor(
        [[[[0.0, 0.0, -1.0], [0.5, 0.0, -1.0], [0.0, 0.5, -1.0]]]]
    )
    anchor = torch.zeros(1, 1, 11)
    anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5)).log()
    anchor[..., 7] = 1.0
    _, geometry, legacy, valid = project_and_describe(
        keypoints,
        anchor,
        metas,
        "full_fov",
    )
    _, compact_geometry, compact, compact_valid = project_and_describe(
        keypoints,
        anchor,
        metas,
        "full_fov",
        compact_relation=True,
    )

    assert type(compact) is RayFeatureRelation
    legacy_bncp = legacy.permute(0, 2, 1, 3, 4)
    assert torch.equal(compact.ray, legacy_bncp[..., 8:11])
    assert torch.equal(compact.valid, legacy_bncp[..., 13] > 0)
    assert torch.equal(compact_geometry, geometry)
    assert torch.equal(compact_valid, valid)
    assert compact.ray._base is None
    assert compact.valid._base is None

    compact.ray.zero_()
    compact.valid.logical_not_()
    assert torch.equal(compact_geometry, geometry)
    assert torch.equal(compact_valid, valid)
