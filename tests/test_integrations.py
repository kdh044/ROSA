from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from rosa import ROSAConfig
import rosa.integrations.sparse4d_scores as sparse4d_scores_module
import rosa.rfl as rfl_module
from rosa.integrations.sparse4d_runtime import (
    NativeRefineCall,
    SingleCallCapture,
    detached_fpn,
    remap_legacy_refiner_state_dict,
    validate_detector_owned_rosa_output,
    validate_native_call,
)
from rosa.integrations.sparse4d_scores import (
    _can_reuse_native_identity_rank_selection,
    _compose_native_identity_rank_and_selection,
    _pack_frozen_decode_fields,
    compose_native_identity_rank,
    compose_quality_rank,
    decode_frozen_jointscore_rank,
    stable_joint_probability,
)


ROOT = Path(__file__).resolve().parents[1]


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    integer_dtype = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }.get(left.dtype, left.dtype)
    return bool(
        torch.equal(
            left.contiguous().view(integer_dtype),
            right.contiguous().view(integer_dtype),
        )
    )


def _detector_host() -> Path:
    return Path(
        os.environ.get(
            "ROSA_DETECTOR_HOST_ROOT",
            ROOT / "external" / "detector_host",
        )
    ).resolve()


def test_integration_package_does_not_eagerly_import_sparse4d():
    code = textwrap.dedent(
        """
        import sys
        import rosa.integrations

        assert rosa.integrations.__name__ == "rosa.integrations"
        assert "rosa.integrations.sparse4d" not in sys.modules
        """
    )
    environment = os.environ.copy()
    pythonpath = [str(ROOT / "src")]
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


def test_gravity_center_keypoints_are_symmetric_about_anchor_z():
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent(
        """
        import torch
        from rosa.integrations.sparse4d import ROSANormalizedKeyPointsGenerator

        generator = ROSANormalizedKeyPointsGenerator(
            embed_dims=16,
            num_learnable_pts=0,
            fix_scale=((0.0, 0.0, 0.0),
                       (0.0, 0.0, 0.45),
                       (0.0, 0.0, -0.45)),
            anchor_z_is_bottom=False,
        )
        anchor = torch.zeros(1, 1, 11)
        anchor[..., :3] = torch.tensor((3.0, -2.0, -1.3))
        anchor[..., 3:6] = torch.log(torch.tensor((2.0, 4.0, 1.6)))
        anchor[..., 7] = 1.0
        points = generator(anchor)
        torch.testing.assert_close(points[..., 0, :], anchor[..., :3])
        upper = points[..., 1, 2] - anchor[..., 2]
        lower = points[..., 2, 2] - anchor[..., 2]
        torch.testing.assert_close(upper, -lower)
        """
    )
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


def test_compact_sidecar_preserves_paired_native_initialization_and_rng():
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent(
        """
        import random
        import numpy as np
        import torch
        from mmengine.config import Config
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
            return model, torch.get_rng_state().clone()

        baseline, baseline_rng = build("rosa_r18_main126_v2_b0_native.py")
        rosa, rosa_rng = build("rosa_r18_main126_v2.py")
        baseline_state = baseline.state_dict()
        rosa_state = rosa.state_dict()
        common = set(baseline_state).intersection(rosa_state)
        assert common
        assert all(
            torch.equal(baseline_state[name], rosa_state[name])
            for name in common
        )
        torch.testing.assert_close(baseline_rng, rosa_rng, rtol=0, atol=0)
        """
    )
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


def test_rosa_query_capture_observes_replacing_hooks_and_cleans_up():
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent(
        """
        import types
        from types import SimpleNamespace
        import torch
        from rosa.integrations import sparse4d as integration

        integration.ROSARefinementHead._validate_native_host_contract()
        released_hook = integration.Sparse4DHead._pre_refine_query_feature
        integration.Sparse4DHead._pre_refine_query_feature = (
            lambda self, **kwargs: kwargs["instance_feature"]
        )
        try:
            try:
                integration.ROSARefinementHead._validate_native_host_contract()
            except RuntimeError as error:
                assert "incompatible Sparse4D detector host" in str(error)
            else:
                raise AssertionError("an incompatible detector host was accepted")
        finally:
            integration.Sparse4DHead._pre_refine_query_feature = released_hook

        class NativeRefine(torch.nn.Module):
            def forward(self, query, anchor, anchor_embed, **kwargs):
                return anchor, None, None

        def make_head(native=None):
            head = object.__new__(integration.ROSARefinementHead)
            torch.nn.Module.__init__(head)
            head.layers = torch.nn.ModuleList([native or NativeRefine()])
            head._native_refine_layer_index = 0
            observed = []
            def append(_self, _output, query, _fpn, _metas):
                observed.append(query.clone())
            head._append_rosa = types.MethodType(append, head)
            return head, observed

        def native_forward(self, feature_maps, metas):
            query = torch.zeros(1, 2, 4)
            anchor = torch.zeros(1, 2, 11)
            anchor_embed = torch.zeros(1, 2, 4)
            query = self._pre_refine_query_feature(
                decoder_index=5,
                instance_feature=query,
                anchor=anchor,
                anchor_embed=anchor_embed,
                feature_maps=feature_maps,
                metas=metas,
                predictions=[],
                classifications=[],
                qualities=[],
                num_free_instance=2,
                classification_feature=None,
                regression_feature=None,
                score_alignment_feature=None,
            )
            self.layers[0](
                query,
                anchor,
                anchor_embed,
                time_interval=None,
                return_cls=True,
                classification_feature=None,
                regression_feature=None,
                score_alignment_feature=None,
            )
            return {}

        feature_maps = [torch.zeros(1, 1, 4, 2, 2)]
        probe, _ = make_head()
        probe.rosa_refiner = torch.nn.Identity()
        probe.rosa_cfg = SimpleNamespace(export_diagnostics=False)
        probe.eval()
        with torch.no_grad():
            assert probe._lean_inference() is True
            hook = probe.rosa_refiner.register_forward_hook(
                lambda *_args: None
            )
            try:
                assert probe._lean_inference() is False
            finally:
                hook.remove()
            global_hook = torch.nn.modules.module.register_module_forward_hook(
                lambda *_args: None
            )
            try:
                assert probe._lean_inference() is False
            finally:
                global_hook.remove()

        released_forward = integration.Sparse4DHead.forward
        integration.Sparse4DHead.forward = native_forward
        try:
            head, observed = make_head()
            local = head.layers[0].register_forward_pre_hook(
                lambda _module, args, kwargs: (
                    (args[0] + 3.0, *args[1:]),
                    kwargs,
                ),
                with_kwargs=True,
            )
            integration.ROSARefinementHead.forward(head, feature_maps, {})
            assert torch.equal(observed[0], torch.full((1, 2, 4), 3.0))
            assert len(head.layers[0]._forward_pre_hooks) == 1
            assert "_rosa_captured_query" not in head.__dict__
            assert "_rosa_capture_via_hook" not in head.__dict__
            local.remove()

            head, observed = make_head()
            def replace_globally(module, args):
                if module is head.layers[0]:
                    return (args[0] + 5.0, *args[1:])
            global_handle = (
                torch.nn.modules.module.register_module_forward_pre_hook(
                    replace_globally
                )
            )
            try:
                integration.ROSARefinementHead.forward(head, feature_maps, {})
            finally:
                global_handle.remove()
            assert torch.equal(observed[0], torch.full((1, 2, 4), 5.0))
            assert not head.layers[0]._forward_pre_hooks

            for capture_via_hook in (False, True):
                head, _ = make_head()
                head._rosa_capture_via_hook = capture_via_hook
                try:
                    head._pre_refine_query_feature(
                        decoder_index=5,
                        instance_feature=torch.zeros(1, 2, 4),
                        anchor=torch.zeros(1, 2, 11),
                        anchor_embed=torch.zeros(1, 2, 4),
                        feature_maps=feature_maps,
                        metas={},
                        predictions=[],
                        classifications=[],
                        qualities=[],
                        num_free_instance=2,
                        classification_feature=torch.zeros(1, 2, 4),
                        regression_feature=None,
                        score_alignment_feature=None,
                    )
                except RuntimeError as error:
                    assert "sidecars must be None" in str(error)
                else:
                    raise AssertionError("native sidecar was accepted")
                finally:
                    head.__dict__.pop("_rosa_capture_via_hook", None)

            head, _ = make_head()
            local = head.layers[0].register_forward_pre_hook(
                lambda *_args: None
            )
            integration.Sparse4DHead.forward = lambda *_args, **_kwargs: (
                (_ for _ in ()).throw(RuntimeError("injected failure"))
            )
            try:
                integration.ROSARefinementHead.forward(head, feature_maps, {})
            except RuntimeError as error:
                assert str(error) == "injected failure"
            else:
                raise AssertionError("injected native failure was swallowed")
            assert len(head.layers[0]._forward_pre_hooks) == 1
            assert "_rosa_captured_query" not in head.__dict__
            assert "_rosa_capture_via_hook" not in head.__dict__
            local.remove()
        finally:
            integration.Sparse4DHead.forward = released_forward
        """
    )
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


def test_minimal_refiner_return_is_sealed_behind_released_inference_abi():
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent(
        """
        import sys
        import types

        import torch

        import rosa.refiner as refiner_module
        from rosa.refiner import ROSAConfig, ROSARefiner


        released_forward = ROSARefiner.forward
        pre_import_calls = []


        def pre_import_forward(
            self,
            baseline_anchor,
            query,
            fpn,
            metas,
            keypoint_generator,
            *,
            materialize_diagnostics=True,
        ):
            pre_import_calls.append({
                "materialize_diagnostics": materialize_diagnostics,
            })
            return released_forward(
                self,
                baseline_anchor,
                query,
                fpn,
                metas,
                keypoint_generator,
                materialize_diagnostics=materialize_diagnostics,
            )


        # Import-order compatibility is the hardest case: an application may
        # patch the public refiner before importing the optional detector bind.
        ROSARefiner.forward = pre_import_forward
        from rosa.integrations import sparse4d as integration
        from rosa.integrations.sparse4d_scores import (
            compose_native_identity_rank,
        )

        assert integration._RELEASED_ROSA_REFINER_FORWARD is released_forward
        assert ROSARefiner.forward is pre_import_forward


        def raw_equal(left, right):
            return bool(
                left.dtype == right.dtype
                and left.shape == right.shape
                and torch.equal(
                    left.detach().contiguous().view(torch.uint8),
                    right.detach().contiguous().view(torch.uint8),
                )
            )


        class DummyDeformable(torch.nn.Module):
            @staticmethod
            def kps_generator(box, _query):
                scale = box.new_tensor(
                    (
                        (0, 0, 0),
                        (0.45, 0, 0),
                        (-0.45, 0, 0),
                        (0, 0.45, 0),
                        (0, -0.45, 0),
                        (0, 0, 0.45),
                        (0, 0, -0.45),
                        (0.2, 0.2, 0.2),
                        (-0.2, 0.2, 0.2),
                        (0.2, -0.2, 0.2),
                        (-0.2, -0.2, 0.2),
                        (0.2, 0.2, -0.2),
                        (-0.2, -0.2, -0.2),
                    )
                )
                return (
                    box[..., :3, None].transpose(-1, -2)
                    + scale[None, None]
                    * box[..., 3:6].exp()[..., None, :]
                )


        batch, queries, cameras, dims = 1, 2, 4, 16
        anchor = torch.zeros(batch, queries, 11)
        anchor[..., 0] = torch.tensor((1.0, 2.0))
        anchor[..., 2] = -5.0
        anchor[..., 3:6] = torch.tensor((1.5, 3.0, 1.5)).log()
        anchor[..., 7] = 1.0
        query = torch.randn(batch, queries, dims)
        fpn = [
            torch.randn(batch, cameras, dims, size, size)
            for size in (8, 4, 2, 1)
        ]
        lidar2cam = (
            torch.eye(4)
            .reshape(1, 1, 4, 4)
            .repeat(batch, cameras, 1, 1)
        )
        intrinsic = (
            torch.eye(3)
            .reshape(1, 1, 3, 3)
            .repeat(batch, cameras, 1, 1)
        )
        intrinsic[..., 0, 2] = 400.0
        intrinsic[..., 1, 2] = 400.0
        polynomial = torch.zeros(batch, cameras, 5)
        polynomial[..., 1] = -150.0
        metas = {
            "lidar2cam": lidar2cam,
            "ocam_intrinsic": intrinsic,
            "ocam_pol": polynomial,
            "image_wh": torch.full((batch, cameras, 2), 800.0),
            "ocam_lens_fov_deg": 205.0,
        }


        def config(profile, refine_topk=queries):
            return ROSAConfig.from_mapping(
                {
                    "profile": profile,
                    "embed_dims": dims,
                    "hidden_dims": 8,
                    "refine_topk": refine_topk,
                }
            )


        def make_head(cfg):
            head = object.__new__(integration.ROSARefinementHead)
            torch.nn.Module.__init__(head)
            head.rosa_cfg = cfg
            head.rosa_refiner = ROSARefiner(cfg, "ocam_fisheye")
            head.layers = torch.nn.ModuleList([DummyDeformable()])
            head._rosa_deformable_layer_index = 0
            head.eval()
            parameter_generator = torch.Generator().manual_seed(20260803)
            with torch.no_grad():
                for parameter in head.rosa_refiner.parameters():
                    parameter.uniform_(
                        -0.05,
                        0.05,
                        generator=parameter_generator,
                    )
            return head


        def native_output():
            classification = torch.tensor(
                (((0.35, -0.4, 0.1), (-0.2, 0.6, 0.25)),),
                dtype=anchor.dtype,
            )
            quality = torch.tensor(
                (((0.2, -0.1), (0.45, 0.3)),),
                dtype=anchor.dtype,
            )
            return {
                "prediction": [anchor.clone() for _ in range(6)],
                "classification": [classification.clone() for _ in range(6)],
                "quality": [quality.clone() for _ in range(6)],
            }


        def run(head):
            output = native_output()
            baseline_classification = output["classification"][5]
            baseline_quality = output["quality"][5]
            with torch.no_grad():
                head._append_rosa(output, query, fpn, metas)
            return output, baseline_classification, baseline_quality


        rank_keys = {
            "frozen_native_label",
            "native_selected_probability",
            "emitted_joint_score",
        }


        def assert_native_identity(head, output, classification, quality):
            record = output[head.aux_record_key]
            assert set(record) == rank_keys
            expected = compose_native_identity_rank(
                classification,
                quality,
                centerness_index=integration.CNS,
            )
            assert raw_equal(
                record["frozen_native_label"],
                expected["frozen_label"],
            )
            assert raw_equal(
                record["native_selected_probability"],
                expected["native_selected_probability"],
            )
            assert raw_equal(
                record["emitted_joint_score"],
                expected["emitted_joint_score"],
            )


        # The pre-import old-signature override must never receive the new
        # private keyword, while identity ranking remains byte-exact.
        pre_import_head = make_head(config("fast"))
        pre_output, pre_cls, pre_quality = run(pre_import_head)
        assert pre_import_calls == [{"materialize_diagnostics": False}]
        assert_native_identity(
            pre_import_head,
            pre_output,
            pre_cls,
            pre_quality,
        )
        ROSARefiner.forward = released_forward

        # An instance-level legacy override receives the historical ABI only.
        # Poisoning its quality delta proves that quality-off ranking ignores
        # the carrier and stays native-identity rather than merely comparing a
        # convenient all-zero result.
        instance_head = make_head(config("fast"))
        instance_calls = []

        def instance_forward(
            self,
            baseline_anchor,
            sidecar_query,
            feature_maps,
            metadata,
            keypoint_generator,
            *,
            materialize_diagnostics=True,
        ):
            instance_calls.append({
                "materialize_diagnostics": materialize_diagnostics,
            })
            result = released_forward(
                self,
                baseline_anchor,
                sidecar_query,
                feature_maps,
                metadata,
                keypoint_generator,
                materialize_diagnostics=materialize_diagnostics,
            )
            result.quality_delta = torch.full_like(result.quality_delta, 31.0)
            assert torch.count_nonzero(result.quality_delta) == (
                result.quality_delta.numel()
            )
            return result

        instance_head.rosa_refiner.forward = types.MethodType(
            instance_forward,
            instance_head.rosa_refiner,
        )
        instance_output, instance_cls, instance_quality = run(instance_head)
        assert instance_calls == [{"materialize_diagnostics": False}]
        assert_native_identity(
            instance_head,
            instance_output,
            instance_cls,
            instance_quality,
        )

        # A hook on the refiner itself opts into the historical observable
        # full record and must not see the private return keyword.
        hooked_head = make_head(config("fast"))
        hooked_kwargs = []

        def observe_refiner(_module, _args, kwargs):
            hooked_kwargs.append(dict(kwargs))

        handle = hooked_head.rosa_refiner.register_forward_pre_hook(
            observe_refiner,
            with_kwargs=True,
        )
        try:
            hooked_output, _, _ = run(hooked_head)
        finally:
            handle.remove()
        assert hooked_kwargs == [{"materialize_diagnostics": True}]
        assert "native_route" in hooked_output[hooked_head.aux_record_key]
        assert set(hooked_output[hooked_head.aux_record_key]) != rank_keys

        # Descendant observation hooks preserve their Tensor relation ABI but
        # do not force the large public carrier back into production output.
        rfl_head = make_head(config("fast"))
        observed_relations = []
        rfl_kwargs = []
        rfl_call_impl = rfl_head.rosa_refiner._call_impl

        def observe_rfl_call(*args, **kwargs):
            rfl_kwargs.append(dict(kwargs))
            return rfl_call_impl(*args, **kwargs)

        rfl_head.rosa_refiner._call_impl = observe_rfl_call

        def observe_relation(_module, args):
            relation = args[1]
            assert torch.is_tensor(relation)
            assert relation.shape[-1] == 19
            observed_relations.append(relation)

        handle = rfl_head.rosa_refiner.rosa.rfl.register_forward_pre_hook(
            observe_relation
        )
        try:
            rfl_output, rfl_cls, rfl_quality = run(rfl_head)
        finally:
            handle.remove()
        assert observed_relations
        assert rfl_kwargs == [
            {
                "materialize_diagnostics": False,
                "return_final_anchor_only": True,
            }
        ]
        assert_native_identity(rfl_head, rfl_output, rfl_cls, rfl_quality)

        # Released quality-off checkpoints still own a legacy quality head.
        # It must execute exactly once even though its result cannot alter the
        # native identity rank.
        legacy_cfg = config("published")
        legacy_cfg = ROSAConfig.from_mapping(
            {
                **legacy_cfg.__dict__,
                "enable_quality_score": False,
            }
        )
        legacy_head = make_head(legacy_cfg)
        assert legacy_head.rosa_refiner.quality_head is not None
        quality_calls = []
        handle = legacy_head.rosa_refiner.quality_head.register_forward_hook(
            lambda _module, _args, _output: quality_calls.append(True)
        )
        try:
            legacy_output, legacy_cls, legacy_quality = run(legacy_head)
        finally:
            handle.remove()
        assert quality_calls == [True]
        assert_native_identity(
            legacy_head,
            legacy_output,
            legacy_cls,
            legacy_quality,
        )

        # Released eager and in-place Module.compile keep the exact public
        # type/forward surface and both delegate the sealed two-key request.
        eager_head = make_head(config("fast"))
        eager_kwargs = []
        eager_call_impl = eager_head.rosa_refiner._call_impl

        def observe_eager_call(*args, **kwargs):
            eager_kwargs.append(dict(kwargs))
            return eager_call_impl(*args, **kwargs)

        eager_head.rosa_refiner._call_impl = observe_eager_call
        eager_output, eager_cls, eager_quality = run(eager_head)
        assert eager_kwargs == [
            {
                "materialize_diagnostics": False,
                "return_final_anchor_only": True,
            }
        ]
        assert_native_identity(
            eager_head,
            eager_output,
            eager_cls,
            eager_quality,
        )

        compiled_head = make_head(config("fast"))
        compiled_head.rosa_refiner.load_state_dict(
            eager_head.rosa_refiner.state_dict()
        )
        assert type(compiled_head.rosa_refiner) is ROSARefiner
        assert "forward" not in compiled_head.rosa_refiner.__dict__
        assert ROSARefiner.forward is released_forward
        compiled_head.rosa_refiner.compile(backend="eager", fullgraph=True)
        assert type(compiled_head.rosa_refiner) is ROSARefiner
        assert "forward" not in compiled_head.rosa_refiner.__dict__
        assert ROSARefiner.forward is released_forward
        compiled_call_impl = compiled_head.rosa_refiner._compiled_call_impl
        compiled_kwargs = []

        def observe_compiled_call(*args, **kwargs):
            compiled_kwargs.append(dict(kwargs))
            return compiled_call_impl(*args, **kwargs)

        compiled_head.rosa_refiner._compiled_call_impl = (
            observe_compiled_call
        )
        compiled_output, compiled_cls, compiled_quality = run(compiled_head)
        assert compiled_kwargs == [
            {
                "materialize_diagnostics": False,
                "return_final_anchor_only": True,
            }
        ]
        assert_native_identity(
            compiled_head,
            compiled_output,
            compiled_cls,
            compiled_quality,
        )
        assert raw_equal(
            eager_output["prediction"][-1],
            compiled_output["prediction"][-1],
        )

        # The exact top-k path shares native sigmoid results and still emits
        # the same three frozen-rank fields.
        selection_head = make_head(config("fast", refine_topk=1))
        shared_code = (
            integration._compose_native_identity_rank_and_selection.__code__
        )
        shared_calls = 0


        def count_shared(frame, event, _argument):
            global shared_calls
            if event == "call" and frame.f_code is shared_code:
                shared_calls += 1


        previous_profile = sys.getprofile()
        sys.setprofile(count_shared)
        try:
            selection_output, selection_cls, selection_quality = run(
                selection_head
            )
        finally:
            sys.setprofile(previous_profile)
        assert shared_calls == 1
        assert_native_identity(
            selection_head,
            selection_output,
            selection_cls,
            selection_quality,
        )

        released_compose = integration.compose_native_identity_rank
        compose_calls = []


        def observed_compose(*args, **kwargs):
            compose_calls.append(True)
            return released_compose(*args, **kwargs)


        integration.compose_native_identity_rank = observed_compose
        fallback_head = make_head(config("fast", refine_topk=1))
        fallback_output, fallback_cls, fallback_quality = run(fallback_head)
        assert compose_calls == [True]
        assert_native_identity(
            fallback_head,
            fallback_output,
            fallback_cls,
            fallback_quality,
        )
        integration.compose_native_identity_rank = released_compose

        released_shared = (
            integration._compose_native_identity_rank_and_selection
        )


        def must_not_run(*_args, **_kwargs):
            raise AssertionError("overridden shared rank helper executed")


        integration._compose_native_identity_rank_and_selection = must_not_run
        private_fallback_head = make_head(config("fast", refine_topk=1))
        private_output, private_cls, private_quality = run(
            private_fallback_head
        )
        assert_native_identity(
            private_fallback_head,
            private_output,
            private_cls,
            private_quality,
        )
        integration._compose_native_identity_rank_and_selection = released_shared
        """
    )
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


def test_released_config_aliases_keep_legacy_projection_default():
    config = ROSAConfig.from_mapping(
        {
            "radial_proposal_mode": "learned_log_gaussian",
            "second_step_candidate_search": False,
            "enable_shape1": True,
            "enable_shape2_feedback": True,
            "max_shape2_log_size_delta": 0.05,
            "max_shape2_yaw_delta_rad": 0.08,
        }
    )
    assert config.projection_contract == "legacy_front_only"
    assert config.enable_shape_refinement
    assert config.radial_proposal_mode == "learned_log_gaussian"


def test_native_refine_capture_forces_and_validates_scores():
    capture = SingleCallCapture(force_return_cls=True)
    args = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    kwargs = {
        "time_interval": None,
        "return_cls": False,
        "classification_feature": None,
        "regression_feature": None,
        "score_alignment_feature": None,
    }
    changed = capture(None, args, kwargs)
    assert changed is not None
    assert changed[1]["return_cls"] is True
    call = capture.take()
    validate_native_call(call)


def test_native_refine_contract_accepts_upstream_minimal_signature():
    call = NativeRefineCall(
        (torch.zeros(1), torch.zeros(1), torch.zeros(1)),
        {"time_interval": None, "return_cls": True},
    )
    validate_native_call(call)

    extended = NativeRefineCall(
        call.args,
        {
            **call.kwargs,
            "classification_feature": None,
            "regression_feature": None,
            "score_alignment_feature": None,
        },
    )
    validate_native_call(extended)

    with pytest.raises(RuntimeError, match="signature changed"):
        validate_native_call(
            NativeRefineCall(
                call.args,
                {**call.kwargs, "unknown_sidecar": None},
            )
        )
    with pytest.raises(RuntimeError, match="signature changed"):
        validate_native_call(
            NativeRefineCall(
                call.args,
                {**call.kwargs, "classification_feature": None},
            )
        )
    with pytest.raises(RuntimeError, match="sidecars must be None"):
        validate_native_call(
            NativeRefineCall(
                extended.args,
                {**extended.kwargs, "classification_feature": torch.zeros(1)},
            )
        )


def test_fpn_detach_accepts_framework_neutral_level_tensors():
    source = [
        torch.randn(2, 4, 8, 5, 7, requires_grad=True),
        torch.randn(2, 4, 8, 3, 4, requires_grad=True),
    ]
    restored = detached_fpn(source)
    assert [value.shape for value in restored] == [value.shape for value in source]
    assert all(not value.requires_grad and value.grad_fn is None for value in restored)


def test_packed_fpn_restore_is_zero_copy_and_sampler_equivalent():
    host = _detector_host()
    if not host.is_dir():
        pytest.skip("optional Sparse4D detector host is not available")
    code = textwrap.dedent(
        """
        import torch

        from projects.mmdet3d_plugin.ops import (
            feature_map_layout_receipt,
            feature_maps_format,
        )
        from rosa.integrations.sparse4d_runtime import detached_fpn, restore_fpn
        from rosa.sampling import (
            DeterministicBilinearSampler,
            FusedBilinearSampler,
        )

        torch.manual_seed(19)
        batch, cameras, channels = 2, 4, 7
        levels = [
            torch.randn(batch, cameras, channels, height, width)
            for height, width in ((9, 11), (5, 7), (3, 4), (2, 2))
        ]
        packed = feature_maps_format(levels)
        level_layout = tuple(tuple(value.shape[-2:]) for value in levels)
        layout = (level_layout,) * cameras
        receipt = feature_map_layout_receipt(packed, layout)
        assert receipt is not None
        restored = restore_fpn(packed, layout_receipt=receipt)
        detached = detached_fpn(packed, layout_receipt=receipt)
        packed_pointer = packed[0].untyped_storage().data_ptr()

        assert len(restored) == len(levels)
        for source, view, detached_view in zip(levels, restored, detached):
            assert torch.equal(view, source)
            assert view.untyped_storage().data_ptr() == packed_pointer
            assert detached_view.untyped_storage().data_ptr() == packed_pointer
            assert not detached_view.requires_grad
            assert detached_view.grad_fn is None
            flattened = view.reshape(
                batch * cameras,
                channels,
                *view.shape[-2:],
            )
            assert flattened.untyped_storage().data_ptr() == packed_pointer

        uv = torch.rand(batch, 6, cameras, 1, 5, 2) * 1.4 - 0.2
        uv[0, 0, 0, 0, 0] = torch.tensor((float("nan"), float("inf")))
        for sampler in (
            FusedBilinearSampler(),
            DeterministicBilinearSampler(),
        ):
            for view in restored:
                contiguous = view.contiguous()
                expected = sampler(contiguous, uv)
                actual = sampler(view, uv)
                assert torch.equal(actual, expected)

        # The production boundary detaches packed FPN features.  Also prove
        # that the deterministic fallback preserves the semantic feature
        # gradient if a future caller intentionally keeps that boundary live.
        sampler = DeterministicBilinearSampler()
        for level_index, restored_level in enumerate(restored):
            contiguous = restored_level.detach().clone().requires_grad_(True)
            packed_values = packed[0].detach().clone().requires_grad_(True)
            packed_level = restore_fpn(
                (packed_values, packed[1], packed[2])
            )[level_index]
            weight = torch.randn(
                batch,
                6,
                cameras,
                1,
                5,
                channels,
            )
            (sampler(contiguous, uv) * weight).sum().backward()
            (sampler(packed_level, uv) * weight).sum().backward()
            packed_gradient = restore_fpn(
                (packed_values.grad, packed[1], packed[2])
            )[level_index]
            assert torch.equal(packed_gradient, contiguous.grad)
        """
    )
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


def test_legacy_checkpoint_prefix_remaps_without_overwriting_modern_key():
    old_weight = torch.tensor([1.0])
    modern_weight = torch.tensor([2.0])
    state = {
        "head.canonical_sixth_block.a": old_weight,
        "head.canonical_sixth_block.b": old_weight,
        "head.rosa_refiner.b": modern_weight,
    }
    consumed = remap_legacy_refiner_state_dict(state, prefix="head.")
    assert consumed == 2
    assert not any("canonical_sixth_block" in key for key in state)
    assert state["head.rosa_refiner.a"] is old_weight
    assert state["head.rosa_refiner.b"] is modern_weight


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_frozen_decode_field_packing_preserves_raw_values_and_ownership(dtype):
    decoded = torch.arange(48, dtype=dtype).reshape(2, 3, 8)
    decoded[0, 0, :4] = torch.tensor(
        (float("nan"), float("inf"), float("-inf"), -0.0),
        dtype=dtype,
    )
    scores = torch.tensor(((0.0, -0.0, 1.0), (2.0, 3.0, 4.0)), dtype=dtype)
    origins = torch.tensor(((5.0, 6.0, 7.0), (8.0, 9.0, 10.0)), dtype=dtype)
    labels = torch.tensor(((0, 1, 2), (3, 4, 5)))
    indices = torch.tensor(((2, 0, 1), (1, 2, 0)))
    instance_ids = torch.tensor(((12, 10, 11), (21, 22, 20)))

    packed = _pack_frozen_decode_fields(
        decoded,
        scores,
        labels,
        indices,
        origins,
        instance_ids,
    )

    expected_fields = {
        "boxes_3d": decoded,
        "scores_3d": scores,
        "labels_3d": labels,
        "query_indices": indices,
        "cls_scores": origins,
        "instance_ids": instance_ids,
    }
    assert len(packed) == 2
    for batch_index, item in enumerate(packed):
        pointers = []
        for name, expected in expected_fields.items():
            actual = item[name]
            assert _raw_equal(actual, expected[batch_index])
            assert actual._base is None
            assert actual.is_contiguous()
            pointers.append(actual.untyped_storage().data_ptr())
        assert len(set(pointers)) == len(pointers)


def test_detector_owned_output_audit_rejects_each_forbidden_mutation():
    anchor = torch.arange(66, dtype=torch.float32).reshape(2, 3, 11)
    classification = torch.arange(36, dtype=torch.float32).reshape(2, 3, 6)
    quality = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    selected = torch.tensor(((0, 2), (0, 2)))
    validate_detector_owned_rosa_output(
        anchor,
        anchor.clone(),
        classification,
        classification.detach(),
        quality,
        quality.detach(),
        selected,
    )

    mutations = []
    changed_classification = classification.clone()
    changed_classification[0, 0, 0] += 1
    mutations.append((anchor.clone(), changed_classification, quality.detach()))
    changed_quality = quality.clone()
    changed_quality[0, 0, 0] += 1
    mutations.append((anchor.clone(), classification.detach(), changed_quality))
    changed_z = anchor.clone()
    changed_z[0, 0, 2] += 1
    mutations.append((changed_z, classification.detach(), quality.detach()))
    changed_velocity = anchor.clone()
    changed_velocity[0, 0, 8] += 1
    mutations.append((changed_velocity, classification.detach(), quality.detach()))
    changed_unselected = anchor.clone()
    changed_unselected[0, 1, 0] += 1
    mutations.append((changed_unselected, classification.detach(), quality.detach()))

    for final_anchor, final_classification, final_quality in mutations:
        with pytest.raises(RuntimeError, match="ROSA changed"):
            validate_detector_owned_rosa_output(
                anchor,
                final_anchor,
                classification,
                final_classification,
                quality,
                final_quality,
                selected,
            )


def test_zero_quality_delta_is_exact_native_joint_score():
    classification = torch.tensor([[[0.5, -0.25], [1.0, 0.1]]])
    quality = torch.tensor([[[0.2], [-0.3]]])
    delta = torch.zeros(1, 2, 1, requires_grad=True)
    supported = torch.ones(1, 2, dtype=torch.bool)
    rank = compose_quality_rank(
        classification,
        quality,
        delta,
        supported,
        centerness_index=0,
    )
    assert torch.equal(
        rank["emitted_joint_score"],
        rank["native_joint_score"],
    )
    stable_joint_probability(
        rank["joint_logit"],
        rank["native_joint_score"],
        rank["effective_delta"],
    ).sum().backward()
    assert delta.grad is not None


@pytest.mark.parametrize(
    ("classification_dtype", "quality_dtype"),
    (
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
        (torch.float16, torch.float32),
        (torch.float32, torch.float64),
    ),
)
def test_native_identity_rank_is_raw_exact_to_zero_quality_residual(
    classification_dtype,
    quality_dtype,
):
    generator = torch.Generator().manual_seed(3407)
    classification = torch.randn(
        2,
        7,
        6,
        generator=generator,
        dtype=classification_dtype,
    )
    quality = torch.randn(
        2,
        7,
        2,
        generator=generator,
        dtype=quality_dtype,
    )
    supported = torch.rand(2, 7, generator=generator) > 0.25
    full = compose_quality_rank(
        classification,
        quality,
        torch.zeros(2, 7, 1, dtype=classification_dtype),
        supported,
        centerness_index=0,
    )
    identity = compose_native_identity_rank(
        classification,
        quality,
        centerness_index=0,
    )

    assert torch.equal(identity["frozen_label"], full["frozen_label"])
    assert identity["native_selected_probability"].dtype == full[
        "native_selected_probability"
    ].dtype
    assert _raw_equal(
        identity["native_selected_probability"],
        full["native_selected_probability"],
    )
    assert identity["emitted_joint_score"].dtype == full[
        "emitted_joint_score"
    ].dtype
    assert _raw_equal(
        identity["emitted_joint_score"],
        full["emitted_joint_score"],
    )


@pytest.mark.parametrize(
    ("classification_dtype", "quality_dtype"),
    (
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
        (torch.float16, torch.float32),
        (torch.float32, torch.float64),
    ),
)
def test_shared_native_rank_selection_is_raw_exact_for_special_values_and_ties(
    classification_dtype,
    quality_dtype,
):
    generator = torch.Generator().manual_seed(7_101_080)
    classification = torch.randn(
        2,
        7,
        6,
        generator=generator,
        dtype=classification_dtype,
    )
    quality = torch.randn(
        2,
        7,
        2,
        generator=generator,
        dtype=quality_dtype,
    )
    bit_patterns = {
        torch.float16: (
            torch.uint16,
            (0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x7FFF),
        ),
        torch.bfloat16: (
            torch.uint16,
            (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7FFF),
        ),
        torch.float32: (
            torch.uint32,
            (
                0x00000000,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC00001,
                0x7FFFFFFF,
            ),
        ),
        torch.float64: (
            torch.uint64,
            (
                0x0000000000000000,
                0x8000000000000000,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x7FF8000000000001,
                0x7FFFFFFFFFFFFFFF,
            ),
        ),
    }

    for value in (classification, quality):
        integer_dtype, patterns = bit_patterns[value.dtype]
        value.view(integer_dtype).reshape(-1)[: len(patterns)].copy_(
            torch.tensor(patterns, dtype=integer_dtype)
        )
    if classification_dtype is torch.float16:
        tie = (-11.9765625, -11.96875)
    elif classification_dtype is torch.bfloat16:
        tie = (-1.90625, -1.8984375)
    else:
        tie = (1.0, 1.0)
    classification[1, 0].fill_(-20.0)
    classification[1, 0, :2] = classification.new_tensor(tie)
    classification_before = classification.clone()
    quality_before = quality.clone()

    expected_selection = (
        classification.detach().sigmoid().amax(dim=-1)
        * quality[..., 0].detach().sigmoid()
    )
    expected_rank = compose_native_identity_rank(
        classification,
        quality,
        centerness_index=0,
    )
    expected_indices = expected_selection.topk(3, dim=1, sorted=False).indices
    actual_rank, actual_selection = _compose_native_identity_rank_and_selection(
        classification,
        quality,
        centerness_index=0,
    )
    actual_indices = actual_selection.topk(3, dim=1, sorted=False).indices

    assert _raw_equal(actual_selection, expected_selection)
    assert actual_selection.stride() == expected_selection.stride()
    assert torch.equal(actual_indices, expected_indices)
    for name in expected_rank:
        assert actual_rank[name].dtype == expected_rank[name].dtype, name
        assert actual_rank[name].shape == expected_rank[name].shape, name
        assert actual_rank[name].stride() == expected_rank[name].stride(), name
        assert _raw_equal(actual_rank[name], expected_rank[name]), name
    assert _raw_equal(classification, classification_before)
    assert _raw_equal(quality, quality_before)
    assert int(actual_rank["frozen_label"][1, 0]) == 0


def test_shared_native_rank_selection_guard_is_plain_eager_only():
    classification = torch.randn(2, 7, 6)
    quality = torch.randn(2, 7, 2)

    assert not _can_reuse_native_identity_rank_selection(
        classification,
        quality,
        centerness_index=0,
    )
    with torch.no_grad():
        assert _can_reuse_native_identity_rank_selection(
            classification,
            quality,
            centerness_index=0,
        )
        assert not _can_reuse_native_identity_rank_selection(
            classification.transpose(1, 2),
            quality,
            centerness_index=0,
        )
        assert not _can_reuse_native_identity_rank_selection(
            classification,
            quality,
            centerness_index=2,
        )

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            del types
            return func(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            del types
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), FunctionMode():
        assert not _can_reuse_native_identity_rank_selection(
            classification,
            quality,
            centerness_index=0,
        )
    with torch.no_grad(), DispatchMode():
        assert not _can_reuse_native_identity_rank_selection(
            classification,
            quality,
            centerness_index=0,
        )


@pytest.mark.parametrize(
    ("owner", "name"),
    (
        (sparse4d_scores_module, "compose_native_identity_rank"),
        (
            sparse4d_scores_module,
            "_compose_native_identity_rank_and_selection",
        ),
        (sparse4d_scores_module, "_probability_dtype"),
        (rfl_module, "_plain_eager_no_grad_execution"),
        (torch, "gather"),
        (torch.Tensor, "detach"),
        (torch.Tensor, "sigmoid"),
        (torch.Tensor, "amax"),
        (torch.Tensor, "argmax"),
        (torch.Tensor, "max"),
        (torch.Tensor, "squeeze"),
        (torch.Tensor, "to"),
        (torch.Tensor, "__mul__"),
        (torch.Tensor, "__getitem__"),
    ),
)
def test_shared_native_rank_selection_guard_rejects_primitive_override(
    monkeypatch,
    owner,
    name,
):
    classification = torch.randn(2, 7, 6)
    quality = torch.randn(2, 7, 2)
    monkeypatch.setattr(owner, name, lambda *_args, **_kwargs: None)
    with torch.no_grad():
        assert not _can_reuse_native_identity_rank_selection(
            classification,
            quality,
            centerness_index=0,
        )


class _IdentityBoxDecoder:
    selection_score_mode = "legacy"
    sorted = True
    num_output = 3
    score_threshold = None

    @staticmethod
    def decode_box(boxes):
        return boxes


def test_packed_decode_fallback_never_speculates_or_repeats_decoder_calls():
    class CountingDecoder(_IdentityBoxDecoder):
        def __init__(self):
            self.calls = 0

        def decode_box(self, boxes):
            self.calls += 1
            return boxes

    decoder = CountingDecoder()
    classification = torch.randn(2, 3, 2)
    boxes = torch.randn(2, 3, 11)
    labels = torch.tensor(((0, 1, 0), (1, 0, 1)))
    class_probability = torch.rand(2, 3)
    joint_score = torch.rand(2, 3)
    instance_ids = torch.tensor(((10, 11, 12), (20, 21, 22)))

    result = decode_frozen_jointscore_rank(
        decoder,
        [classification],
        [boxes],
        labels,
        class_probability,
        joint_score,
        instance_id=instance_ids,
        output_idx=0,
        validate_values=False,
        pack_host_transfers=True,
    )

    assert len(result) == 2
    assert decoder.calls == 2


def test_packed_decode_accepts_empty_batch_without_decoder_calls():
    class CountingDecoder(_IdentityBoxDecoder):
        def __init__(self):
            self.calls = 0

        def decode_box(self, boxes):
            self.calls += 1
            return boxes

    decoder = CountingDecoder()
    result = decode_frozen_jointscore_rank(
        decoder,
        [torch.empty(0, 3, 2)],
        [torch.empty(0, 3, 11)],
        torch.empty(0, 3, dtype=torch.long),
        torch.empty(0, 3),
        torch.empty(0, 3),
        instance_id=torch.empty(0, 3, dtype=torch.long),
        output_idx=0,
        validate_values=False,
        pack_host_transfers=True,
    )

    assert result == []
    assert decoder.calls == 0


def test_quality_residual_changes_only_query_ranking_at_inference():
    classification = torch.tensor(
        [[[2.0, -1.0], [-1.0, 1.5], [1.0, -2.0]]]
    )
    native_quality = torch.zeros(1, 3, 1)
    boxes = torch.arange(33, dtype=torch.float32).reshape(1, 3, 11)
    instance_id = torch.tensor(((10, 20, 30),))
    supported = torch.ones(1, 3, dtype=torch.bool)
    classification_before = classification.clone()
    quality_before = native_quality.clone()

    native = compose_quality_rank(
        classification,
        native_quality,
        torch.zeros(1, 3, 1),
        supported,
        centerness_index=0,
    )
    reranked = compose_quality_rank(
        classification,
        native_quality,
        torch.tensor([[[0.0], [1.0], [0.0]]]),
        supported,
        centerness_index=0,
    )
    decoder = _IdentityBoxDecoder()
    native_result = decode_frozen_jointscore_rank(
        decoder,
        [classification],
        [boxes],
        native["frozen_label"],
        native["native_selected_probability"],
        native["emitted_joint_score"],
        instance_id=instance_id,
        output_idx=0,
    )[0]
    reranked_result = decode_frozen_jointscore_rank(
        decoder,
        [classification],
        [boxes],
        reranked["frozen_label"],
        reranked["native_selected_probability"],
        reranked["emitted_joint_score"],
        instance_id=instance_id,
        output_idx=0,
    )[0]

    assert torch.equal(classification, classification_before)
    assert torch.equal(native_quality, quality_before)
    assert not torch.equal(
        native_result["query_indices"],
        reranked_result["query_indices"],
    )
    for query_index in range(3):
        native_position = (
            native_result["query_indices"] == query_index
        ).nonzero(as_tuple=True)[0]
        reranked_position = (
            reranked_result["query_indices"] == query_index
        ).nonzero(as_tuple=True)[0]
        assert torch.equal(
            native_result["boxes_3d"][native_position],
            reranked_result["boxes_3d"][reranked_position],
        )
        assert torch.equal(
            native_result["labels_3d"][native_position],
            reranked_result["labels_3d"][reranked_position],
        )
        assert torch.equal(
            native_result["cls_scores"][native_position],
            reranked_result["cls_scores"][reranked_position],
        )
        assert torch.equal(
            native_result["instance_ids"][native_position],
            reranked_result["instance_ids"][reranked_position],
        )
