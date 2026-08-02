"""Sparse4D host binding for detector-agnostic :class:`ROSARefiner`.

This module is optional.  It is imported only by an MMEngine config running
the original detector host; the ordinary ``import rosa`` path has no
MMDetection3D or Sparse4D dependency.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    from mmdet3d.registry import MODELS
    from projects.mmdet3d_plugin.core.box3d import (
        CNS,
        COS_YAW,
        SIN_YAW,
        YNS,
    )
    from projects.mmdet3d_plugin.models.blocks import (
        DeformableFeatureAggregation,
    )
    from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import (
        SparseBox3DKeyPointsGenerator,
    )
    from projects.mmdet3d_plugin.models.detection3d.losses import (
        SparseBox3DLoss,
    )
    from projects.mmdet3d_plugin.models.sparse4d_head import Sparse4DHead
except ImportError as exc:  # pragma: no cover - exercised in host environment
    raise ImportError(
        "rosa.integrations.sparse4d requires MMDetection3D and the Sparse4D "
        "plugin. Core ROSA remains available via `from rosa import "
        "ROSAConfig, ROSARefiner`."
    ) from exc

from ..geometry import ocam_project_points
from ..refiner import ROSAConfig, ROSARefiner
from .sparse4d_losses import (
    evaluator_aligned_centerness_target,
    rosa_losses,
    validate_centerness_target_mode,
)
from .sparse4d_runtime import (
    SingleCallCapture,
    detached_fpn,
    find_operation_indices,
    remap_legacy_refiner_state_dict,
    validate_detector_owned_rosa_output,
    validate_native_call,
)
from .sparse4d_scores import (
    compose_native_identity_rank,
    compose_quality_rank,
    decode_frozen_jointscore_rank,
)

_NATIVE_DECODERS = 6
_NATIVE_FINAL = 5
_ROSA_OUTPUT = 6
_RELEASED_MHA_FORWARD = torch.nn.MultiheadAttention.forward
_RELEASED_LINEAR_FORWARD = torch.nn.Linear.forward
_HOST_CONTROL_DEFAULTS = {
    "max_temporal_gnn_interactions": 0,
    "cache_decoder_index": _NATIVE_FINAL,
    "inference_output_decoder_index": _NATIVE_FINAL,
    "inference_classification_decoder_index": _NATIVE_FINAL,
    "inference_quality_decoder_index": _NATIVE_FINAL,
    "inference_box_source_by_dimension": None,
}
_EXPECTED_OPERATION_ORDER = (
    "deformable",
    "ffn",
    "norm",
    "refine",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
)


def _normalized_yaw_geometry(anchor: Tensor) -> Tensor:
    """Normalize the sin/cos pair used for geometry without changing boxes."""

    yaw = anchor[..., [SIN_YAW, COS_YAW]]
    finite = torch.isfinite(yaw).all(dim=-1, keepdim=True)
    safe_yaw = torch.where(finite, yaw, torch.zeros_like(yaw))
    norm = torch.linalg.vector_norm(safe_yaw, dim=-1, keepdim=True)
    fallback = torch.zeros_like(safe_yaw)
    fallback[..., 1] = 1.0
    normalized = torch.where(
        finite & norm.gt(1e-6),
        safe_yaw / norm.clamp_min(1e-6),
        fallback,
    )
    geometry_anchor = anchor.clone()
    geometry_anchor[..., SIN_YAW] = normalized[..., 0]
    geometry_anchor[..., COS_YAW] = normalized[..., 1]
    return geometry_anchor


@MODELS.register_module()
class ROSANormalizedKeyPointsGenerator(SparseBox3DKeyPointsGenerator):
    """Generate gravity-centred keypoints from a unit yaw direction.

    Sparse4D's yaw two-vector is not guaranteed to have unit norm. The raw
    pair therefore scales every non-central keypoint by ``||yaw||``. Only a
    geometry copy is normalized here; detector state and regression targets
    remain untouched.
    """

    def __init__(self, *args, anchor_z_is_bottom: bool = False, **kwargs) -> None:
        super().__init__(
            *args,
            anchor_z_is_bottom=anchor_z_is_bottom,
            **kwargs,
        )

    def forward(self, anchor: Tensor, *args, **kwargs):
        return super().forward(
            _normalized_yaw_geometry(anchor),
            *args,
            **kwargs,
        )


@MODELS.register_module()
class ROSAFullFOVDeformableFeatureAggregation(DeformableFeatureAggregation):
    """Use the declared OCam lens cone in every native decoder block."""

    camera_transform = "homogeneous_4x4"
    inference_storage_reuse = False

    def __init__(self, *args, projection_type: str = "ocam_fisheye", **kwargs):
        if projection_type != "ocam_fisheye":
            raise ValueError(
                "ROSAFullFOVDeformableFeatureAggregation requires "
                "projection_type='ocam_fisheye'"
            )
        super().__init__(
            *args,
            projection_type=projection_type,
            **kwargs,
        )

    def project_points(
        self,
        key_points: Tensor,
        projection_mat: Tensor,
        image_wh: Tensor | None = None,
        metas: dict | None = None,
    ) -> Tensor:
        if metas is None or image_wh is None:
            raise ValueError("full-FOV OCam projection requires metadata and image_wh")
        return ocam_project_points(
            key_points,
            metas,
            image_wh,
            contract="full_fov",
            camera_transform=self.camera_transform,
            inference_storage_reuse=self.inference_storage_reuse,
        )


@MODELS.register_module()
class ROSAMetricGatedAffine3x3FullFOVDeformableFeatureAggregation(
    ROSAFullFOVDeformableFeatureAggregation
):
    """Evaluation-only affine port pending official Main126 metric parity."""

    camera_transform = "affine_3x3"

    def project_points(
        self,
        key_points: Tensor,
        projection_mat: Tensor,
        image_wh: Tensor | None = None,
        metas: dict | None = None,
    ) -> Tensor:
        if self.training:
            raise RuntimeError(
                "the affine 3x3 OCam port is evaluation-only until official "
                "Main126 metric parity is proven"
            )
        return super().project_points(
            key_points,
            projection_mat,
            image_wh=image_wh,
            metas=metas,
        )


@MODELS.register_module()
class ROSAMetricGatedInferenceReuseAffine3x3FullFOVDeformableFeatureAggregation(
    ROSAMetricGatedAffine3x3FullFOVDeformableFeatureAggregation
):
    """Reuse projection storage in the no-grad affine evaluation arm."""

    inference_storage_reuse = True


@MODELS.register_module()
class ROSAInferenceReuseExactHomogeneous4x4FullFOVDeformableFeatureAggregation(
    ROSAFullFOVDeformableFeatureAggregation
):
    """Exact homogeneous OCam projection with no-grad owned-storage reuse."""

    inference_storage_reuse = True

    def project_points(
        self,
        key_points: Tensor,
        projection_mat: Tensor,
        image_wh: Tensor | None = None,
        metas: dict | None = None,
    ) -> Tensor:
        if self.training:
            raise RuntimeError(
                "the exact homogeneous storage-reuse path is evaluation-only"
            )
        if torch.is_grad_enabled():
            raise RuntimeError(
                "exact homogeneous OCam storage reuse requires "
                "torch.no_grad() or torch.inference_mode()"
            )
        return super().project_points(
            key_points,
            projection_mat,
            image_wh=image_wh,
            metas=metas,
        )


@MODELS.register_module()
class ROSAEvaluatorAlignedSparseBox3DLoss(SparseBox3DLoss):
    """Sparse4D box loss with a detached BEV-centerness target."""

    def __init__(
        self,
        *args,
        centerness_target_mode: str = "exp_bev",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.centerness_target_mode = validate_centerness_target_mode(
            centerness_target_mode
        )

    def forward(
        self,
        box: Tensor,
        box_target: Tensor,
        weight: Tensor | None = None,
        avg_factor=None,
        suffix: str = "",
        quality: Tensor | None = None,
        cls_target: Tensor | None = None,
        **kwargs,
    ):
        output = super().forward(
            box,
            box_target,
            weight=weight,
            avg_factor=avg_factor,
            suffix=suffix,
            quality=None,
            cls_target=cls_target,
            **kwargs,
        )
        if quality is None:
            return output
        centerness = quality[..., CNS]
        centerness_target = evaluator_aligned_centerness_target(
            box,
            box_target,
            mode=self.centerness_target_mode,
        )
        output[f"loss_cns{suffix}"] = self.loss_cns(
            centerness,
            centerness_target,
            avg_factor=avg_factor,
        )
        yawness = quality[..., YNS].sigmoid()
        yawness_target = (
            F.cosine_similarity(
                box_target[..., [SIN_YAW, COS_YAW]],
                box[..., [SIN_YAW, COS_YAW]],
                dim=-1,
            )
            > 0
        ).to(yawness.dtype)
        output[f"loss_yns{suffix}"] = self.loss_yns(
            yawness,
            yawness_target,
            avg_factor=avg_factor,
        )
        return output


@MODELS.register_module()
class ROSARefinementHead(Sparse4DHead):
    """Append one ROSA output after six untouched Sparse4D decoders.

    Published checkpoints and results use the ``legacy_front_only`` OCam
    projection contract.  The corrected ``full_fov`` mode is an explicit
    retraining/evaluation choice, never an implicit compatibility upgrade.
    """

    aux_record_key = "rosa_refinement_aux"
    requires_feature_map_layout_receipt = True

    def __init__(
        self,
        rosa_cfg: Mapping | None = None,
        closed_loop_rosa_cfg: Mapping | None = None,
        canonical_task_splice_cfg: Mapping | None = None,
        **kwargs,
    ) -> None:
        if rosa_cfg is not None and closed_loop_rosa_cfg is not None:
            raise ValueError("set either rosa_cfg or closed_loop_rosa_cfg, not both")
        # ``canonical_task_splice_cfg`` existed only because the old class
        # inherited two unrelated Canonical Surface heads.  Accept the field
        # so published configs still build, but never instantiate that stack.
        compatibility_cfg = dict(canonical_task_splice_cfg or {})
        config = ROSAConfig.from_mapping(
            rosa_cfg if rosa_cfg is not None else closed_loop_rosa_cfg
        )
        host_controls = {
            name: kwargs.pop(name, default)
            for name, default in _HOST_CONTROL_DEFAULTS.items()
        }
        Sparse4DHead.__init__(self, **kwargs)
        for name, value in host_controls.items():
            setattr(self, name, value)
        self._validate_native_topology()

        if self.embed_dims != config.embed_dims:
            raise ValueError("ROSA width must match Sparse4D embed_dims")
        deformable_indices = find_operation_indices(self.operation_order, "deformable")
        self._validate_anchor_z_contract(config, deformable_indices)
        refine_indices = find_operation_indices(self.operation_order, "refine")
        self._rosa_deformable_layer_index = deformable_indices[-1]
        self._native_refine_layer_index = refine_indices[_NATIVE_FINAL]
        deformable = self.layers[self._rosa_deformable_layer_index]
        if int(deformable.num_pts) != config.num_keypoints:
            raise ValueError(
                "final Sparse4D deformable layer must generate 13 keypoints"
            )
        if (
            int(deformable.num_cams) != config.num_cameras
            or int(deformable.num_levels) != config.num_levels
            or int(deformable.embed_dims) != config.embed_dims
        ):
            raise ValueError(
                "final Sparse4D deformable layer disagrees with ROSA "
                "camera/level/feature dimensions"
            )
        projection_type = str(deformable.projection_type)
        if projection_type != "ocam_fisheye":
            raise ValueError(
                "this ROSA release binds only native OCam projection, got "
                f"{projection_type!r}"
            )
        legacy_projection = compatibility_cfg.get("projection_type")
        if legacy_projection is not None and str(legacy_projection) != projection_type:
            raise ValueError("legacy host config projection disagrees with Sparse4D")

        self.rosa_cfg = config
        if config.architecture_profile == "compact":
            # A sidecar must not advance the native detector's construction
            # RNG stream. This keeps paired B0/ROSA initialization exact.
            with torch.random.fork_rng(devices=[]):
                self.rosa_refiner = ROSARefiner(
                    config,
                    projection_type=projection_type,
                )
        else:
            self.rosa_refiner = ROSARefiner(
                config,
                projection_type=projection_type,
            )
        self.inference_output_decoder_index = _ROSA_OUTPUT
        self.inference_classification_decoder_index = _ROSA_OUTPUT
        self.inference_quality_decoder_index = _ROSA_OUTPUT

    @property
    def canonical_sixth_block(self) -> ROSARefiner:
        """Read-only source compatibility; not a registered module name."""

        return self.rosa_refiner

    @property
    def closed_loop_cfg(self) -> ROSAConfig:
        """Read-only compatibility for released analysis utilities."""

        return self.rosa_cfg

    def _lean_inference(self) -> bool:
        """Select the sealed latency path only on ordinary no-grad eval."""

        observed_modules = (self, self.rosa_refiner)
        hooks_active = bool(
            any(
                module._forward_pre_hooks
                or module._forward_hooks
                or module._backward_pre_hooks
                or module._backward_hooks
                for module in observed_modules
            )
            or torch.nn.modules.module._global_forward_pre_hooks
            or torch.nn.modules.module._global_forward_hooks
            or torch.nn.modules.module._global_backward_pre_hooks
            or torch.nn.modules.module._global_backward_hooks
        )
        return bool(
            not self.training
            and not torch.is_grad_enabled()
            and not self.rosa_cfg.export_diagnostics
            and not hooks_active
        )

    def _validate_native_topology(self) -> None:
        if int(self.num_decoder) != _NATIVE_DECODERS:
            raise ValueError("ROSA requires exactly six native decoders")
        if tuple(self.operation_order) != _EXPECTED_OPERATION_ORDER:
            raise ValueError("Sparse4D operation_order must end at native D5")
        if int(self.instance_bank.num_anchor) != 300:
            raise ValueError("the released ROSA run requires 300 queries")
        if int(self.instance_bank.num_temp_instances) != 0:
            raise ValueError("the released ROSA run is temporal-free")
        if self.max_temporal_gnn_interactions != 0:
            raise ValueError("temporal GNN interactions must be disabled")
        if self.sampler is None:
            raise ValueError("Sparse4D Hungarian sampler is required")
        if int(self.sampler.num_dn_groups) != 5:
            raise ValueError("five native denoising groups are required")
        if int(self.sampler.num_temp_dn_groups) != 0:
            raise ValueError("temporal denoising must be disabled")
        if (
            int(self.cache_decoder_index) != _NATIVE_FINAL
            or int(self.inference_output_decoder_index) != _NATIVE_FINAL
            or int(self.inference_classification_decoder_index) != _NATIVE_FINAL
            or int(self.inference_quality_decoder_index) != _NATIVE_FINAL
        ):
            raise ValueError("constructor routes must initially select native D5")
        if self.inference_box_source_by_dimension is not None:
            raise ValueError("dimension-wise decoder routing is forbidden")

    def _validate_anchor_z_contract(
        self,
        config: ROSAConfig,
        deformable_indices: list[int],
    ) -> None:
        """Keep every image sampler on the detector state's z convention."""

        expected = bool(config.anchor_z_is_bottom)
        generators = [
            getattr(self.instance_bank, "anchor_handler", None),
            *[
                getattr(self.layers[index], "kps_generator", None)
                for index in deformable_indices
            ],
        ]
        if any(generator is None for generator in generators):
            raise ValueError("all native samplers require a keypoint generator")
        conventions = [
            bool(getattr(generator, "anchor_z_is_bottom", False))
            for generator in generators
        ]
        if any(convention != expected for convention in conventions):
            raise ValueError(
                "instance-bank and all six native samplers must share "
                "ROSA anchor_z_is_bottom"
            )

    def init_weights(self) -> None:
        Sparse4DHead.init_weights(self)
        if self.rosa_cfg.architecture_profile == "compact":
            # Refiner initialization is reproducible while leaving the native
            # dropout RNG stream bitwise paired with the sidecar-free B0 arm.
            with torch.random.fork_rng(devices=[]):
                self.rosa_refiner.reset_parameters()
        else:
            self.rosa_refiner.reset_parameters()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Remap released ``canonical_sixth_block.*`` checkpoint keys."""

        remap_legacy_refiner_state_dict(state_dict, prefix=prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _native_output(output: dict) -> tuple[Tensor, Tensor, Tensor]:
        lists = [
            output.get(name) for name in ("prediction", "classification", "quality")
        ]
        if any(
            not isinstance(value, (list, tuple)) or len(value) != _NATIVE_DECODERS
            for value in lists
        ):
            raise RuntimeError("native outputs must contain exactly D0-D5")
        anchor = lists[0][_NATIVE_FINAL]
        classification = lists[1][_NATIVE_FINAL]
        quality = lists[2][_NATIVE_FINAL]
        if classification is None or quality is None:
            raise RuntimeError("native D5 class/quality must materialize")
        return anchor, classification, quality

    def _pre_refine_query_feature(
        self,
        *,
        decoder_index: int,
        instance_feature: Tensor,
        anchor: Tensor,
        anchor_embed: Tensor,
        feature_maps,
        metas: dict,
        predictions,
        classifications,
        qualities,
        num_free_instance: int,
        classification_feature: Tensor | None,
        regression_feature: Tensor | None,
        score_alignment_feature: Tensor | None,
    ) -> Tensor:
        """Capture D5 through the host's identity extension point."""

        query = super()._pre_refine_query_feature(
            decoder_index=decoder_index,
            instance_feature=instance_feature,
            anchor=anchor,
            anchor_embed=anchor_embed,
            feature_maps=feature_maps,
            metas=metas,
            predictions=predictions,
            classifications=classifications,
            qualities=qualities,
            num_free_instance=num_free_instance,
            classification_feature=classification_feature,
            regression_feature=regression_feature,
            score_alignment_feature=score_alignment_feature,
        )
        if int(decoder_index) == _NATIVE_FINAL:
            populated_sidecars = {
                name
                for name, value in (
                    ("classification_feature", classification_feature),
                    ("regression_feature", regression_feature),
                    ("score_alignment_feature", score_alignment_feature),
                )
                if value is not None
            }
            if populated_sidecars:
                raise RuntimeError(
                    "native refine sidecars must be None at the audited "
                    f"call boundary: {sorted(populated_sidecars)}"
                )
        if int(decoder_index) == _NATIVE_FINAL and not self.__dict__.get(
            "_rosa_capture_via_hook",
            False,
        ):
            if self.__dict__.get("_rosa_captured_query") is not None:
                raise RuntimeError("native D5 query was captured more than once")
            self._rosa_captured_query = query
        return query

    def _append_rosa(
        self,
        output: dict,
        native_query: Tensor,
        live_fpn: list[Tensor],
        metas: dict,
    ) -> None:
        lean_inference = self._lean_inference()
        native_identity_rank = bool(
            lean_inference and not self.rosa_cfg.enable_quality_score
        )
        baseline_anchor, baseline_cls, baseline_quality = self._native_output(output)
        num_free = int(baseline_anchor.shape[1])
        if (
            not torch.is_tensor(native_query)
            or native_query.shape[0] != baseline_anchor.shape[0]
            or native_query.shape[1] < num_free
        ):
            raise RuntimeError("captured query and D5 proposal axes disagree")
        query = native_query[:, :num_free]
        refine_count = min(int(self.rosa_cfg.refine_topk), num_free)
        if refine_count < num_free:
            with torch.no_grad():
                selection_score = (
                    baseline_cls.detach().sigmoid().amax(dim=-1)
                    * baseline_quality[..., CNS].detach().sigmoid()
                )
                selected_indices = selection_score.topk(
                    refine_count, dim=1, sorted=False
                ).indices
            anchor_index = selected_indices[..., None].expand(
                -1, -1, baseline_anchor.shape[-1]
            )
            query_index = selected_indices[..., None].expand(-1, -1, query.shape[-1])
            stage_anchor = baseline_anchor.gather(1, anchor_index)
            stage_query = query.gather(1, query_index)
        else:
            selected_indices = None
            anchor_index = None
            stage_anchor = baseline_anchor
            stage_query = query

        deformable = self.layers[self._rosa_deformable_layer_index]
        required_metadata = (
            "lidar2cam",
            "image_wh",
            "ocam_intrinsic",
            "ocam_pol",
        )
        missing_metadata = [
            name for name in required_metadata if name not in metas
        ]
        if missing_metadata:
            raise KeyError(
                f"ROSA metadata missing: {missing_metadata}"
            )
        rosa_metadata = {
            name: metas[name] for name in required_metadata
        }
        if "ocam_lens_fov_deg" in metas:
            rosa_metadata["ocam_lens_fov_deg"] = metas[
                "ocam_lens_fov_deg"
            ]
        result = self.rosa_refiner(
            stage_anchor,
            stage_query,
            live_fpn,
            rosa_metadata,
            deformable.kps_generator,
            materialize_diagnostics=not lean_inference,
            return_final_anchor_only=native_identity_rank,
        )
        if native_identity_rank:
            if not torch.is_tensor(result):
                raise RuntimeError("lean ROSA refiner did not return its final anchor")
            refined_anchor = result
        else:
            if torch.is_tensor(result):
                raise RuntimeError("full ROSA refiner omitted its output record")
            refined_anchor = result.final_anchor
        if selected_indices is None:
            final_anchor = refined_anchor
            if not native_identity_rank:
                quality_delta = result.quality_delta
                supported = result.candidate_supported.any(dim=2)
        else:
            final_anchor = baseline_anchor.detach().scatter(
                1, anchor_index, refined_anchor
            )
            if not native_identity_rank:
                scalar_index = selected_indices[..., None]
                quality_delta = torch.zeros_like(baseline_cls[..., :1]).scatter(
                    1, scalar_index, result.quality_delta
                )
                supported = torch.zeros(
                    baseline_anchor.shape[:2],
                    dtype=torch.bool,
                    device=baseline_anchor.device,
                ).scatter(
                    1,
                    selected_indices,
                    result.candidate_supported.any(dim=2),
                )
        if native_identity_rank:
            rank = compose_native_identity_rank(
                baseline_cls,
                baseline_quality,
                centerness_index=CNS,
            )
        else:
            rank = compose_quality_rank(
                baseline_cls,
                baseline_quality,
                quality_delta,
                supported,
                centerness_index=CNS,
                validate_values=not lean_inference,
            )
        final_cls = baseline_cls.detach()
        final_quality = baseline_quality.detach()
        if not lean_inference:
            validate_detector_owned_rosa_output(
                baseline_anchor,
                final_anchor,
                baseline_cls,
                final_cls,
                baseline_quality,
                final_quality,
                selected_indices,
            )

        output["prediction"] = [*output["prediction"], final_anchor]
        output["classification"] = [*output["classification"], final_cls]
        output["quality"] = [*output["quality"], final_quality]
        if lean_inference:
            # Post-processing needs only these three tensors.  Keeping the
            # full training/analysis record here would retain every candidate
            # observation until decoding even though none is exported.
            output[self.aux_record_key] = {
                "frozen_native_label": rank["frozen_label"],
                "native_selected_probability": rank[
                    "native_selected_probability"
                ],
                "emitted_joint_score": rank["emitted_joint_score"],
            }
            return
        record = {
            "native_route": _NATIVE_FINAL,
            "rosa_route": _ROSA_OUTPUT,
            "baseline_anchor": baseline_anchor.detach(),
            "baseline_classification": baseline_cls.detach(),
            "baseline_quality": baseline_quality.detach(),
            "final_anchor": final_anchor,
            "final_classification": final_cls,
            "final_quality": final_quality,
            "shape1_anchor": result.shape1_anchor,
            "shape1_size_delta": result.shape1_size_delta,
            "shape1_yaw_delta": result.shape1_yaw_delta,
            "shape2_size_delta": result.shape2_size_delta,
            "shape2_yaw_delta": result.shape2_yaw_delta,
            "tangent_delta": result.tangent_delta,
            "radial_raw": result.radial_raw,
            "radial_gate": result.radial_gate,
            "radial_safe": result.radial_safe,
            "candidate_probabilities": result.candidate_probabilities,
            "candidate_logits": result.candidate_logits,
            "candidate_supported": result.candidate_supported,
            "posterior_entropy": result.posterior_entropy,
            "posterior_margin": result.posterior_margin,
            "supported": result.initial_supported,
            "quality_delta": result.quality_delta,
            "frozen_native_label": rank["frozen_label"],
            "native_selected_probability": rank["native_selected_probability"],
            "native_joint_score": rank["native_joint_score"],
            "native_joint_logit": rank["native_joint_logit"],
            "joint_logit": rank["joint_logit"],
            "effective_jointscore_delta": rank["effective_delta"],
            "emitted_joint_score": rank["emitted_joint_score"],
            "jointscore_eligible": rank["eligible"],
            "native_extra_decoder_executed": False,
            "rosa_dn_executed": False,
            "production_gt_access": False,
            "projection_contract": self.rosa_cfg.projection_contract,
        }
        if selected_indices is not None:
            record["selected_indices"] = selected_indices.detach()
        for name in (
            "distribution_mean",
            "distribution_std",
            "candidate_offsets",
            "iteration_base_anchors",
            "iteration_anchors",
            "iteration_distribution_mean",
            "iteration_distribution_std",
            "iteration_candidate_offsets",
            "iteration_candidate_probabilities",
            "iteration_candidate_supported",
            "iteration_observation_supported",
            "iteration_radial_proposal",
            "iteration_radial_gate",
            "iteration_radial_gate_logit",
            "iteration_radial_safe",
            "iteration_tangent_delta",
            "iteration_yaw_raw_delta",
            "iteration_yaw_gate",
            "iteration_yaw_gate_logit",
            "iteration_yaw_delta",
            "iteration_reliability_context",
            "iteration_camera_weight_mean",
            "iteration_camera_geometry_summary",
            "searched_candidate_probabilities",
            "searched_candidate_logits",
            "searched_candidate_supported",
            "total_radial_delta",
            "total_tangent_delta",
        ):
            value = getattr(result, name)
            if value is not None:
                record[name] = value
        output[self.aux_record_key] = record

    def forward(self, feature_maps, metas: dict):
        if "_rosa_captured_query" in self.__dict__:
            raise RuntimeError("ROSA head forward is not reentrant")
        native_refine = self.layers[self._native_refine_layer_index]
        # A pre-hook may legally replace the query before the native box head
        # sees it.  The host extension point is allocation-free in production,
        # while this compatibility fallback is registered after all existing
        # hooks and therefore observes their final positional arguments.
        capture_via_hook = bool(
            native_refine._forward_pre_hooks
            or torch.nn.modules.module._global_forward_pre_hooks
        )
        capture = None
        handle = None
        self._rosa_captured_query = None
        self._rosa_capture_via_hook = capture_via_hook
        try:
            if capture_via_hook:
                capture = SingleCallCapture(force_return_cls=True)
                handle = native_refine.register_forward_pre_hook(
                    capture,
                    with_kwargs=True,
                )
            output = Sparse4DHead.forward(self, feature_maps, metas)
            if capture is None:
                native_query = self._rosa_captured_query
                if native_query is None:
                    raise RuntimeError("native D5 query was not captured")
            else:
                call = capture.take()
                validate_native_call(call)
                native_query = call.query
            self._append_rosa(
                output,
                native_query,
                detached_fpn(
                    feature_maps,
                    layout_receipt=metas.get("_feature_map_layout_receipt"),
                ),
                metas,
            )
            return output
        finally:
            try:
                if handle is not None:
                    handle.remove()
            finally:
                self.__dict__.pop("_rosa_captured_query", None)
                self.__dict__.pop("_rosa_capture_via_hook", None)

    def loss(self, model_outs, data, feature_maps=None):
        if not self.training:
            raise RuntimeError("loss must be called in training mode")
        record = model_outs.get(self.aux_record_key)
        if record is None:
            raise RuntimeError("missing ROSA refinement record")
        native_outs = dict(model_outs)
        for key in ("prediction", "classification", "quality"):
            native_outs[key] = list(model_outs[key][:_NATIVE_DECODERS])
        native_outs.pop(self.aux_record_key, None)
        native = Sparse4DHead.loss(self, native_outs, data, feature_maps=feature_maps)
        extra = rosa_losses(self, record, data)
        collision = set(native).intersection(extra)
        if collision:
            raise RuntimeError(f"ROSA loss name collision: {collision}")
        native.update(extra)
        return native

    def post_process(self, model_outs, output_idx=-1):
        if int(output_idx) == -1:
            output_idx = _ROSA_OUTPUT
        if int(output_idx) != _ROSA_OUTPUT:
            return Sparse4DHead.post_process(self, model_outs, output_idx=output_idx)
        record = model_outs.get(self.aux_record_key)
        if record is None:
            raise RuntimeError("missing ROSA decode record")
        lean_inference = self._lean_inference()
        results = decode_frozen_jointscore_rank(
            self.decoder,
            model_outs["classification"],
            model_outs["prediction"],
            record["frozen_native_label"],
            record["native_selected_probability"],
            record["emitted_joint_score"],
            instance_id=model_outs.get("instance_id"),
            output_idx=_ROSA_OUTPUT,
            validate_values=not lean_inference,
            pack_host_transfers=lean_inference,
        )
        if not self.rosa_cfg.export_diagnostics:
            return results
        selected = record.get("selected_indices")
        if selected is None:
            selected = torch.arange(
                record["baseline_anchor"].shape[1],
                device=record["baseline_anchor"].device,
            )[None].expand(len(results), -1)
        fields = (
            "baseline_anchor",
            "final_anchor",
            "shape1_anchor",
            "iteration_base_anchors",
            "iteration_anchors",
            "iteration_distribution_mean",
            "iteration_distribution_std",
            "iteration_candidate_offsets",
            "iteration_candidate_probabilities",
            "iteration_candidate_supported",
            "iteration_observation_supported",
            "iteration_radial_proposal",
            "iteration_radial_gate",
            "iteration_radial_gate_logit",
            "iteration_radial_safe",
            "iteration_tangent_delta",
            "iteration_yaw_raw_delta",
            "iteration_yaw_gate",
            "iteration_yaw_gate_logit",
            "iteration_yaw_delta",
            "iteration_reliability_context",
            "iteration_camera_weight_mean",
            "iteration_camera_geometry_summary",
            "searched_candidate_probabilities",
            "searched_candidate_logits",
            "searched_candidate_supported",
            "total_radial_delta",
            "total_tangent_delta",
            "posterior_entropy",
            "posterior_margin",
            "radial_gate",
        )
        for batch_index, item in enumerate(results):
            diagnostic = {
                "selected_indices": selected[batch_index].detach().cpu(),
                "projection_contract": self.rosa_cfg.projection_contract,
                "camera_geometry_channels": (
                    "optical_angle_rad",
                    "boundary",
                    "log_radial_scale",
                    "log_tangential_scale",
                    "anisotropy",
                    "footprint_short_axis",
                    "valid_fraction",
                ),
            }
            for name in fields:
                value = record.get(name)
                if value is not None:
                    diagnostic[name] = value[batch_index].detach().cpu()
            for name in (
                "baseline_anchor",
                "final_anchor",
                "shape1_anchor",
            ):
                value = record.get(name)
                if value is not None:
                    diagnostic[f"{name}_decoded"] = (
                        self.decoder.decode_box(value[batch_index]).detach().cpu()
                    )
            for name in (
                "iteration_base_anchors",
                "iteration_anchors",
            ):
                value = record.get(name)
                if value is None:
                    continue
                sample = value[batch_index]
                flat = sample.reshape(-1, sample.shape[-1])
                diagnostic[f"{name}_decoded"] = (
                    self.decoder.decode_box(flat)
                    .reshape(*sample.shape[:-1], -1)
                    .detach()
                    .cpu()
                )
            # Keep the released analysis field while exposing the clean class
            # and module names everywhere else.
            item["closed_loop_diagnostics"] = diagnostic
        return results


@MODELS.register_module()
class ROSANativeBaselineHead(Sparse4DHead):
    """Run the corrected native detector contract without a ROSA sidecar."""

    def __init__(self, rosa_cfg: None = None, **kwargs) -> None:
        if rosa_cfg is not None:
            raise ValueError("the native baseline requires rosa_cfg=None")
        super().__init__(**kwargs)


def _can_use_inplace_official_residual(
    head,
    layer,
    query: Tensor,
    attention: Tensor,
) -> bool:
    """Match the exact no-grad surface where the decoupled query is owned."""

    fc_after = head.fc_after
    return bool(
        type(head) is ROSAOfficialResidualNativeBaselineHead
        and head.decouple_attn is True
        and not head.training
        and not torch.is_grad_enabled()
        and type(layer) is torch.nn.MultiheadAttention
        and "forward" not in layer.__dict__
        and type(layer).forward is _RELEASED_MHA_FORWARD
        and not layer.training
        and not layer._forward_hooks
        and not layer._forward_pre_hooks
        and not layer._backward_hooks
        and not layer._backward_pre_hooks
        and type(fc_after) is torch.nn.Linear
        and "forward" not in fc_after.__dict__
        and type(fc_after).forward is _RELEASED_LINEAR_FORWARD
        and not fc_after.training
        and not fc_after._forward_hooks
        and not fc_after._forward_pre_hooks
        and not fc_after._backward_hooks
        and not fc_after._backward_pre_hooks
        and not torch.nn.modules.module._global_forward_hooks
        and not torch.nn.modules.module._global_forward_pre_hooks
        and not torch.nn.modules.module._global_backward_hooks
        and not torch.nn.modules.module._global_backward_pre_hooks
        and type(query) is torch.Tensor
        and type(attention) is torch.Tensor
        and query.layout == torch.strided
        and attention.layout == torch.strided
        and not query.is_quantized
        and not attention.is_quantized
        and query.device.type != "meta"
        and query.device == attention.device
        and query.dtype == attention.dtype
        and query.shape == attention.shape
        and query._base is None
        and all(name is None for name in query.names)
        and all(name is None for name in attention.names)
    )


@MODELS.register_module()
class ROSAOfficialResidualNativeBaselineHead(ROSANativeBaselineHead):
    """Restore Sparse4D's released residual around query self-attention.

    The detector host ports MMCV's ``MultiheadAttention`` wrapper to
    ``torch.nn.MultiheadAttention``.  The inner PyTorch module returns only
    the attention branch, whereas the released MMCV wrapper returns
    ``identity + dropout(attention)``.  This override restores that stateless
    wrapper operation without changing the host dependency or adding a module,
    parameter, buffer, or user-facing numeric setting.
    """

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)

        layer = self.layers[index]
        if not isinstance(layer, torch.nn.MultiheadAttention):
            raise TypeError(
                "official Sparse4D residual canary requires the audited "
                "torch.nn.MultiheadAttention host port"
            )
        if not layer.batch_first:
            raise RuntimeError(
                "official Sparse4D residual canary requires batch_first=True"
            )
        if query_pos is not None or key_pos is not None:
            raise RuntimeError(
                "the non-decoupled positional-attention path is outside the "
                "Main126 C32 contract"
            )

        if key is None:
            key = query
        if value is None:
            value = key
        allowed = {"attn_mask", "key_padding_mask"}
        unexpected = sorted(set(kwargs).difference(allowed))
        if unexpected:
            raise TypeError(f"unsupported attention arguments: {unexpected}")
        attention, _ = layer(
            query,
            key,
            value,
            attn_mask=kwargs.get("attn_mask"),
            key_padding_mask=kwargs.get("key_padding_mask"),
        )
        # MMCV's deprecated ``dropout`` argument configures both the inner
        # attention-probability dropout and this residual-branch dropout.
        # Reuse the already-fixed host value; do not introduce another knob.
        attention = F.dropout(
            attention,
            p=float(layer.dropout),
            training=self.training,
        )
        if _can_use_inplace_official_residual(self, layer, query, attention):
            # ``query`` is the owned result of ``torch.cat`` on the sealed
            # decoupled-attention path.  Reusing it removes one [B,N,2D]
            # allocation per GNN stage without touching training/autograd or
            # hook-observable execution.
            query.add_(attention)
            return self.fc_after(query)
        return self.fc_after(query + attention)


# The released configs used this type string.  It resolves to the same direct
# Sparse4DHead subclass and does not reintroduce the V15/V17 inheritance tree.
MODELS.register_module(name="ClosedLoopROSAHead", module=ROSARefinementHead)
ClosedLoopROSAHead = ROSARefinementHead


__all__ = [
    "ClosedLoopROSAHead",
    "ROSAEvaluatorAlignedSparseBox3DLoss",
    "ROSAFullFOVDeformableFeatureAggregation",
    "ROSAInferenceReuseExactHomogeneous4x4FullFOVDeformableFeatureAggregation",
    "ROSAMetricGatedAffine3x3FullFOVDeformableFeatureAggregation",
    "ROSAMetricGatedInferenceReuseAffine3x3FullFOVDeformableFeatureAggregation",
    "ROSANativeBaselineHead",
    "ROSAOfficialResidualNativeBaselineHead",
    "ROSANormalizedKeyPointsGenerator",
    "ROSARefinementHead",
]
