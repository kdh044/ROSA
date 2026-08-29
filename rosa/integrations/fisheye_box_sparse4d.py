"""Sparse4D task binding for the V3 and V4 fisheye box decoders."""

from __future__ import annotations

import torch
from torch import Tensor
from mmengine.model import BaseModule

try:
    from mmdet3d.registry import MODELS, TASK_UTILS
    from projects.mmdet3d_plugin.models.sparse4d_head import Sparse4DHead
except ImportError as exc:  # pragma: no cover - optional detector host
    raise ImportError(
        "fisheye_box_sparse4d requires MMDetection3D and the Sparse4D host"
    ) from exc

from ..fisheye_box_decoder import (
    FisheyeBoxDecoderConfig,
    FisheyeSparseBoxDecoder,
)
from ..hybrid_box_decoder import (
    FisheyeBoxDecoderV4Config,
    FisheyeSparseBoxDecoderV4,
    HybridBoxState,
)
from ..ray_losses import (
    FisheyeHybridRayAuxiliaryLoss,
    FisheyeRayAuxiliaryLoss,
    RayAuxiliaryLossConfig,
)
from ..ray_state import RayState
from .fisheye_ray_sparse4d import FisheyeSparseRayHead


@MODELS.register_module()
class FisheyeSparseBoxHead(FisheyeSparseRayHead):
    """C0 InstanceBank plus fisheye initialization and D0--D5 box evidence.

    The head keeps the real single-frame Sparse4D InstanceBank, full 11-value
    learned anchors, target sampler, losses, and output decoder.  It replaces
    only the per-stage image observation/refinement path.  Cross-frame cache is
    rejected by the first V3 experiment so the comparison remains matched to
    temporal-free corrected C0.
    """

    def __init__(
        self,
        instance_bank: dict,
        loss_cls: dict,
        loss_reg: dict,
        decoder: dict,
        sampler: dict,
        reg_weights=None,
        box_decoder_cfg: dict | None = None,
        v4_decoder_cfg: dict | None = None,
        ray_aux_loss: dict | None = None,
        num_decoder: int = 6,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        cls_threshold_to_reg: float = -1.0,
        dn_loss_weight: float = 5.0,
        inference_output_decoder_index: int = -1,
        inference_classification_decoder_index: int | None = None,
        inference_quality_decoder_index: int | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        # Bypass the V2 constructor, which deliberately deletes InstanceBank.
        BaseModule.__init__(self, init_cfg=init_cfg)
        if int(num_decoder) != 6:
            raise ValueError("the fisheye box head fixes six D0-D5 stages")
        if (box_decoder_cfg is None) == (v4_decoder_cfg is None):
            raise ValueError(
                "provide exactly one of box_decoder_cfg (V3) or v4_decoder_cfg (V4)"
            )
        self.use_hybrid_v4 = v4_decoder_cfg is not None
        supplied_cfg = dict(v4_decoder_cfg if self.use_hybrid_v4 else box_decoder_cfg)
        supplied_cfg.setdefault("num_layers", int(num_decoder))
        config = (
            FisheyeBoxDecoderV4Config(**supplied_cfg)
            if self.use_hybrid_v4
            else FisheyeBoxDecoderConfig(**supplied_cfg)
        )

        self.num_decoder = int(num_decoder)
        self.num_classes = int(config.num_classes)
        self.num_single_frame_decoder = self.num_decoder
        self.final_output_decoder_index = self.num_decoder - 1
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = float(cls_threshold_to_reg)
        self.dn_loss_weight = float(dn_loss_weight)
        self.reg_weights = [1.0] * 10 if reg_weights is None else list(reg_weights)
        self.inference_output_decoder_index = self._normalize_decoder_index(
            inference_output_decoder_index, "inference_output_decoder_index"
        )
        self.inference_classification_decoder_index = self._normalize_decoder_index(
            (
                self.inference_output_decoder_index
                if inference_classification_decoder_index is None
                else inference_classification_decoder_index
            ),
            "inference_classification_decoder_index",
        )
        self.inference_quality_decoder_index = self._normalize_decoder_index(
            (
                self.inference_output_decoder_index
                if inference_quality_decoder_index is None
                else inference_quality_decoder_index
            ),
            "inference_quality_decoder_index",
        )
        self.inference_box_source_by_dimension = None
        self.operation_order = tuple(
            operation
            for _ in range(self.num_decoder)
            for operation in (
                "query_self_attention",
                "13_point_fisheye_evidence",
                "ffn",
                "hybrid_box_refine" if self.use_hybrid_v4 else "spherical_box_refine",
            )
        )

        self.instance_bank = MODELS.build(instance_bank)
        if self.instance_bank.num_anchor != config.num_queries:
            raise ValueError(
                "InstanceBank anchor count must match decoder query budget"
            )
        if self.instance_bank.num_temp_instances != 0:
            raise ValueError("the matched fisheye box arm requires temporal slots=0")
        self.sampler = TASK_UTILS.build(sampler)
        self.decoder = TASK_UTILS.build(decoder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_reg = MODELS.build(loss_reg)
        self.fisheye_box_decoder = (
            FisheyeSparseBoxDecoderV4(config)
            if self.use_hybrid_v4
            else FisheyeSparseBoxDecoder(config)
        )
        if self.use_hybrid_v4:
            if (ray_aux_loss is None) != (not config.predict_range_uncertainty):
                raise ValueError(
                    "V4 ray_aux_loss and predict_range_uncertainty must be "
                    "enabled or disabled together"
                )
            if ray_aux_loss is not None:
                self.ray_auxiliary_loss = FisheyeHybridRayAuxiliaryLoss(
                    RayAuxiliaryLossConfig(**dict(ray_aux_loss))
                )
        else:
            self.ray_auxiliary_loss = FisheyeRayAuxiliaryLoss(
                RayAuxiliaryLossConfig(**dict(ray_aux_loss or {}))
            )

        # Inherited loss/post-process hooks inspect these attributes.  V3 has
        # no ROSA sidecar, legacy DAF, temporal cache, or duplicate observation.
        self.use_overlap_support_distillation = False
        self.use_fisheye_information_update = False
        self.use_ray_range_aux = False
        self.use_ray_yaw_aux = False
        self.use_decoder_feature_aux = False
        self._last_decoder_aggregation_weights = [None] * self.num_decoder
        self._last_decoder_aggregation_weights_by_keypoint = [None] * self.num_decoder
        self._last_decoder_aggregation_keypoint_mass = [None] * self.num_decoder
        self._last_decoder_ray_object_attention_pre_dropout = [None] * self.num_decoder
        self._last_decoder_ray_object_valid_views = [None] * self.num_decoder

    def init_weights(self) -> None:
        """Initialize the retained InstanceBank; decoder modules self-initialize."""

        self.instance_bank.init_weight()

    def _append_dn_queries(
        self,
        instance_feature: Tensor,
        anchor: Tensor,
        metas: dict,
    ) -> tuple[Tensor, Tensor, Tensor | None, dict | None]:
        if not self.training or not hasattr(self.sampler, "get_dn_anchors"):
            return instance_feature, anchor, None, None
        dn_metas = self.sampler.get_dn_anchors(
            metas[self.gt_cls_key],
            metas[self.gt_reg_key],
            None,
        )
        if dn_metas is None:
            return instance_feature, anchor, None, None
        (
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            dn_attention_mask,
            dn_valid_mask,
            _dn_id_target,
        ) = dn_metas
        dn_anchor = dn_anchor.to(device=anchor.device, dtype=anchor.dtype)
        if dn_anchor.shape[-1] > anchor.shape[-1]:
            raise ValueError("DN anchor has more state dimensions than InstanceBank")
        if dn_anchor.shape[-1] < anchor.shape[-1]:
            dn_anchor = torch.cat(
                (
                    dn_anchor,
                    dn_anchor.new_zeros(
                        *dn_anchor.shape[:-1],
                        anchor.shape[-1] - dn_anchor.shape[-1],
                    ),
                ),
                dim=-1,
            )
        num_free = instance_feature.shape[1]
        num_dn = dn_anchor.shape[1]
        dn_feature = instance_feature.new_zeros(
            instance_feature.shape[0], num_dn, instance_feature.shape[-1]
        )
        instance_feature = torch.cat((instance_feature, dn_feature), dim=1)
        anchor = torch.cat((anchor, dn_anchor), dim=1)
        attention_mask = torch.ones(
            num_free + num_dn,
            num_free + num_dn,
            dtype=torch.bool,
            device=anchor.device,
        )
        attention_mask[:num_free, :num_free] = False
        attention_mask[num_free:, num_free:] = dn_attention_mask.to(
            device=anchor.device, dtype=torch.bool
        )
        return (
            instance_feature,
            anchor,
            attention_mask,
            {
                "reg_target": dn_reg_target,
                "cls_target": dn_cls_target,
                "valid_mask": dn_valid_mask,
            },
        )

    def forward(self, feature_maps, metas: dict):
        if not isinstance(feature_maps, (list, tuple)):
            raise TypeError(
                "fisheye box decoding requires unpacked FPN levels; "
                "set model.use_deformable_func=False"
            )
        batch_size = feature_maps[0].shape[0]
        if getattr(self.sampler, "dn_metas", None) is not None:
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_feature,
            temp_anchor,
            _time_interval,
        ) = self.instance_bank.get(batch_size, metas)
        if temp_feature is not None or temp_anchor is not None:
            raise RuntimeError("temporal-free fisheye box head received a cache")
        instance_feature, anchor, attention_mask, dn = self._append_dn_queries(
            instance_feature,
            anchor,
            metas,
        )
        output = self.fisheye_box_decoder(
            feature_maps,
            metas,
            instance_feature,
            anchor,
            self_attention_mask=attention_mask,
        )
        num_free = self.instance_bank.num_anchor
        if self.use_hybrid_v4:
            states = tuple(
                self._slice_hybrid_state(state, num_free) for state in output.states
            )
        else:
            states = tuple(
                self._slice_state(state, num_free) for state in output.states
            )
        result = {
            "classification": [value[:, :num_free] for value in output.classification],
            "prediction": [value[:, :num_free] for value in output.prediction],
            "quality": [value[:, :num_free] for value in output.quality],
            "ray_direction": [state.direction for state in states],
            "ray_log_range": [state.log_range for state in states],
            "ray_log_size": [state.log_size for state in states],
            "fisheye_camera_level_attention": [
                value[:, :num_free].detach() for value in output.camera_level_attention
            ],
            "fisheye_point_attention": [
                value[:, :num_free].detach() for value in output.point_attention
            ],
            "fisheye_level_compatibility": [
                value[:, :num_free].detach() for value in output.level_compatibility
            ],
        }
        if self.use_hybrid_v4:
            result.update(
                {
                    "global_yaw": [state.yaw for state in states],
                    "global_velocity": [state.velocity for state in states],
                }
            )
            if output.raw_log_range_std is not None:
                result["ray_raw_log_range_std"] = [
                    value[:, :num_free] for value in output.raw_log_range_std
                ]
        else:
            initializer_state = self._slice_state(output.initializer_state, num_free)
            result.update(
                {
                    "ray_log_range_std": [state.log_range_std for state in states],
                    "ray_yaw": [state.yaw for state in states],
                    "ray_velocity": [state.velocity for state in states],
                    "initializer_direction": initializer_state.direction,
                    "initializer_log_range": initializer_state.log_range,
                    "initializer_log_range_std": initializer_state.log_range_std,
                    "initializer_log_size": initializer_state.log_size,
                    "initializer_yaw": initializer_state.yaw,
                    "initializer_velocity": initializer_state.velocity,
                    "initializer_prediction": initializer_state.to_sparse_anchor()[
                        ..., :10
                    ],
                    "initializer_gate": output.initializer_gate[:, :num_free].detach(),
                }
            )
        if dn is not None:
            dn_cls_target = torch.where(
                dn["cls_target"].eq(-3),
                dn["cls_target"].new_full((), self.num_classes),
                dn["cls_target"],
            )
            result.update(
                {
                    "dn_classification": [
                        value[:, num_free:] for value in output.classification
                    ],
                    "dn_prediction": [
                        value[:, num_free:] for value in output.prediction
                    ],
                    "dn_reg_target": dn["reg_target"],
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": dn["valid_mask"],
                }
            )
        return result

    @staticmethod
    def _slice_hybrid_state(state: HybridBoxState, end: int) -> HybridBoxState:
        return HybridBoxState(
            direction=state.direction[:, :end],
            log_range=state.log_range[:, :end],
            log_size=state.log_size[:, :end],
            yaw=state.yaw[:, :end],
            velocity=state.velocity[:, :end],
        )

    @staticmethod
    def _initializer_state_from_outputs(model_outs) -> RayState:
        return RayState(
            direction=model_outs["initializer_direction"],
            log_range=model_outs["initializer_log_range"],
            log_range_std=model_outs["initializer_log_range_std"],
            log_size=model_outs["initializer_log_size"],
            yaw=model_outs["initializer_yaw"],
            velocity=model_outs["initializer_velocity"],
        )

    def _loss_fp32(self, model_outs, data, feature_maps=None):
        if self.use_hybrid_v4:
            native_fp32 = getattr(Sparse4DHead, "_loss_fp32", None)
            if native_fp32 is None:
                raise RuntimeError(
                    "the fisheye box head requires Sparse4DHead._loss_fp32"
                )

            def native_loss():
                return native_fp32(self, model_outs, data, feature_maps)

            if not hasattr(self, "ray_auxiliary_loss"):
                return native_loss()
            # Sparse4D's native loss already performs the six Hungarian
            # assignments needed by the hybrid ray supervision.  Capture
            # those exact results while the native implementation runs so the
            # auxiliary losses do not solve the same assignment problem a
            # second time.  The sampler method is restored even if the native
            # loss raises, keeping the optimization local to this call.
            assignments = []
            sample_was_instance_attribute = "sample" in vars(self.sampler)
            original_instance_sample = vars(self.sampler).get("sample")
            original_sample = self.sampler.sample

            def record_sample(*args, **kwargs):
                assignment = original_sample(*args, **kwargs)
                assignments.append(assignment)
                return assignment

            self.sampler.sample = record_sample
            try:
                losses = native_loss()
            finally:
                if sample_was_instance_attribute:
                    self.sampler.sample = original_instance_sample
                else:
                    del self.sampler.sample
            num_stages = len(model_outs["classification"])
            if len(assignments) != num_stages:
                raise RuntimeError(
                    "native Sparse4D loss produced an unexpected number of "
                    f"assignments: expected {num_stages}, got {len(assignments)}"
                )
            for decoder_index, (_, target, _) in enumerate(assignments):
                matched = torch.logical_not(torch.all(target == 0, dim=-1))
                auxiliary = self.ray_auxiliary_loss(
                    model_outs["ray_direction"][decoder_index],
                    model_outs["ray_log_range"][decoder_index],
                    model_outs["ray_raw_log_range_std"][decoder_index],
                    target,
                    matched,
                )
                for name, value in auxiliary.items():
                    losses[f"{name}_{decoder_index}"] = value
            return losses
        losses = super()._loss_fp32(model_outs, data, feature_maps)
        classification = model_outs["classification"][0]
        prediction = model_outs["initializer_prediction"]
        _, target, _ = self.sampler.sample(
            classification,
            prediction,
            data[self.gt_cls_key],
            data[self.gt_reg_key],
        )
        matched = torch.logical_not(torch.all(target == 0, dim=-1))
        auxiliary = self.ray_auxiliary_loss(
            self._initializer_state_from_outputs(model_outs),
            target,
            matched,
        )
        for name, value in auxiliary.items():
            losses[f"initializer_{name}"] = value
        return losses


__all__ = ["FisheyeSparseBoxHead"]
