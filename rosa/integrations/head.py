"""Sparse4D head that runs the ROSA hybrid box decoder."""

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

from ..projection import make_native_projector
from ..decoder import (
    HybridBoxDecoderConfig,
    HybridSparseBoxDecoder,
    HybridBoxState,
)
from ..losses import (
    FisheyeHybridRayAuxiliaryLoss,
    RayAuxiliaryLossConfig,
)


@MODELS.register_module()
class ROSAHead(Sparse4DHead):
    """Sparse4D head with fisheye box observation in all six decoder stages.

    Keeps the single-frame Sparse4D InstanceBank, 11-value anchors, target
    sampler, losses and output decoder, and replaces the per-stage image
    observation and refinement with the ROSA decoder.
    """

    def __init__(
        self,
        instance_bank: dict,
        loss_cls: dict,
        loss_reg: dict,
        decoder: dict,
        sampler: dict,
        reg_weights=None,
        hybrid_decoder_cfg: dict | None = None,
        ray_aux_loss: dict | None = None,
        num_decoder: int = 6,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        cls_threshold_to_reg: float = -1.0,
        init_cfg: dict | None = None,
    ) -> None:
        # Skip the parent constructor, which does not build an InstanceBank.
        BaseModule.__init__(self, init_cfg=init_cfg)
        supplied_cfg = dict(hybrid_decoder_cfg or {})
        supplied_cfg.setdefault("num_layers", int(num_decoder))
        config = HybridBoxDecoderConfig(**supplied_cfg)

        self.num_decoder = int(num_decoder)
        self.num_classes = int(config.num_classes)
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = float(cls_threshold_to_reg)
        self.reg_weights = [1.0] * 10 if reg_weights is None else list(reg_weights)

        self.instance_bank = MODELS.build(instance_bank)
        self.sampler = TASK_UTILS.build(sampler)
        self.decoder = TASK_UTILS.build(decoder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_reg = MODELS.build(loss_reg)
        self.fisheye_box_decoder = HybridSparseBoxDecoder(
            config, projector=make_native_projector(config.projection_type)
        )
        self.ray_auxiliary_loss = (
            FisheyeHybridRayAuxiliaryLoss(RayAuxiliaryLossConfig(**dict(ray_aux_loss)))
            if ray_aux_loss is not None
            else None
        )

    def init_weights(self) -> None:
        """Initialize the InstanceBank; decoder modules initialize themselves."""

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
        if dn_metas is None:  # no ground truth in the batch
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
        instance_feature, anchor = self.instance_bank.get(batch_size)
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
        states = tuple(
            self._slice_hybrid_state(state, num_free) for state in output.states
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
        result["global_yaw"] = [state.yaw for state in states]
        result["global_velocity"] = [state.velocity for state in states]
        if output.raw_log_range_std is not None:
            result["ray_raw_log_range_std"] = [
                value[:, :num_free] for value in output.raw_log_range_std
            ]
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


    def _loss_fp32(self, model_outs, data, feature_maps=None):
        if self.ray_auxiliary_loss is None:
            return super()._loss_fp32(model_outs, data, feature_maps)
        # Record the Hungarian assignment of every decoder stage made by the
        # Sparse4D loss and reuse it for the ray auxiliary losses.
        assignments = []
        original_sample = self.sampler.sample

        def record_sample(*args, **kwargs):
            assignment = original_sample(*args, **kwargs)
            assignments.append(assignment)
            return assignment

        self.sampler.sample = record_sample
        try:
            losses = super()._loss_fp32(model_outs, data, feature_maps)
        finally:
            del self.sampler.sample
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

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        """Map the background class back to ``-3`` for the parent DN loss.

        ``SparseBox3DTarget`` marks negative DN queries with ``-3``, which the
        parent uses to mask regression, while focal loss needs a non-negative
        background class.
        """

        target_key = f"{prefix}dn_cls_target"
        model_outs_with_sentinel = dict(model_outs)
        target = model_outs[target_key]
        model_outs_with_sentinel[target_key] = torch.where(
            target.eq(self.num_classes), target.new_full((), -3), target
        )
        prepared = list(
            super().prepare_for_dn_loss(model_outs_with_sentinel, prefix=prefix)
        )
        prepared[1] = torch.where(
            prepared[1].eq(-3),
            prepared[1].new_full((), self.num_classes),
            prepared[1],
        )
        return tuple(prepared)


__all__ = ["ROSAHead"]
