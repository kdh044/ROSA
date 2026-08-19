"""MMDetection3D head binding for the box-free native-fisheye ray decoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from mmengine.model import BaseModule

try:
    from mmdet3d.registry import MODELS, TASK_UTILS
    from projects.mmdet3d_plugin.models.sparse4d_head import Sparse4DHead
except ImportError as exc:  # pragma: no cover - optional detector host
    raise ImportError(
        "fisheye_ray_sparse4d requires MMDetection3D and the Sparse4D host"
    ) from exc

from ..fisheye_decoder import FisheyeRayDecoder, FisheyeRayDecoderConfig
from ..ray_losses import FisheyeRayAuxiliaryLoss, RayAuxiliaryLossConfig
from ..ray_state import RayState, normalize_direction


def _load_initial_anchors(initial_anchor) -> Tensor:
    """Load only C0 centre directions, without constructing an InstanceBank."""

    raw = initial_anchor
    if isinstance(raw, str):
        anchors = np.load(raw)
    elif isinstance(raw, (list, tuple, np.ndarray)):
        anchors = np.asarray(raw)
    elif torch.is_tensor(raw):
        anchors = raw.detach().cpu().numpy()
    else:
        raise TypeError("initial_anchor must be a path, array, or tensor")
    anchors = torch.as_tensor(anchors, dtype=torch.float32)
    if anchors.ndim != 2 or anchors.shape[-1] != 11:
        raise ValueError("Sparse4D direction initialization requires [Q,11] anchors")
    return anchors


@MODELS.register_module()
class FisheyeSparseRayHead(Sparse4DHead):
    """Fisheye sparse-ray detector with Sparse4D-compatible task boundaries.

    ``Sparse4DHead`` is used strictly as an implementation mixin for the
    established matching, loss, and decode contracts.  It does not define the
    detector architecture: this class intentionally does not construct InstanceBank,
    SparseBox3DKeyPointsGenerator, DeformableFeatureAggregation, or the native
    Cartesian refinement layers, so no unused legacy branch can overlap the
    V2 decoder or enter the optimizer.
    """

    def _normalize_decoder_index(self, value: int, name: str) -> int:
        """Normalize decoder routing without requiring a patched host."""

        value = int(value)
        if value < 0:
            value += int(self.num_decoder)
        if not 0 <= value < int(self.num_decoder):
            raise ValueError(f"{name} must index one of {self.num_decoder} decoders")
        return value

    def __init__(
        self,
        initial_anchor,
        loss_cls: dict,
        loss_reg: dict,
        decoder: dict,
        sampler: dict,
        reg_weights=None,
        ray_decoder_cfg: dict | None = None,
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
        # Do not call Sparse4DHead.__init__: that would instantiate the 13-point
        # aggregation and then leave a scientifically ambiguous dead branch.
        BaseModule.__init__(self, init_cfg=init_cfg)
        if int(num_decoder) != 6:
            raise ValueError("the V2 main method fixes six propagated decoders")
        anchors = _load_initial_anchors(initial_anchor)
        supplied_cfg = dict(ray_decoder_cfg or {})
        supplied_cfg.setdefault("num_queries", anchors.shape[0])
        supplied_cfg.setdefault("num_layers", int(num_decoder))
        config = FisheyeRayDecoderConfig(**supplied_cfg)
        if config.num_queries != anchors.shape[0]:
            raise ValueError("ray decoder query count must match anchor directions")

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
            for index in range(self.num_decoder)
            for operation in (
                "ray_self_attention",
                "fisheye_evidence",
                "ffn",
                "ray_refine",
            )
        )

        self.sampler = TASK_UTILS.build(sampler)
        self.decoder = TASK_UTILS.build(decoder)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_reg = MODELS.build(loss_reg)
        self.fisheye_ray_decoder = FisheyeRayDecoder(anchors[..., :3], config)
        self.ray_auxiliary_loss = FisheyeRayAuxiliaryLoss(
            RayAuxiliaryLossConfig(**dict(ray_aux_loss or {}))
        )

        # Inherited loss hooks inspect these switches.  The V2 head exposes no
        # legacy sidecar, temporal, overlap-distillation, or P3 path.  Training
        # DN is retained below, but its noisy box is reduced to a direction so
        # D0 never receives range, shape, or yaw privileged information.
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
        """Keep constructor initialization and skip the removed legacy stack.

        MMEngine calls the most-derived head initializer after construction.
        ``Sparse4DHead.init_weights`` walks its ``layers``/InstanceBank/DAF
        graph, none of which exists in V2 by design.  All V2 modules initialize
        themselves in their constructors, including the classifier prior and
        zero-initialized ray-refinement readouts.
        """

        return None

    def _build_direction_only_dn(self, metas: dict, reference: Tensor):
        if not self.training or not hasattr(self.sampler, "get_dn_anchors"):
            return None
        dn_metas = self.sampler.get_dn_anchors(
            metas[self.gt_cls_key],
            metas[self.gt_reg_key],
            None,
        )
        if dn_metas is None:
            return None
        (
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            dn_attention_mask,
            dn_valid_mask,
            _dn_id_target,
        ) = dn_metas
        dn_anchor = dn_anchor.to(device=reference.device, dtype=reference.dtype)
        dn_directions = normalize_direction(dn_anchor[..., :3])
        num_free = self.fisheye_ray_decoder.config.num_queries
        num_dn = dn_directions.shape[1]
        attention_mask = torch.ones(
            num_free + num_dn,
            num_free + num_dn,
            dtype=torch.bool,
            device=reference.device,
        )
        attention_mask[:num_free, :num_free] = False
        attention_mask[num_free:, num_free:] = dn_attention_mask.to(
            device=reference.device, dtype=torch.bool
        )
        return {
            "direction": dn_directions,
            "attention_mask": attention_mask,
            "reg_target": dn_reg_target,
            "cls_target": dn_cls_target,
            "valid_mask": dn_valid_mask,
        }

    @staticmethod
    def _slice_state(state: RayState, end: int) -> RayState:
        return RayState(
            direction=state.direction[:, :end],
            log_range=state.log_range[:, :end],
            log_range_std=state.log_range_std[:, :end],
            log_size=state.log_size[:, :end],
            yaw=state.yaw[:, :end],
            velocity=state.velocity[:, :end],
        )

    def forward(self, feature_maps, metas: dict):
        if not isinstance(feature_maps, (list, tuple)):
            raise TypeError(
                "V2 requires unpacked FPN levels; set model.use_deformable_func=False"
            )
        dn = self._build_direction_only_dn(metas, feature_maps[0])
        output = self.fisheye_ray_decoder(
            feature_maps,
            metas,
            extra_directions=None if dn is None else dn["direction"],
            self_attention_mask=None if dn is None else dn["attention_mask"],
        )
        num_free = self.fisheye_ray_decoder.config.num_queries
        states = tuple(self._slice_state(state, num_free) for state in output.states)
        result = {
            "classification": [value[:, :num_free] for value in output.classification],
            "prediction": [value[:, :num_free] for value in output.prediction],
            "quality": [value[:, :num_free] for value in output.quality],
            "ray_direction": [state.direction for state in states],
            "ray_log_range": [state.log_range for state in states],
            "ray_log_range_std": [state.log_range_std for state in states],
            "ray_log_size": [state.log_size for state in states],
            "ray_yaw": [state.yaw for state in states],
            "ray_velocity": [state.velocity for state in states],
            # Diagnostics are detached so loss output does not retain six
            # additional attention graphs beyond the propagated query graph.
            "fisheye_attention": [
                value[:, :num_free].detach() for value in output.attention
            ],
            "fisheye_level_compatibility": [
                value[:, :num_free].detach() for value in output.level_compatibility
            ],
        }
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

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        """Preserve native DN masks while exposing an explicit background.

        ``SparseBox3DTarget`` uses ``-3`` as its negative-DN sentinel.  Modern
        MMDetection focal loss requires a non-negative background class, while
        the inherited regression preparation relies on that sentinel to keep
        negative queries out of box regression.  Convert only around the
        inherited preparation boundary so both contracts remain true.
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

    @staticmethod
    def _state_from_outputs(model_outs, decoder_index: int) -> RayState:
        return RayState(
            direction=model_outs["ray_direction"][decoder_index],
            log_range=model_outs["ray_log_range"][decoder_index],
            log_range_std=model_outs["ray_log_range_std"][decoder_index],
            log_size=model_outs["ray_log_size"][decoder_index],
            yaw=model_outs["ray_yaw"][decoder_index],
            velocity=model_outs["ray_velocity"][decoder_index],
        )

    def _loss_fp32(self, model_outs, data, feature_maps=None):
        losses = super()._loss_fp32(model_outs, data, feature_maps)
        for decoder_index, (classification, prediction) in enumerate(
            zip(model_outs["classification"], model_outs["prediction"])
        ):
            _, target, _ = self.sampler.sample(
                classification,
                prediction,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            matched = torch.logical_not(torch.all(target == 0, dim=-1))
            state = self._state_from_outputs(model_outs, decoder_index)
            auxiliary = self.ray_auxiliary_loss(state, target, matched)
            for name, value in auxiliary.items():
                losses[f"{name}_{decoder_index}"] = value
        return losses


__all__ = ["FisheyeSparseRayHead"]
