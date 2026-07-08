# ------------------------------------------------------------------------
# Ported from Sparse4D (HorizonRobotics) to mmcv 2.x / mmdet3d 1.4
# Changes:
#   HEADS / LOSSES → mmdet3d MODELS
#   BBOX_SAMPLERS / BBOX_CODERS → mmdet3d TASK_UTILS
#   ATTENTION / PLUGIN_LAYERS / POSITIONAL_ENCODING / FEEDFORWARD_NETWORK /
#       NORM_LAYERS → mmdet3d MODELS (all unified)
#   BaseModule from mmengine.model
#   reduce_mean from mmengine.dist.utils
#   force_fp32 dropped
# ------------------------------------------------------------------------
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmengine.dist import all_reduce
import torch.distributed as dist

from mmdet3d.registry import MODELS, TASK_UTILS

from .blocks import DeformableFeatureAggregation as DFG  # noqa: F401
from projects.mmdet3d_plugin.core.box3d import X, Y, Z, W, L, H

__all__ = ["Sparse4DHead"]


def reduce_mean(tensor):
    """All-reduce mean of a tensor across DDP workers (no-op when single GPU)."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    all_reduce(tensor, op="mean")
    return tensor


def _build_norm(cfg):
    """Resolve mmcv-1 style norm configs (LN / BN / GN) to torch modules."""
    if cfg is None:
        return None
    cfg = cfg.copy()
    norm_type = cfg.pop("type")
    if norm_type in ("LN", "LayerNorm"):
        normalized_shape = cfg.pop("normalized_shape")
        return nn.LayerNorm(normalized_shape, **cfg)
    if norm_type in ("BN", "BN2d", "BatchNorm2d"):
        return nn.BatchNorm2d(**cfg)
    if norm_type in ("BN1d", "BatchNorm1d"):
        return nn.BatchNorm1d(**cfg)
    if norm_type in ("GN", "GroupNorm"):
        return nn.GroupNorm(**cfg)
    return MODELS.build(dict(type=norm_type, **cfg))


def _build_attention(cfg):
    """MultiheadAttention from mmcv-1 ATTENTION → torch.nn.MultiheadAttention.

    Falls back to MODELS registry otherwise.
    """
    if cfg is None:
        return None
    cfg = cfg.copy()
    if cfg.get("type") == "MultiheadAttention":
        cfg.pop("type")
        embed_dims = cfg.pop("embed_dims")
        num_heads = cfg.pop("num_heads")
        dropout = cfg.pop("dropout", 0.0)
        batch_first = cfg.pop("batch_first", True)
        return nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
            **cfg,
        )
    return MODELS.build(cfg)


@MODELS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        gate_invalid_temp_gnn: bool = False,
        ray_range_aux: Optional[dict] = None,
        init_cfg: dict = None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn
        self.gate_invalid_temp_gnn = gate_invalid_temp_gnn
        default_ray_range_aux = dict(
            enabled=False,
            loss_weight=0.01,
            min_range=0.2,
            max_range=40.0,
            beta=0.1,
            target_z_is_bottom=True,
            min_visible_views=2,
            min_projection_quality=0.02,
            boundary_margin=0.08,
            boundary_weight=1.0,
            center_weight=0.0,
            incidence_weight=0.0,
            max_incidence_deg=95.0,
            detach_geometry=True,
            projection_type=None,
            use_fusion_support=False,
            multiply_target_quality=False,
        )
        if ray_range_aux is not None:
            default_ray_range_aux.update(ray_range_aux)
        self.ray_range_aux_cfg = default_ray_range_aux
        self.use_ray_range_aux = bool(
            self.ray_range_aux_cfg["enabled"]
            and float(self.ray_range_aux_cfg["loss_weight"]) > 0.0
        )

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        self.instance_bank = MODELS.build(instance_bank)
        self.anchor_encoder = MODELS.build(anchor_encoder)
        self.sampler = TASK_UTILS.build(sampler) if sampler is not None else None
        self.decoder = TASK_UTILS.build(decoder) if decoder is not None else None
        self.loss_cls = MODELS.build(loss_cls) if loss_cls is not None else None
        self.loss_reg = MODELS.build(loss_reg) if loss_reg is not None else None

        self.layers = nn.ModuleList()
        for op in self.operation_order:
            if op == "temp_gnn":
                self.layers.append(_build_attention(temp_graph_model))
            elif op == "gnn":
                self.layers.append(_build_attention(graph_model))
            elif op == "norm":
                self.layers.append(_build_norm(norm_layer))
            elif op == "ffn":
                self.layers.append(MODELS.build(ffn))
            elif op == "deformable":
                self.layers.append(MODELS.build(deformable_model))
            elif op == "refine":
                self.layers.append(MODELS.build(refine_layer))
            else:
                raise NotImplementedError(f"unknown operation '{op}'")

        self.embed_dims = self.instance_bank.embed_dims
        if self.use_ray_range_aux:
            hidden_dims = max(64, self.embed_dims // 2)
            self.ray_range_relation_encoder = nn.Sequential(
                nn.Linear(8, hidden_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dims),
                nn.Linear(hidden_dims, self.embed_dims),
            )
            self.ray_range_head = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims),
                nn.Linear(self.embed_dims, 1),
            )
        else:
            self.ray_range_relation_encoder = None
            self.ray_range_head = None
        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()
        if self.ray_range_relation_encoder is not None:
            for m in self.ray_range_relation_encoder:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
            for m in self.ray_range_head[:-1]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(self.ray_range_head[-1].weight, 0.0)
            nn.init.constant_(self.ray_range_head[-1].bias, 0.0)

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
        # torch MultiheadAttention returns (out, attn_weights)
        layer = self.layers[index]
        if isinstance(layer, nn.MultiheadAttention):
            if key is None:
                key = query
            if value is None:
                value = query
            attn_mask = kwargs.get("attn_mask")
            out, _ = layer(query, key, value, attn_mask=attn_mask)
        else:
            out = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        return self.fc_after(out)

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        if (
            self.sampler is not None
            and self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size,
            metas,
            dn_metas=self.sampler.dn_metas if self.sampler is not None else None,
        )

        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if (
            self.training
            and self.sampler is not None
            and hasattr(self.sampler, "get_dn_anchors")
        ):
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        prediction = []
        classification = []
        quality = []
        decoder_instance_features = []
        decoder_view_supports = []
        last_view_support = None
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                current_instance_feature = instance_feature
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
                temp_mask = getattr(self.instance_bank, "mask", None)
                if (
                    self.gate_invalid_temp_gnn
                    and temp_instance_feature is not None
                    and temp_mask is not None
                ):
                    instance_feature = torch.where(
                        temp_mask[:, None, None],
                        instance_feature,
                        current_instance_feature,
                    )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                deformable_layer = self.layers[i]
                instance_feature = deformable_layer(
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
                if self.use_ray_range_aux:
                    last_view_support = getattr(
                        deformable_layer, "_last_view_support", None
                    )
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if self.use_ray_range_aux:
                    decoder_instance_features.append(instance_feature)
                    decoder_view_supports.append(last_view_support)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            if self.use_ray_range_aux:
                decoder_instance_features = [
                    x[:, :num_free_instance] for x in decoder_instance_features
                ]
                decoder_view_supports = [
                    x[:, :num_free_instance] if x is not None else None
                    for x in decoder_view_supports
                ]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )
        if self.use_ray_range_aux:
            output["decoder_instance_features"] = decoder_instance_features
            output["decoder_view_supports"] = decoder_view_supports

        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if not self.training:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    def loss(self, model_outs, data, feature_maps=None):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        decoder_instance_features = model_outs.get("decoder_instance_features")
        decoder_view_supports = model_outs.get("decoder_view_supports")
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if (
                self.use_ray_range_aux
                and decoder_instance_features is not None
                and decoder_idx < len(decoder_instance_features)
            ):
                view_support = None
                if (
                    decoder_view_supports is not None
                    and decoder_idx < len(decoder_view_supports)
                ):
                    view_support = decoder_view_supports[decoder_idx]
                output.update(
                    self.loss_ray_range_aux(
                        decoder_instance_features[decoder_idx],
                        reg,
                        reg_target,
                        mask_valid,
                        data,
                        decoder_idx,
                        view_support=view_support,
                    )
                )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output

    def loss_ray_range_aux(
        self,
        instance_feature,
        reg_pred,
        reg_target,
        mask_valid,
        data,
        decoder_idx,
        view_support=None,
    ):
        """Object/query-level camera ray-range auxiliary loss.

        GT boxes are used only as matched supervision targets. The head input
        uses the predicted query geometry plus query feature, so inference does
        not require GT centers.
        """
        if "lidar2cam" not in data or "image_wh" not in data:
            return {}
        cfg = self.ray_range_aux_cfg
        pred_center = self._ray_aux_geometric_center(reg_pred, cfg)
        pred_center = torch.nan_to_num(
            pred_center, nan=0.0, posinf=0.0, neginf=0.0
        )
        target_center = self._ray_aux_geometric_center(reg_target, cfg)
        target_center = torch.nan_to_num(
            target_center, nan=0.0, posinf=0.0, neginf=0.0
        )

        target_points = target_center[:, :, None, :]
        target_points_2d = self._project_ray_aux_points(
            target_points, data, cfg
        )
        target_quality, target_view_mask = self._ray_aux_projection_quality(
            target_points_2d, target_center, data, cfg
        )

        visible = target_view_mask & (
            target_quality > float(cfg["min_projection_quality"])
        )
        visible_count = visible.sum(dim=1)
        query_mask = mask_valid & (
            visible_count >= int(cfg["min_visible_views"])
        )
        view_mask = visible & query_mask[:, None]
        if not bool(view_mask.any()):
            return {f"loss_ray_range_{decoder_idx}": reg_pred.new_zeros(())}

        pred_for_relation = pred_center
        if bool(cfg["detach_geometry"]):
            pred_for_relation = pred_for_relation.detach()
        relation = self._build_ray_aux_relation(pred_for_relation, data, cfg)
        relation_embed = self.ray_range_relation_encoder(relation)
        ray_feature = instance_feature[:, :, None, :] + relation_embed
        pred = self.ray_range_head(ray_feature).squeeze(-1)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)

        target_cam = self._points_to_cam(target_center, data["lidar2cam"])
        target_range = torch.linalg.norm(target_cam, dim=-1).transpose(1, 2)
        target = self._normalize_ray_range(target_range, cfg)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

        view_mask_bn = view_mask.transpose(1, 2)
        if bool(cfg["use_fusion_support"]) and view_support is not None:
            weights = self._reduce_view_support(view_support).to(dtype=pred.dtype)
            weights = weights * view_mask_bn.to(dtype=pred.dtype)
            if bool(cfg["multiply_target_quality"]):
                weights = weights * target_quality.transpose(1, 2).to(
                    dtype=pred.dtype
                )
        else:
            weights = target_quality.transpose(1, 2).to(dtype=pred.dtype)
            weights = weights * view_mask_bn.to(dtype=pred.dtype)
        weights = torch.nan_to_num(
            weights, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(min=0.0)

        raw_loss = F.smooth_l1_loss(
            pred,
            target.to(dtype=pred.dtype),
            beta=float(cfg["beta"]),
            reduction="none",
        )
        raw_loss = torch.nan_to_num(raw_loss, nan=0.0, posinf=0.0, neginf=0.0)
        denom = weights.sum().clamp(min=1.0)
        loss = (raw_loss * weights).sum() / denom
        loss = loss * float(cfg["loss_weight"])
        return {f"loss_ray_range_{decoder_idx}": loss}

    @staticmethod
    def _reduce_view_support(view_support):
        """Reduce DFA support from keypoint/group support to object-camera support."""
        support = view_support
        while support.dim() > 3:
            support = support.mean(dim=-1)
        return torch.nan_to_num(
            support, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)

    @staticmethod
    def _ray_aux_geometric_center(box, cfg):
        center = box[..., [X, Y, Z]].clone()
        if bool(cfg["target_z_is_bottom"]) and box.shape[-1] > H:
            height = box[..., H].clamp(min=-10.0, max=10.0).exp()
            center[..., 2] = center[..., 2] + height * 0.5
        return center

    def _project_ray_aux_points(self, points, data, cfg):
        projection_type = cfg.get("projection_type")
        if projection_type is None:
            if "kb_intrinsic" in data and "kb_distortion" in data:
                projection_type = "kb_fisheye"
            elif "ocam_intrinsic" in data and "ocam_pol" in data:
                projection_type = "ocam_fisheye"
            else:
                projection_type = "pinhole"
        if projection_type == "kb_fisheye":
            from .projection import kb_project_points

            return kb_project_points(points, data, data["image_wh"])
        if projection_type == "ocam_fisheye":
            from .projection import ocam_project_points

            return ocam_project_points(points, data, data["image_wh"])
        return DFG._project_points_pinhole(
            points, data["projection_mat"], data["image_wh"]
        )

    def _ray_aux_projection_quality(self, points_2d, center_lidar, data, cfg):
        points_2d = points_2d[..., 0, :]
        finite = torch.isfinite(points_2d).all(dim=-1)
        safe_points = torch.nan_to_num(
            points_2d, nan=-1.0, posinf=2.0, neginf=-1.0
        )
        valid = (
            (safe_points[..., 0] >= 0.0)
            & (safe_points[..., 0] <= 1.0)
            & (safe_points[..., 1] >= 0.0)
            & (safe_points[..., 1] <= 1.0)
            & finite
        )
        boundary = torch.minimum(
            torch.minimum(safe_points[..., 0], 1.0 - safe_points[..., 0]),
            torch.minimum(safe_points[..., 1], 1.0 - safe_points[..., 1]),
        )
        boundary_conf = (
            boundary / max(float(cfg["boundary_margin"]), 1e-6)
        ).clamp(0.0, 1.0)
        center_dist = torch.linalg.norm(safe_points - 0.5, dim=-1) / (
            0.5 ** 0.5
        )
        center_conf = (1.0 - center_dist).clamp(0.0, 1.0)
        angle_conf = self._ray_aux_incidence_confidence(center_lidar, data, cfg)
        quality = valid.to(dtype=points_2d.dtype)
        quality = quality * boundary_conf.pow(float(cfg["boundary_weight"]))
        quality = quality * center_conf.pow(float(cfg["center_weight"]))
        quality = quality * angle_conf.pow(float(cfg["incidence_weight"]))
        quality = torch.nan_to_num(
            quality, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        return quality, valid

    def _ray_aux_incidence_confidence(self, center_lidar, data, cfg):
        pts_cam = self._points_to_cam(center_lidar, data["lidar2cam"])
        x, y, z = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]
        radial = torch.sqrt(x * x + y * y + 1e-12)
        projection_type = cfg.get("projection_type")
        if projection_type is None and "kb_intrinsic" in data:
            projection_type = "kb_fisheye"
        if projection_type == "ocam_fisheye":
            incidence = torch.atan2(radial, -z)
            lens_key = "ocam_lens_fov_deg"
        else:
            incidence = torch.atan2(radial, z)
            lens_key = "kb_lens_fov_deg"

        max_incidence_deg = cfg.get("max_incidence_deg")
        if max_incidence_deg is None:
            lens_fov = data.get(lens_key, data.get("ocam_lens_fov_deg", 360.0))
            if isinstance(lens_fov, torch.Tensor):
                max_incidence_deg = float(lens_fov.flatten()[0].item()) / 2.0
            else:
                max_incidence_deg = float(lens_fov) / 2.0
        max_incidence = torch.deg2rad(
            center_lidar.new_tensor(float(max_incidence_deg))
        )
        confidence = (1.0 - incidence / max_incidence.clamp(min=1e-6)).clamp(
            0.0, 1.0
        )
        return torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_ray_aux_relation(self, center_lidar, data, cfg):
        pts_cam = self._points_to_cam(center_lidar, data["lidar2cam"])
        pts_cam_bn = pts_cam.transpose(1, 2)
        cam_range = torch.linalg.norm(pts_cam_bn, dim=-1).clamp(
            min=float(cfg["min_range"])
        )
        direction = pts_cam_bn / cam_range[..., None]
        range_norm = self._normalize_ray_range(cam_range, cfg).unsqueeze(-1)

        points_2d = self._project_ray_aux_points(
            center_lidar[:, :, None, :], data, cfg
        )
        points_2d = torch.nan_to_num(
            points_2d[..., 0, :], nan=-1.0, posinf=2.0, neginf=-1.0
        ).transpose(1, 2)
        xy_norm = ((points_2d - 0.5) * 2.0).clamp(-2.0, 2.0)
        boundary = torch.minimum(
            torch.minimum(points_2d[..., 0], 1.0 - points_2d[..., 0]),
            torch.minimum(points_2d[..., 1], 1.0 - points_2d[..., 1]),
        )
        boundary_conf = (
            boundary / max(float(cfg["boundary_margin"]), 1e-6)
        ).clamp(0.0, 1.0)
        center_dist = torch.linalg.norm(points_2d - 0.5, dim=-1) / (
            0.5 ** 0.5
        )
        center_conf = (1.0 - center_dist).clamp(0.0, 1.0)
        relation = torch.cat(
            [
                direction,
                range_norm,
                xy_norm,
                boundary_conf.unsqueeze(-1),
                center_conf.unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.nan_to_num(relation, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _points_to_cam(points_lidar, lidar2cam):
        bs, num_points = points_lidar.shape[:2]
        ones = points_lidar.new_ones(bs, num_points, 1)
        points_h = torch.cat([points_lidar, ones], dim=-1)
        points_h = points_h[:, None, :, :, None]
        points_cam = torch.matmul(lidar2cam[:, :, None], points_h).squeeze(-1)
        return points_cam[..., :3]

    @staticmethod
    def _normalize_ray_range(ray_range, cfg):
        min_range = float(cfg["min_range"])
        max_range = float(cfg["max_range"])
        min_log = ray_range.new_tensor(min_range).log()
        max_log = ray_range.new_tensor(max_range).log()
        ray_range = ray_range.clamp(min=min_range, max=max_range)
        return (ray_range.log() - min_log) / (max_log - min_log).clamp(min=1e-6)

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )
