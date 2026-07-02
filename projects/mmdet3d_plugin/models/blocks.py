# ------------------------------------------------------------------------
# Ported from Sparse4D (HorizonRobotics) to mmcv 2.x / mmdet3d 1.4
# Changes:
#   ATTENTION/PLUGIN_LAYERS/FEEDFORWARD_NETWORK → mmdet3d MODELS
#   BaseModule from mmengine.model
#   build_from_cfg → MODELS.build
#   project_points retained pinhole baseline; OCamCalib variant lives in
#   `projects.mmdet3d_plugin.models.projection.fisheye_project_points`
#   and is swapped in by the head config.
# ------------------------------------------------------------------------
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast

from mmcv.cnn import Linear
from mmengine.model import BaseModule, Sequential
from mmengine.model import xavier_init, constant_init

from mmdet3d.registry import MODELS

try:
    from ..ops import deformable_aggregation_function as DAF
except Exception:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@MODELS.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        projection_type="pinhole",
        use_fisheye_geometry_attention=False,
        geometry_attention_cfg=None,
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.use_fisheye_geometry_attention = bool(use_fisheye_geometry_attention)
        default_geometry_cfg = dict(
            boundary_margin=0.08,
            boundary_weight=1.0,
            center_weight=0.5,
            angle_weight=1.0,
            bias_scale=1.0,
            min_confidence=1e-4,
            max_incidence_deg=None,
        )
        if geometry_attention_cfg is not None:
            default_geometry_cfg.update(geometry_attention_cfg)
        self.geometry_attention_cfg = default_geometry_cfg
        self._last_geometry_confidence = None
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = MODELS.build(kps_generator)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = MODELS.build(temporal_fusion_module)
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        assert projection_type in ("pinhole", "ocam_fisheye", "kb_fisheye"), (
            f"unknown projection_type={projection_type}"
        )
        self.projection_type = projection_type

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        points_2d = None
        geometry_confidence = None
        if self.use_fisheye_geometry_attention:
            points_2d = self.project_points(
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
                metas=metas,
            )
            geometry_confidence = self._get_geometry_confidence(
                key_points, points_2d, metas
            )
            self._last_geometry_confidence = geometry_confidence.detach()
        else:
            self._last_geometry_confidence = None

        weights = self._get_weights(
            instance_feature, anchor_embed, metas, geometry_confidence
        )

        if self.use_deformable_func:
            if points_2d is None:
                points_2d = self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                    metas=metas,
                )
            points_2d = (
                points_2d
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
                metas=metas,
                projection_type=self.projection_type,
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(
        self, instance_feature, anchor_embed, metas=None, geometry_confidence=None
    ):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        logits = self.weights_fc(feature).reshape(
            bs,
            num_anchor,
            self.num_cams,
            self.num_levels,
            self.num_pts,
            self.num_groups,
        )
        if geometry_confidence is not None:
            cfg = self.geometry_attention_cfg
            min_conf = float(cfg["min_confidence"])
            bias_scale = float(cfg["bias_scale"])
            geo_bias = torch.log(
                geometry_confidence.clamp(min=min_conf)
            ) * bias_scale
            logits = logits + geo_bias[:, :, :, None, :, None]

        weights = (
            logits.reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    def _get_geometry_confidence(self, key_points, points_2d, metas):
        """Build a geometry-derived attention prior for each view/keypoint.

        Returns:
            confidence: (B, N_anchor, N_cam, N_pts), in [0, 1].
        """
        cfg = self.geometry_attention_cfg
        valid = (points_2d[..., 0] >= 0.0) & (points_2d[..., 0] <= 1.0)
        valid = valid & (points_2d[..., 1] >= 0.0) & (points_2d[..., 1] <= 1.0)

        boundary = torch.minimum(
            torch.minimum(points_2d[..., 0], 1.0 - points_2d[..., 0]),
            torch.minimum(points_2d[..., 1], 1.0 - points_2d[..., 1]),
        )
        boundary_margin = float(cfg["boundary_margin"])
        boundary_conf = (boundary / max(boundary_margin, 1e-6)).clamp(0.0, 1.0)

        center_dist = torch.linalg.norm(points_2d - 0.5, dim=-1) / (0.5 ** 0.5)
        center_conf = (1.0 - center_dist).clamp(0.0, 1.0)

        angle_conf = self._get_incidence_confidence(key_points, metas)

        confidence = valid.to(dtype=points_2d.dtype)
        confidence = confidence * boundary_conf.pow(float(cfg["boundary_weight"]))
        confidence = confidence * center_conf.pow(float(cfg["center_weight"]))
        confidence = confidence * angle_conf.pow(float(cfg["angle_weight"]))
        return confidence.permute(0, 2, 1, 3).contiguous()

    def _get_incidence_confidence(self, key_points, metas):
        lidar2cam = metas.get("lidar2cam")
        if lidar2cam is None:
            return key_points.new_ones(
                key_points.shape[0],
                self.num_cams,
                key_points.shape[1],
                key_points.shape[2],
            )
        bs, num_anchor, num_pts = key_points.shape[:3]
        ones = key_points.new_ones(bs, num_anchor, num_pts, 1)
        pts_h = torch.cat([key_points, ones], dim=-1)[:, None, ..., None]
        pts_cam = torch.matmul(lidar2cam[:, :, None, None], pts_h).squeeze(-1)
        x = pts_cam[..., 0]
        y = pts_cam[..., 1]
        z = pts_cam[..., 2]
        radial = torch.sqrt(x * x + y * y + 1e-12)

        if self.projection_type == "ocam_fisheye":
            incidence = torch.atan2(radial, -z)
            lens_key = "ocam_lens_fov_deg"
        else:
            incidence = torch.atan2(radial, z)
            lens_key = "kb_lens_fov_deg"

        max_incidence_deg = self.geometry_attention_cfg.get("max_incidence_deg")
        if max_incidence_deg is None:
            lens_fov = metas.get(lens_key, metas.get("ocam_lens_fov_deg", 360.0))
            if isinstance(lens_fov, torch.Tensor):
                max_incidence_deg = float(lens_fov.flatten()[0].item()) / 2.0
            else:
                max_incidence_deg = float(lens_fov) / 2.0
        max_incidence = np.deg2rad(float(max_incidence_deg))
        return (1.0 - incidence / max(max_incidence, 1e-6)).clamp(0.0, 1.0)

    def project_points(
        self, key_points, projection_mat, image_wh=None, metas=None
    ):
        if self.projection_type == "ocam_fisheye":
            from .projection import ocam_project_points
            return ocam_project_points(
                key_points, metas, image_wh
            )
        if self.projection_type == "kb_fisheye":
            from .projection import kb_project_points
            return kb_project_points(
                key_points, metas, image_wh
            )
        return self._project_points_pinhole(key_points, projection_mat, image_wh)

    @staticmethod
    def _project_points_pinhole(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
        metas: Optional[dict] = None,
        projection_type: str = "pinhole",
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        if projection_type == "ocam_fisheye":
            from .projection import ocam_project_points
            points_2d = ocam_project_points(key_points, metas, image_wh)
        elif projection_type == "kb_fisheye":
            from .projection import kb_project_points
            points_2d = kb_project_points(key_points, metas, image_wh)
        else:
            points_2d = DeformableFeatureAggregation._project_points_pinhole(
                key_points, projection_mat, image_wh
            )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@MODELS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast("cuda", enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


@MODELS.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        from mmcv.cnn import build_activation_layer, build_norm_layer
        from mmcv.cnn.bricks.drop import build_dropout
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
