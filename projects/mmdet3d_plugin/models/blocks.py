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
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

from mmcv.cnn import Linear
from mmengine.model import BaseModule, Sequential
from mmengine.model import xavier_init, constant_init

from mmdet3d.registry import MODELS
from projects.mmdet3d_plugin.core.box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW

try:
    from ..ops import deformable_aggregation_function as DAF
    from ..ops import feature_maps_format
except Exception:
    DAF = None
    feature_maps_format = None

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
            assert feature_maps_format is not None, (
                "feature_maps_format needs to be set up with deformable_aggregation."
            )
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.use_fisheye_geometry_attention = bool(use_fisheye_geometry_attention)
        default_geometry_cfg = dict(
            boundary_margin=0.08,
            boundary_weight=1.0,
            center_weight=0.5,
            angle_weight=1.0,
            relative_area_weight=0.25,
            short_axis_weight=0.75,
            aspect_weight=0.0,
            anisotropy_weight=0.5,
            learnable_quality_weights=True,
            learnable_quality_weight_names=None,
            max_learnable_quality_weight=1.0,
            min_quality_component=1e-4,
            short_axis_ref=0.04,
            max_aspect_ratio=4.0,
            max_anisotropy_ratio=4.0,
            support_bias_mode="log",
            bounded_penalty_init=0.05,
            bias_scale=1.0,
            learnable_bias_scale=True,
            positive_bias_scale=True,
            use_learned_view_confidence=True,
            use_view_feature_confidence=False,
            use_geometry_feature_adapter=False,
            use_view_relation_attention=False,
            view_relation_num_heads=4,
            view_relation_attn_drop=0.0,
            view_relation_use_support_bias=True,
            view_relation_support_bias_scale=1.0,
            view_relation_gate_bias=-2.0,
            detach_geometry=True,
            active_geometry_relation_names=None,
            use_camera_conditioned_keypoints=False,
            use_tangent_patch_sampling=False,
            tangent_patch_grid_size=3,
            tangent_patch_anchor_chunk_size=128,
            visible_surface_blend=1.0,
            visible_surface_eps=1e-6,
            learned_confidence_init=0.95,
            learned_confidence_ramp_start=0,
            learned_confidence_ramp_iters=1000,
            min_confidence=1e-4,
            min_support=1e-4,
            max_incidence_deg=95.0,
        )
        if geometry_attention_cfg is not None:
            default_geometry_cfg.update(geometry_attention_cfg)
        self.geometry_attention_cfg = default_geometry_cfg
        if self.geometry_attention_cfg["support_bias_mode"] not in (
            "log",
            "bounded_penalty",
        ):
            raise ValueError(
                "support_bias_mode must be 'log' or 'bounded_penalty', "
                f"but got {self.geometry_attention_cfg['support_bias_mode']}"
            )
        self.use_learned_view_confidence = bool(
            self.geometry_attention_cfg["use_learned_view_confidence"]
        )
        self.use_view_relation_attention = bool(
            self.use_fisheye_geometry_attention
            and self.geometry_attention_cfg["use_view_relation_attention"]
        )
        if self.use_view_relation_attention and self.use_deformable_func:
            raise ValueError(
                "Fisheye view relation attention requires "
                "use_deformable_func=False so per-camera object features are "
                "available before fusion."
            )
        self.geometry_relation_dims = 10
        self.quality_weight_names = (
            "boundary_weight",
            "center_weight",
            "angle_weight",
            "relative_area_weight",
            "short_axis_weight",
            "aspect_weight",
            "anisotropy_weight",
        )
        self.geometry_relation_names = (
            "valid",
            "boundary",
            "center",
            "angle",
            "x_offset",
            "y_offset",
            "relative_area",
            "short_axis",
            "aspect",
            "anisotropy",
        )
        learnable_quality_weight_names = self.geometry_attention_cfg[
            "learnable_quality_weight_names"
        ]
        if learnable_quality_weight_names is None:
            self.learnable_quality_weight_names = self.quality_weight_names
        else:
            unknown_names = set(learnable_quality_weight_names) - set(
                self.quality_weight_names
            )
            if unknown_names:
                raise ValueError(
                    "Unknown learnable quality weights: "
                    f"{sorted(unknown_names)}"
                )
            self.learnable_quality_weight_names = tuple(
                learnable_quality_weight_names
            )
        self.quality_weight_name_to_logit = {
            name: idx for idx, name in enumerate(self.learnable_quality_weight_names)
        }
        active_geometry_relation_names = self.geometry_attention_cfg[
            "active_geometry_relation_names"
        ]
        if active_geometry_relation_names is None:
            active_geometry_relation_names = self.geometry_relation_names
        else:
            unknown_names = set(active_geometry_relation_names) - set(
                self.geometry_relation_names
            )
            if unknown_names:
                raise ValueError(
                    "Unknown active geometry relation names: "
                    f"{sorted(unknown_names)}"
                )
            active_geometry_relation_names = tuple(active_geometry_relation_names)
        self.active_geometry_relation_names = active_geometry_relation_names
        relation_mask = [
            1.0 if name in active_geometry_relation_names else 0.0
            for name in self.geometry_relation_names
        ]
        self.register_buffer(
            "_geometry_relation_mask",
            torch.tensor(relation_mask, dtype=torch.float32),
            persistent=False,
        )
        if self.use_fisheye_geometry_attention and bool(
            self.geometry_attention_cfg["learnable_quality_weights"]
        ) and len(self.learnable_quality_weight_names) > 0:
            max_weight = max(
                float(self.geometry_attention_cfg["max_learnable_quality_weight"]),
                1e-6,
            )
            init_weights = [
                float(self.geometry_attention_cfg[name])
                for name in self.learnable_quality_weight_names
            ]
            init_weights = torch.tensor(init_weights, dtype=torch.float32)
            init_ratio = (init_weights / max_weight).clamp(1e-4, 1.0 - 1e-4)
            init_logits = torch.log(init_ratio / (1.0 - init_ratio))
            self.quality_weight_logits = nn.Parameter(init_logits)
        else:
            self.quality_weight_logits = None
        if self.use_fisheye_geometry_attention and bool(
            self.geometry_attention_cfg["learnable_bias_scale"]
        ):
            bias_scale = float(self.geometry_attention_cfg["bias_scale"])
            if bool(self.geometry_attention_cfg["positive_bias_scale"]):
                bias_scale = max(bias_scale, 1e-6)
                bias_scale = float(np.log(np.expm1(bias_scale)))
            self.geometry_bias_scale = nn.Parameter(
                torch.tensor(
                    bias_scale,
                    dtype=torch.float32,
                )
            )
        else:
            self.geometry_bias_scale = None
        if self.use_fisheye_geometry_attention and self.use_learned_view_confidence:
            hidden_dims = max(32, embed_dims // 4)
            self.view_confidence_mlp = nn.Sequential(
                Linear(self.geometry_relation_dims, hidden_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dims),
                Linear(hidden_dims, num_groups),
            )
            if bool(self.geometry_attention_cfg["use_view_feature_confidence"]):
                self.query_confidence_proj = Linear(embed_dims, num_groups)
                self.view_confidence_proj = Linear(embed_dims, num_groups)
            else:
                self.query_confidence_proj = None
                self.view_confidence_proj = None
            self._init_view_confidence_mlp()
        else:
            self.view_confidence_mlp = None
            self.query_confidence_proj = None
            self.view_confidence_proj = None
        if (
            self.use_fisheye_geometry_attention
            and self.geometry_attention_cfg["support_bias_mode"] == "bounded_penalty"
        ):
            hidden_dims = max(32, embed_dims // 4)
            self.view_penalty_mlp = nn.Sequential(
                Linear(self.geometry_relation_dims, hidden_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dims),
                Linear(hidden_dims, num_groups),
            )
            self._init_view_penalty_mlp()
        else:
            self.view_penalty_mlp = None
        if self.use_fisheye_geometry_attention and bool(
            self.geometry_attention_cfg["use_geometry_feature_adapter"]
        ):
            hidden_dims = max(32, embed_dims // 4)
            self.geometry_feature_adapter = nn.Sequential(
                Linear(self.geometry_relation_dims, hidden_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dims),
                Linear(hidden_dims, num_groups * 2),
            )
            self._init_geometry_feature_adapter()
        else:
            self.geometry_feature_adapter = None
        if self.use_view_relation_attention:
            num_heads = int(self.geometry_attention_cfg["view_relation_num_heads"])
            if embed_dims % num_heads != 0:
                raise ValueError(
                    "embed_dims must be divisible by view_relation_num_heads, "
                    f"but got {embed_dims} and {num_heads}"
                )
            self.view_relation_num_heads = num_heads
            self.view_relation_head_dims = embed_dims // num_heads
            self.view_relation_norm = nn.LayerNorm(embed_dims)
            self.view_relation_q = Linear(embed_dims, embed_dims)
            self.view_relation_k = Linear(embed_dims, embed_dims)
            self.view_relation_v = Linear(embed_dims, embed_dims)
            self.view_relation_out = Linear(embed_dims, embed_dims)
            self.view_relation_gate = Linear(embed_dims * 2, embed_dims)
            self._init_view_relation_attention()
        else:
            self.view_relation_num_heads = 0
            self.view_relation_head_dims = 0
            self.view_relation_norm = None
            self.view_relation_q = None
            self.view_relation_k = None
            self.view_relation_v = None
            self.view_relation_out = None
            self.view_relation_gate = None
        self.register_buffer(
            "_support_step",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.use_camera_conditioned_keypoints = bool(
            self.geometry_attention_cfg["use_camera_conditioned_keypoints"]
        )
        self.use_tangent_patch_sampling = bool(
            self.geometry_attention_cfg["use_tangent_patch_sampling"]
        )
        if self.use_tangent_patch_sampling and self.use_deformable_func:
            raise ValueError(
                "Tangent patch sampling expands each keypoint into a local "
                "fisheye patch and therefore requires use_deformable_func=False."
            )
        self._init_tangent_patch_grid()
        self._last_geometry_confidence = None
        self._last_geometry_relation = None
        self._last_learned_confidence = None
        self._last_view_support = None
        self._last_view_penalty = None
        self._last_view_features = None
        self._last_view_relation_weights = None
        self._last_view_relation_gate = None
        self._last_base_key_points = None
        self._last_sampling_key_points = None
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

    def _init_tangent_patch_grid(self):
        grid_size = int(self.geometry_attention_cfg["tangent_patch_grid_size"])
        if grid_size < 1 or grid_size % 2 == 0:
            raise ValueError(
                "tangent_patch_grid_size must be a positive odd integer, "
                f"but got {grid_size}"
            )
        coords = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        offsets = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

        order = grid_size - 1
        idx = torch.arange(grid_size, dtype=torch.float32)
        order_tensor = torch.tensor(float(order), dtype=torch.float32)
        log_coef = (
            torch.lgamma(order_tensor + 1.0)
            - torch.lgamma(idx + 1.0)
            - torch.lgamma(order_tensor - idx + 1.0)
        )
        weights_1d = torch.exp(log_coef)
        weights = (weights_1d[:, None] * weights_1d[None]).reshape(-1)
        weights = weights / weights.sum().clamp(min=1e-6)
        self.register_buffer(
            "_tangent_patch_offsets", offsets, persistent=False
        )
        self.register_buffer(
            "_tangent_patch_weights", weights, persistent=False
        )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._init_view_confidence_mlp()
        self._init_view_penalty_mlp()
        self._init_view_relation_attention()

    def _init_view_confidence_mlp(self):
        if self.view_confidence_mlp is not None:
            xavier_init(self.view_confidence_mlp[0], distribution="uniform", bias=0.0)
            init_conf = float(self.geometry_attention_cfg["learned_confidence_init"])
            init_conf = min(max(init_conf, 1e-4), 1.0 - 1e-4)
            init_bias = float(np.log(init_conf / (1.0 - init_conf)))
            constant_init(self.view_confidence_mlp[-1], val=0.0, bias=init_bias)
            if self.query_confidence_proj is not None:
                constant_init(self.query_confidence_proj, val=0.0, bias=0.0)
            if self.view_confidence_proj is not None:
                constant_init(self.view_confidence_proj, val=0.0, bias=0.0)

    def _init_view_penalty_mlp(self):
        if self.view_penalty_mlp is not None:
            xavier_init(self.view_penalty_mlp[0], distribution="uniform", bias=0.0)
            init_badness = float(self.geometry_attention_cfg["bounded_penalty_init"])
            init_badness = min(max(init_badness, 1e-4), 1.0 - 1e-4)
            init_bias = float(np.log(init_badness / (1.0 - init_badness)))
            constant_init(self.view_penalty_mlp[-1], val=0.0, bias=init_bias)

    def _init_geometry_feature_adapter(self):
        if self.geometry_feature_adapter is not None:
            xavier_init(
                self.geometry_feature_adapter[0], distribution="uniform", bias=0.0
            )
            constant_init(self.geometry_feature_adapter[-1], val=0.0, bias=0.0)

    def _init_view_relation_attention(self):
        if self.view_relation_q is None:
            return
        xavier_init(self.view_relation_q, distribution="uniform", bias=0.0)
        xavier_init(self.view_relation_k, distribution="uniform", bias=0.0)
        xavier_init(self.view_relation_v, distribution="uniform", bias=0.0)
        xavier_init(self.view_relation_out, distribution="uniform", bias=0.0)
        gate_bias = float(self.geometry_attention_cfg["view_relation_gate_bias"])
        constant_init(self.view_relation_gate, val=0.0, bias=gate_bias)

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
        sampling_key_points = self._get_camera_conditioned_key_points(
            key_points, anchor, metas
        )
        self._last_base_key_points = key_points.detach()
        self._last_sampling_key_points = sampling_key_points.detach()
        points_2d = None
        geometry_confidence = None
        geometry_relation = None
        if self.use_fisheye_geometry_attention:
            points_2d = self.project_points(
                sampling_key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
                metas=metas,
            )
            geometry_key_points = sampling_key_points
            geometry_points_2d = points_2d
            if bool(self.geometry_attention_cfg["detach_geometry"]):
                # Detach the projected geometry inputs so support learning does
                # not pull anchors/keypoints through projection geometry. Do not
                # detach the support output itself: learnable quality weights and
                # bias gates must still receive gradients from the attention loss.
                geometry_key_points = geometry_key_points.detach()
                geometry_points_2d = geometry_points_2d.detach()
            geometry_confidence, geometry_relation = self._get_geometry_confidence(
                geometry_key_points, geometry_points_2d, metas
            )
            self._last_geometry_confidence = geometry_confidence.detach()
            self._last_geometry_relation = geometry_relation.detach()
        else:
            self._last_geometry_confidence = None
            self._last_geometry_relation = None
            self._last_learned_confidence = None
            self._last_view_support = None
            self._last_view_penalty = None
            self._last_view_features = None
            self._last_view_relation_weights = None
            self._last_view_relation_gate = None

        sampled_features = None
        view_context = None
        if not self.use_deformable_func:
            if self.use_tangent_patch_sampling:
                sampled_features = self.tangent_patch_feature_sampling(
                    feature_maps,
                    sampling_key_points,
                    anchor,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                    metas=metas,
                    projection_type=self.projection_type,
                )
            else:
                sampled_features = self.feature_sampling(
                    feature_maps,
                    sampling_key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                    metas=metas,
                    projection_type=self.projection_type,
                )
            view_context = self._summarize_view_features(
                sampled_features, geometry_confidence
            )

        weight_geometry_confidence = geometry_confidence
        weight_geometry_relation = geometry_relation
        if (
            self.use_fisheye_geometry_attention
            and bool(self.geometry_attention_cfg["detach_geometry"])
        ):
            if (
                weight_geometry_confidence is not None
                and not bool(self.geometry_attention_cfg["learnable_quality_weights"])
            ):
                weight_geometry_confidence = weight_geometry_confidence.detach()
            if weight_geometry_relation is not None:
                weight_geometry_relation = weight_geometry_relation.detach()

        weights = self._get_weights(
            instance_feature,
            anchor_embed,
            metas,
            weight_geometry_confidence,
            weight_geometry_relation,
            view_context,
        )

        if self.use_deformable_func:
            self._last_view_features = None
            if points_2d is None:
                points_2d = self.project_points(
                    sampling_key_points,
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
            if (
                isinstance(feature_maps, (list, tuple))
                and len(feature_maps) == 3
                and torch.is_tensor(feature_maps[0])
                and torch.is_tensor(feature_maps[1])
                and torch.is_tensor(feature_maps[2])
                and feature_maps[0].dim() == 3
                and feature_maps[1].dim() == 3
                and feature_maps[2].dim() == 2
            ):
                col_feats, spatial_shape, scale_start_index = feature_maps
            else:
                col_feats, spatial_shape, scale_start_index = feature_maps_format(
                    feature_maps
                )
            features = DAF(
                col_feats,
                spatial_shape,
                scale_start_index,
                points_2d,
                weights,
            ).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features, view_features = self.multi_view_level_fusion(
                sampled_features,
                weights,
                geometry_relation=geometry_relation,
                return_view_features=True,
            )
            if self.use_view_relation_attention:
                view_features = self._apply_view_relation_attention(
                    view_features,
                    geometry_confidence=geometry_confidence,
                    view_support=self._last_view_support,
                )
                features = view_features.sum(dim=2)
            else:
                self._last_view_relation_weights = None
                self._last_view_relation_gate = None
                features = features.sum(dim=2)
            self._last_view_features = view_features.detach()
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_camera_conditioned_key_points(self, key_points, anchor, metas):
        """Return sampling keypoints, optionally adjusted per camera.

        The camera-conditioned/local-patch path is a scaffold and is off in the
        current configs. Preserve the standard Sparse4D keypoints unless that
        ablation is explicitly enabled.
        """
        if not self.use_camera_conditioned_keypoints:
            return key_points
        raise NotImplementedError(
            "use_camera_conditioned_keypoints requires camera-specific keypoint "
            "offsets before it can be enabled."
        )

    def _build_tangent_patch_key_points(self, key_points, anchor, metas):
        """Expand each 3D keypoint into a camera-ray tangent patch.

        The returned tensor is camera-conditioned: each camera gets its own
        local tangent grid because the same 3D point has a different fisheye ray
        and local image geometry in each view.
        """
        lidar2cam = metas["lidar2cam"]
        bs, num_cam = lidar2cam.shape[:2]
        num_anchor, num_pts = key_points.shape[1:3]

        ones = key_points.new_ones(bs, num_anchor, num_pts, 1)
        pts_h = torch.cat([key_points, ones], dim=-1)
        pts_h = pts_h[:, None, ..., None]
        pts_cam = torch.matmul(
            lidar2cam[:, :, None, None], pts_h
        ).squeeze(-1)[..., :3]

        cam_range = torch.linalg.norm(pts_cam, dim=-1).clamp(min=1e-4)
        ray = pts_cam / cam_range[..., None]

        radial_axis, tangential_axis = self._get_tangent_patch_axes(ray)

        radius = self._get_tangent_patch_radius(cam_range, anchor, key_points)
        radial_scale, tangential_scale = self._get_tangent_patch_projection_scales(
            ray, metas
        )
        scale_sum = (radial_scale + tangential_scale).clamp(min=1e-6)
        radius_radial = radius * (2.0 * tangential_scale / scale_sum)
        radius_tangential = radius * (2.0 * radial_scale / scale_sum)
        offsets = self._tangent_patch_offsets.to(
            device=key_points.device, dtype=key_points.dtype
        )
        angle_radial = radius_radial[..., None] * offsets[:, 0]
        angle_tangential = radius_tangential[..., None] * offsets[:, 1]
        dir_patch = (
            ray[..., None, :]
            + torch.tan(angle_radial)[..., None] * radial_axis[..., None, :]
            + torch.tan(angle_tangential)[..., None]
            * tangential_axis[..., None, :]
        )
        dir_patch = F.normalize(dir_patch, dim=-1, eps=1e-6)
        patch_cam = dir_patch * cam_range[..., None, None]

        cam2lidar = torch.linalg.inv(lidar2cam)
        patch_ones = patch_cam.new_ones(*patch_cam.shape[:-1], 1)
        patch_cam_h = torch.cat([patch_cam, patch_ones], dim=-1)[..., None]
        patch_lidar_h = torch.matmul(
            cam2lidar[:, :, None, None, None], patch_cam_h
        ).squeeze(-1)
        patch_lidar = patch_lidar_h[..., :3] / patch_lidar_h[
            ..., 3:4
        ].clamp(min=1e-6)
        patch_lidar = patch_lidar.reshape(
            bs, num_cam, num_anchor, num_pts * offsets.shape[0], 3
        )
        patch_weights = self._tangent_patch_weights.to(
            device=key_points.device, dtype=key_points.dtype
        )
        return patch_lidar, patch_weights, offsets.shape[0]

    def _get_tangent_patch_axes(self, ray):
        if self.projection_type == "ocam_fisheye":
            forward = ray.new_tensor([0.0, 0.0, -1.0])
        else:
            forward = ray.new_tensor([0.0, 0.0, 1.0])
        forward = forward.view(1, 1, 1, 1, 3)
        ray_forward = (ray * forward).sum(dim=-1, keepdim=True)
        radial_axis = ray_forward * ray - forward
        radial_norm = torch.linalg.norm(radial_axis, dim=-1, keepdim=True)

        ref_x = ray.new_tensor([1.0, 0.0, 0.0]).view(1, 1, 1, 1, 3)
        fallback = ref_x - (ref_x * ray).sum(dim=-1, keepdim=True) * ray
        fallback_norm = torch.linalg.norm(fallback, dim=-1, keepdim=True)
        ref_y = ray.new_tensor([0.0, 1.0, 0.0]).view(1, 1, 1, 1, 3)
        fallback_y = ref_y - (ref_y * ray).sum(dim=-1, keepdim=True) * ray
        fallback = torch.where(
            fallback_norm > 1e-6,
            fallback / fallback_norm.clamp(min=1e-6),
            F.normalize(fallback_y, dim=-1, eps=1e-6),
        )
        radial_axis = torch.where(
            radial_norm > 1e-6,
            radial_axis / radial_norm.clamp(min=1e-6),
            fallback,
        )
        tangential_axis = F.normalize(
            torch.cross(ray, radial_axis, dim=-1), dim=-1, eps=1e-6
        )
        return radial_axis, tangential_axis

    def _get_tangent_patch_radius(self, cam_range, anchor, key_points):
        bs, num_anchor, num_pts = key_points.shape[:3]
        if num_pts > 1:
            flat_points = key_points.reshape(bs * num_anchor, num_pts, 3)
            pair_dist = torch.cdist(flat_points, flat_points, p=2)
            diag = torch.eye(num_pts, device=key_points.device, dtype=torch.bool)
            pair_dist = pair_dist.masked_fill(
                diag[None], torch.finfo(pair_dist.dtype).max
            )
            local_radius = pair_dist.min(dim=-1).values.reshape(
                bs, num_anchor, num_pts
            )
            local_radius = local_radius * 0.5
        else:
            size = anchor[..., [W, L, H]].exp().clamp(min=1e-3, max=100.0)
            local_radius = torch.amin(size, dim=-1, keepdim=True) * 0.5
        radius = torch.atan(local_radius[:, None] / cam_range.clamp(min=1e-4))
        return torch.nan_to_num(radius, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_tangent_patch_projection_scales(self, ray, metas):
        if self.projection_type == "kb_fisheye":
            return self._get_kb_patch_projection_scales(ray, metas)
        if self.projection_type == "ocam_fisheye":
            return self._get_ocam_patch_projection_scales(ray, metas)
        shape = ray.shape[:-1]
        return ray.new_ones(shape), ray.new_ones(shape)

    def _get_kb_patch_projection_scales(self, ray, metas):
        distortion = metas.get("kb_distortion", metas.get("ocam_pol"))
        if distortion is None:
            shape = ray.shape[:-1]
            return ray.new_ones(shape), ray.new_ones(shape)

        x = ray[..., 0]
        y = ray[..., 1]
        z = ray[..., 2]
        radial = torch.sqrt(x * x + y * y + 1e-12)
        theta = torch.atan2(radial, z)

        distortion = distortion[..., :4].to(device=theta.device, dtype=theta.dtype)
        k1 = distortion[..., 0][:, :, None, None]
        k2 = distortion[..., 1][:, :, None, None]
        k3 = distortion[..., 2][:, :, None, None]
        k4 = distortion[..., 3][:, :, None, None]
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        theta_d = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)
        radial_scale = torch.abs(
            1.0
            + 3.0 * k1 * t2
            + 5.0 * k2 * t4
            + 7.0 * k3 * t6
            + 9.0 * k4 * t8
        )
        sin_theta = torch.sin(theta).abs()
        tangential_scale = torch.where(
            sin_theta > 1e-4,
            torch.abs(theta_d) / sin_theta.clamp(min=1e-4),
            radial_scale,
        )
        return self._sanitize_patch_projection_scales(radial_scale, tangential_scale)

    def _get_ocam_patch_projection_scales(self, ray, metas):
        pol = metas.get("ocam_pol")
        if pol is None:
            shape = ray.shape[:-1]
            return ray.new_ones(shape), ray.new_ones(shape)

        x = ray[..., 0]
        y = ray[..., 1]
        z = ray[..., 2]
        norm = torch.sqrt(x * x + y * y + 1e-12)
        theta = torch.atan2(z, norm)
        alpha = torch.atan2(norm, -z)

        pol = pol.to(device=theta.device, dtype=theta.dtype)
        coefs = pol.unbind(dim=-1)
        rho = coefs[-1][:, :, None, None].expand_as(theta).clone()
        for idx in range(len(coefs) - 2, -1, -1):
            rho = rho * theta + coefs[idx][:, :, None, None]

        if len(coefs) > 1:
            deriv = torch.zeros_like(theta)
            for idx in range(1, len(coefs)):
                deriv = deriv + idx * coefs[idx][:, :, None, None] * (
                    theta ** (idx - 1)
                )
            radial_scale = torch.abs(deriv)
        else:
            radial_scale = torch.ones_like(theta)
        sin_alpha = torch.sin(alpha).abs()
        tangential_scale = torch.where(
            sin_alpha > 1e-4,
            torch.abs(rho) / sin_alpha.clamp(min=1e-4),
            radial_scale,
        )
        return self._sanitize_patch_projection_scales(radial_scale, tangential_scale)

    @staticmethod
    def _sanitize_patch_projection_scales(radial_scale, tangential_scale):
        radial_scale = torch.nan_to_num(
            radial_scale, nan=1.0, posinf=1.0, neginf=1.0
        ).abs()
        tangential_scale = torch.nan_to_num(
            tangential_scale, nan=1.0, posinf=1.0, neginf=1.0
        ).abs()
        valid = (radial_scale > 1e-6) & (tangential_scale > 1e-6)
        radial_scale = torch.where(valid, radial_scale, torch.ones_like(radial_scale))
        tangential_scale = torch.where(
            valid, tangential_scale, torch.ones_like(tangential_scale)
        )
        return radial_scale.clamp(min=1e-6), tangential_scale.clamp(min=1e-6)

    def _get_weights(
        self,
        instance_feature,
        anchor_embed,
        metas=None,
        geometry_confidence=None,
        geometry_relation=None,
        view_features=None,
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
            view_support = self._get_view_support(
                geometry_confidence,
                geometry_relation,
                instance_feature=instance_feature,
                view_features=view_features,
            )
            bias_scale = self._get_geometry_bias_scale(
                geometry_confidence.device, geometry_confidence.dtype
            )
            if self.geometry_attention_cfg["support_bias_mode"] == "log":
                min_conf = float(self.geometry_attention_cfg["min_confidence"])
                support_bias = torch.log(view_support.clamp(min=min_conf)) * bias_scale
                self._last_view_penalty = None
            else:
                support_bias = self._get_bounded_penalty_bias(
                    geometry_confidence,
                    geometry_relation,
                    max_penalty=bias_scale,
                )
            logits = logits + support_bias[:, :, :, None]
            self._last_view_support = view_support.detach()
        else:
            self._last_learned_confidence = None
            self._last_view_support = None
            self._last_view_penalty = None
            self._last_view_features = None

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

    def _get_geometry_bias_scale(self, device, dtype):
        if self.geometry_bias_scale is not None:
            bias_scale = self.geometry_bias_scale.to(device=device, dtype=dtype)
            if bool(self.geometry_attention_cfg["positive_bias_scale"]):
                bias_scale = F.softplus(bias_scale)
            return bias_scale
        return torch.tensor(
            float(self.geometry_attention_cfg["bias_scale"]),
            device=device,
            dtype=dtype,
        )

    def _get_quality_weight(self, name, ref_tensor):
        cfg = self.geometry_attention_cfg
        if self.quality_weight_logits is None or (
            name not in self.quality_weight_name_to_logit
        ):
            return ref_tensor.new_tensor(float(cfg[name]))
        idx = self.quality_weight_name_to_logit[name]
        max_weight = float(cfg["max_learnable_quality_weight"])
        weight = torch.sigmoid(
            self.quality_weight_logits[idx].to(
                device=ref_tensor.device, dtype=ref_tensor.dtype
            )
        )
        return weight * max(max_weight, 1e-6)

    def _apply_quality_weight(self, confidence, component, name):
        weight = self._get_quality_weight(name, confidence)
        min_component = float(self.geometry_attention_cfg["min_quality_component"])
        component = component.clamp(min=min_component, max=1.0)
        return confidence * component.pow(weight)

    def _get_learned_confidence_lambda(self, ref_tensor):
        if self.view_confidence_mlp is None:
            return ref_tensor.new_tensor(0.0)
        if not self.training:
            return ref_tensor.new_tensor(1.0)
        start = int(self.geometry_attention_cfg["learned_confidence_ramp_start"])
        ramp_iters = int(self.geometry_attention_cfg["learned_confidence_ramp_iters"])
        step = int(self._support_step.item())
        with torch.no_grad():
            self._support_step += 1
        if ramp_iters <= 0:
            return ref_tensor.new_tensor(1.0)
        progress = (step - start) / float(ramp_iters)
        progress = min(max(progress, 0.0), 1.0)
        return ref_tensor.new_tensor(progress)

    def _get_bounded_penalty_bias(
        self,
        geometry_confidence,
        geometry_relation=None,
        max_penalty=None,
    ):
        if self.view_penalty_mlp is not None and geometry_relation is not None:
            penalty_logits = self.view_penalty_mlp(geometry_relation)
            badness = torch.sigmoid(penalty_logits)
        else:
            badness = 1.0 - geometry_confidence[..., None].clamp(0.0, 1.0)
            badness = badness.expand(*geometry_confidence.shape, self.num_groups)
        valid = (geometry_confidence > 0)[..., None]
        badness = torch.where(valid, badness, torch.ones_like(badness))
        badness = torch.nan_to_num(
            badness, nan=1.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        if max_penalty is None:
            max_penalty = self._get_geometry_bias_scale(
                geometry_confidence.device, geometry_confidence.dtype
            )
        bias = -badness * max_penalty
        self._last_view_penalty = badness.detach()
        return bias

    def _get_view_support(
        self,
        geometry_confidence,
        geometry_relation=None,
        instance_feature=None,
        view_features=None,
    ):
        cfg = self.geometry_attention_cfg
        min_support = float(cfg["min_support"])
        base_support = torch.where(
            geometry_confidence > 0,
            geometry_confidence.clamp(min=min_support),
            geometry_confidence.new_zeros(()),
        )
        support = base_support[..., None].expand(
            *base_support.shape, self.num_groups
        )
        learned_confidence = None
        if self.view_confidence_mlp is not None and geometry_relation is not None:
            learned_logits = self.view_confidence_mlp(geometry_relation)
            if instance_feature is not None and self.query_confidence_proj is not None:
                learned_logits = learned_logits + self.query_confidence_proj(
                    instance_feature
                )[:, :, None, None]
            if view_features is not None and self.view_confidence_proj is not None:
                learned_logits = learned_logits + self.view_confidence_proj(
                    view_features
                )[:, :, :, None]
            learned_confidence = torch.sigmoid(learned_logits)
            lam = self._get_learned_confidence_lambda(geometry_confidence)
            support = support * ((1.0 - lam) + lam * learned_confidence)
            self._last_learned_confidence = learned_confidence.detach()
        else:
            self._last_learned_confidence = None
        support = torch.nan_to_num(
            support, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        return support

    def _get_geometry_confidence(self, key_points, points_2d, metas):
        """Build a geometry-derived attention prior for each view/keypoint.

        Returns:
            confidence: (B, N_anchor, N_cam, N_pts), in [0, 1].
            relation: (B, N_anchor, N_cam, N_pts, 10).
        """
        cfg = self.geometry_attention_cfg
        finite = torch.isfinite(points_2d).all(dim=-1)
        safe_points_2d = torch.nan_to_num(
            points_2d, nan=-1.0, posinf=2.0, neginf=-1.0
        )
        valid = (safe_points_2d[..., 0] >= 0.0) & (
            safe_points_2d[..., 0] <= 1.0
        )
        valid = valid & (safe_points_2d[..., 1] >= 0.0) & (
            safe_points_2d[..., 1] <= 1.0
        )
        valid = valid & finite

        boundary = torch.minimum(
            torch.minimum(
                safe_points_2d[..., 0], 1.0 - safe_points_2d[..., 0]
            ),
            torch.minimum(
                safe_points_2d[..., 1], 1.0 - safe_points_2d[..., 1]
            ),
        )
        boundary_margin = float(cfg["boundary_margin"])
        boundary_conf = (boundary / max(boundary_margin, 1e-6)).clamp(0.0, 1.0)

        center_dist = torch.linalg.norm(
            safe_points_2d - 0.5, dim=-1
        ) / (0.5 ** 0.5)
        center_conf = (1.0 - center_dist).clamp(0.0, 1.0)

        angle_conf = self._get_incidence_confidence(key_points, metas)
        angle_conf = torch.nan_to_num(
            angle_conf, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)

        view_valid = valid.to(dtype=points_2d.dtype)
        valid_count = view_valid.sum(dim=-1)
        inf = torch.full_like(safe_points_2d, float("inf"))
        neg_inf = torch.full_like(safe_points_2d, float("-inf"))
        min_xy = torch.where(valid[..., None], safe_points_2d, inf).amin(dim=-2)
        max_xy = torch.where(valid[..., None], safe_points_2d, neg_inf).amax(dim=-2)
        footprint_wh = (max_xy - min_xy).clamp(min=0.0)
        long_axis = footprint_wh.amax(dim=-1)
        short_axis = footprint_wh.amin(dim=-1)
        area = footprint_wh[..., 0] * footprint_wh[..., 1]
        has_footprint = valid_count >= 2
        area = torch.where(has_footprint, area, area.new_zeros(()))
        short_axis = torch.where(has_footprint, short_axis, short_axis.new_zeros(()))
        max_area = area.amax(dim=1, keepdim=True).clamp(min=1e-6)
        relative_area_conf = (area / max_area).clamp(0.0, 1.0)
        short_axis_ref = max(float(cfg["short_axis_ref"]), 1e-6)
        short_axis_conf = (short_axis / short_axis_ref).clamp(0.0, 1.0)
        aspect = long_axis / short_axis.clamp(min=1e-6)
        max_aspect = max(float(cfg["max_aspect_ratio"]), 1.0 + 1e-6)
        aspect_conf = ((max_aspect - aspect) / (max_aspect - 1.0)).clamp(0.0, 1.0)
        aspect_conf = torch.where(has_footprint, aspect_conf, aspect_conf.new_zeros(()))
        anisotropy_conf = self._get_kb_anisotropy_confidence(key_points, metas)
        anisotropy_conf = torch.nan_to_num(
            anisotropy_conf, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)

        confidence = valid.to(dtype=points_2d.dtype)
        confidence = self._apply_quality_weight(
            confidence, boundary_conf, "boundary_weight"
        )
        confidence = self._apply_quality_weight(
            confidence, center_conf, "center_weight"
        )
        confidence = self._apply_quality_weight(
            confidence, angle_conf, "angle_weight"
        )
        confidence = self._apply_quality_weight(
            confidence,
            relative_area_conf[..., None].expand_as(confidence),
            "relative_area_weight",
        )
        confidence = self._apply_quality_weight(
            confidence,
            short_axis_conf[..., None].expand_as(confidence),
            "short_axis_weight",
        )
        confidence = self._apply_quality_weight(
            confidence,
            aspect_conf[..., None].expand_as(confidence),
            "aspect_weight",
        )
        confidence = self._apply_quality_weight(
            confidence,
            anisotropy_conf,
            "anisotropy_weight",
        )
        confidence = torch.nan_to_num(
            confidence, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        relation = torch.stack(
            [
                valid.to(dtype=points_2d.dtype),
                boundary_conf,
                center_conf,
                angle_conf,
                (safe_points_2d[..., 0] - 0.5) * 2.0,
                (safe_points_2d[..., 1] - 0.5) * 2.0,
                relative_area_conf[..., None].expand_as(boundary_conf),
                short_axis_conf[..., None].expand_as(boundary_conf),
                aspect_conf[..., None].expand_as(boundary_conf),
                anisotropy_conf,
            ],
            dim=-1,
        )
        relation = torch.nan_to_num(
            relation, nan=0.0, posinf=0.0, neginf=0.0
        )
        relation = relation * self._geometry_relation_mask.to(
            device=relation.device, dtype=relation.dtype
        )
        confidence = confidence.permute(0, 2, 1, 3).contiguous()
        relation = relation.permute(0, 2, 1, 3, 4).contiguous()
        return confidence, relation

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
        confidence = (1.0 - incidence / max(max_incidence, 1e-6)).clamp(0.0, 1.0)
        return torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_kb_anisotropy_confidence(self, key_points, metas):
        """Compute local KB radial/tangential distortion balance.

        For KB projection r(theta)=theta_d, local angular scale differs between
        the radial direction and tangential direction:
            radial     = d theta_d / d theta
            tangential = theta_d / sin(theta)
        Their ratio is 1 for locally isotropic projection and grows when one
        direction is stretched or compressed more than the other.
        """
        if self.projection_type != "kb_fisheye":
            return key_points.new_ones(
                key_points.shape[0],
                self.num_cams,
                key_points.shape[1],
                key_points.shape[2],
            )
        lidar2cam = metas.get("lidar2cam")
        distortion = metas.get("kb_distortion", metas.get("ocam_pol"))
        if lidar2cam is None or distortion is None:
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
        theta = torch.atan2(radial, z)

        distortion = distortion[..., :4].to(device=theta.device, dtype=theta.dtype)
        k1 = distortion[..., 0][:, :, None, None]
        k2 = distortion[..., 1][:, :, None, None]
        k3 = distortion[..., 2][:, :, None, None]
        k4 = distortion[..., 3][:, :, None, None]
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        theta_d = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)
        radial_scale = torch.abs(
            1.0
            + 3.0 * k1 * t2
            + 5.0 * k2 * t4
            + 7.0 * k3 * t6
            + 9.0 * k4 * t8
        )
        sin_theta = torch.sin(theta).abs()
        tangential_scale = torch.where(
            sin_theta > 1e-4,
            torch.abs(theta_d) / sin_theta.clamp(min=1e-4),
            radial_scale,
        )
        min_scale = torch.minimum(radial_scale, tangential_scale).clamp(min=1e-6)
        max_scale = torch.maximum(radial_scale, tangential_scale)
        anisotropy = max_scale / min_scale
        max_anisotropy = max(
            float(self.geometry_attention_cfg["max_anisotropy_ratio"]),
            1.0 + 1e-6,
        )
        confidence = ((max_anisotropy - anisotropy) / (max_anisotropy - 1.0)).clamp(
            0.0, 1.0
        )
        return torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)

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

    def tangent_patch_feature_sampling(
        self,
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        anchor: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
        metas: Optional[dict] = None,
        projection_type: str = "pinhole",
    ) -> torch.Tensor:
        patch_key_points, patch_weights, num_patch = (
            self._build_tangent_patch_key_points(key_points, anchor, metas)
        )
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        if projection_type == "ocam_fisheye":
            from .projection import ocam_project_points

            points_2d = ocam_project_points(patch_key_points, metas, image_wh)
        elif projection_type == "kb_fisheye":
            from .projection import kb_project_points

            points_2d = kb_project_points(patch_key_points, metas, image_wh)
        else:
            raise ValueError(
                "Tangent patch sampling is defined for fisheye projection, "
                f"but got projection_type={projection_type}"
            )

        finite = torch.isfinite(points_2d).all(dim=-1)
        safe_points = torch.nan_to_num(
            points_2d, nan=2.0, posinf=2.0, neginf=-2.0
        )
        valid = (
            (safe_points[..., 0] >= 0.0)
            & (safe_points[..., 0] <= 1.0)
            & (safe_points[..., 1] >= 0.0)
            & (safe_points[..., 1] <= 1.0)
            & finite
        )
        grid = safe_points * 2 - 1
        valid = valid.reshape(bs, num_cams, num_anchor, num_pts, num_patch)
        patch_weights = patch_weights.view(1, 1, 1, 1, 1, num_patch, 1)
        chunk_size = int(
            self.geometry_attention_cfg["tangent_patch_anchor_chunk_size"]
        )
        if chunk_size <= 0:
            chunk_size = num_anchor

        sampled_chunks = []
        flat_feature_maps = [fm.flatten(end_dim=1) for fm in feature_maps]
        for start in range(0, num_anchor, chunk_size):
            end = min(start + chunk_size, num_anchor)
            grid_chunk = grid[:, :, start:end].flatten(end_dim=1)

            level_features = []
            for fm in flat_feature_maps:
                level_features.append(
                    torch.nn.functional.grid_sample(fm, grid_chunk)
                )
            features = torch.stack(level_features, dim=1)
            features = features.reshape(
                bs,
                num_cams,
                num_levels,
                -1,
                end - start,
                num_pts,
                num_patch,
            ).permute(0, 4, 1, 2, 5, 6, 3)

            valid_chunk = valid[:, :, start:end]
            valid_chunk = (
                valid_chunk.permute(0, 2, 1, 3, 4)
                .unsqueeze(3)
                .unsqueeze(-1)
            )
            weights = patch_weights * valid_chunk.to(dtype=features.dtype)
            denom = weights.sum(dim=5, keepdim=True).clamp(min=1e-6)
            sampled_chunks.append((features * weights).sum(dim=5) / denom.squeeze(5))

        return torch.cat(sampled_chunks, dim=1)

    def _apply_view_relation_attention(
        self,
        view_features: torch.Tensor,
        geometry_confidence: Optional[torch.Tensor] = None,
        view_support: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exchange information between camera-specific object features."""
        if self.view_relation_q is None:
            return view_features

        bs, num_anchor, num_cams, embed_dims = view_features.shape
        num_heads = self.view_relation_num_heads
        head_dims = self.view_relation_head_dims
        x = self.view_relation_norm(view_features)
        flat_shape = (bs * num_anchor, num_cams, num_heads, head_dims)
        q = self.view_relation_q(x).reshape(flat_shape).transpose(1, 2)
        k = self.view_relation_k(x).reshape(flat_shape).transpose(1, 2)
        v = self.view_relation_v(x).reshape(flat_shape).transpose(1, 2)

        scale = head_dims ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        valid_view = None
        if geometry_confidence is not None:
            valid_view = (geometry_confidence > 0).any(dim=-1)
            valid_key = valid_view | (~valid_view.any(dim=-1, keepdim=True))
            valid_key = valid_key.reshape(bs * num_anchor, num_cams)
            scores = scores.masked_fill(
                ~valid_key[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )

        if (
            view_support is not None
            and bool(self.geometry_attention_cfg["view_relation_use_support_bias"])
        ):
            support = view_support.mean(dim=(-1, -2))
            min_conf = float(self.geometry_attention_cfg["min_confidence"])
            support_bias_scale = float(
                self.geometry_attention_cfg["view_relation_support_bias_scale"]
            )
            support_bias = (
                torch.log(support.clamp(min=min_conf)) * support_bias_scale
            )
            scores = scores + support_bias.reshape(bs * num_anchor, 1, 1, num_cams)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        drop = float(self.geometry_attention_cfg["view_relation_attn_drop"])
        if self.training and drop > 0:
            attn = F.dropout(attn, p=drop, training=True)
        context = torch.matmul(attn, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .reshape(bs, num_anchor, num_cams, embed_dims)
        )
        delta = self.view_relation_out(context)
        gate = torch.sigmoid(
            self.view_relation_gate(torch.cat([view_features, context], dim=-1))
        )
        updated = view_features + gate * delta
        if valid_view is not None:
            updated = torch.where(valid_view[..., None], updated, view_features)
        self._last_view_relation_weights = attn.reshape(
            bs, num_anchor, num_heads, num_cams, num_cams
        ).detach()
        self._last_view_relation_gate = gate.detach()
        return updated

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        geometry_relation: Optional[torch.Tensor] = None,
        return_view_features: bool = False,
    ):
        bs, num_anchor = weights.shape[:2]
        features = features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        if self.geometry_feature_adapter is not None and geometry_relation is not None:
            adapter = self.geometry_feature_adapter(geometry_relation.to(features.dtype))
            adapter = adapter.reshape(*geometry_relation.shape[:-1], 2, self.num_groups)
            scale = torch.tanh(adapter[..., 0, :])[:, :, :, None, :, :, None]
            bias = adapter[..., 1, :][:, :, :, None, :, :, None]
            features = features * (1.0 + scale) + bias
        weighted_features = weights[..., None] * features
        if return_view_features:
            view_features = weighted_features.sum(dim=3).sum(dim=3)
            view_features = view_features.reshape(
                bs, num_anchor, self.num_cams, self.embed_dims
            )
        else:
            view_features = None
        fused = weighted_features.sum(dim=2).sum(dim=2)
        fused = fused.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        if return_view_features:
            return fused, view_features
        return fused

    @staticmethod
    def _summarize_view_features(features, geometry_confidence=None):
        if geometry_confidence is None:
            return features.mean(dim=(3, 4))
        mask = (geometry_confidence > 0).to(dtype=features.dtype)
        mask = mask[:, :, :, None, :, None]
        denom = (mask.sum(dim=(3, 4)) * features.shape[3]).clamp(min=1.0)
        return (features * mask).sum(dim=(3, 4)) / denom


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
