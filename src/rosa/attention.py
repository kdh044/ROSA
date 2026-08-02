"""Camera- and keypoint-level observation fusion."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from . import rfl as _rfl
from .config import ROSAConfig
from .rfl import SphericalRayFeatureLifting
from .types import ObservationEncoding, RayFeatureRelation


def masked_softmax(logits: Tensor, valid: Tensor, dim: int) -> Tensor:
    """Apply softmax without producing NaNs for fully masked rows."""

    if logits.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("masked softmax tensors are misaligned")
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked.float(), dim=dim).to(logits.dtype)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    denominator = weights.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > 0,
        weights / denominator.clamp_min(1e-8),
        torch.zeros_like(weights),
    )


def mean_pairwise_camera_cosine(values: Tensor, valid: Tensor) -> Tensor:
    """Average valid camera-pair cosine without materializing ``C x C``."""

    if values.ndim != 5 or valid.shape != values.shape[:-1]:
        raise ValueError("camera consistency tensors are misaligned")
    masked = values.permute(0, 1, 3, 2, 4) * valid.permute(0, 1, 3, 2)[..., None].to(
        values.dtype
    )
    camera_count = valid.sum(dim=2).to(values.dtype)
    pair_count = camera_count * (camera_count - 1.0) * 0.5
    summed = masked.sum(dim=3)
    pair_sum = (summed.square().sum(dim=-1) - masked.square().sum(dim=(-1, -2))) * 0.5
    return torch.where(
        pair_count > 0,
        pair_sum / pair_count.clamp_min(1.0),
        torch.zeros_like(pair_sum),
    )


def _is_owned_plain_fp32_tensor(
    value: Tensor,
    *,
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
    protected: tuple[Tensor, ...],
) -> bool:
    """Validate storage before a private eager in-place operation."""

    alias_check = getattr(torch._C, "_is_alias_of", None)
    if (
        alias_check is not _RELEASED_ALIAS_CHECK
        or type(value) is not torch.Tensor
        or value.shape != shape
        or value.dtype is not torch.float32
        or value.device != device
        or value.layout != torch.strided
        or value.is_quantized
        or value.requires_grad
        or value._base is not None
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or any(name is not None for name in value.names)
        or any(type(other) not in (torch.Tensor, nn.Parameter) for other in protected)
    ):
        return False
    try:
        return not any(alias_check(value, other) for other in protected)
    except Exception:
        return False


def _mean_pairwise_camera_cosine_owned_inference(
    normalized_values: Tensor,
    valid: Tensor,
) -> Tensor:
    """Consume an owned normalized-value buffer with the released math order."""

    masked = normalized_values.permute(0, 1, 3, 2, 4)
    masked.mul_(valid.permute(0, 1, 3, 2)[..., None].to(normalized_values.dtype))
    camera_count = valid.sum(dim=2).to(normalized_values.dtype)
    pair_count = camera_count * (camera_count - 1.0) * 0.5
    summed = masked.sum(dim=3)
    summed.square_()
    summed_square = summed.sum(dim=-1)
    masked.square_()
    pair_sum = (summed_square - masked.sum(dim=(-1, -2))) * 0.5
    return torch.where(
        pair_count > 0,
        pair_sum / pair_count.clamp_min(1.0),
        torch.zeros_like(pair_sum),
    )


class PointwiseROSA(nn.Module):
    """Fuse camera evidence independently at each object keypoint."""

    geometry_dims = 14

    def __init__(self, config: ROSAConfig) -> None:
        super().__init__()
        self.cfg = config
        dims = config.embed_dims
        hidden = config.hidden_dims
        self.rfl = (
            SphericalRayFeatureLifting.for_compact_profile(embed_dims=dims)
            if config.architecture_profile == "compact"
            else SphericalRayFeatureLifting(
                embed_dims=dims,
                hidden_dims=max(16, hidden // 4),
            )
        )
        self.feature_norm = nn.LayerNorm(dims)
        self.query_proj = nn.Linear(dims, hidden, bias=False)
        self.key_proj = nn.Linear(
            dims + self.geometry_dims,
            hidden,
            bias=False,
        )
        self.value_proj = (
            nn.Identity()
            if config.architecture_profile == "compact"
            else nn.Linear(dims, dims, bias=False)
        )
        self.geometry_bias = nn.Sequential(
            nn.Linear(self.geometry_dims, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.point_embedding = nn.Embedding(config.num_keypoints, dims)
        if config.architecture_profile == "compact":
            self.point_query = None
            self.point_key = None
            self.point_output = None
        else:
            self.point_query = nn.Linear(dims, dims, bias=False)
            self.point_key = nn.Linear(dims, dims, bias=False)
            self.point_output = nn.Sequential(
                nn.LayerNorm(dims),
                nn.Linear(dims, dims),
                nn.SiLU(),
            )

    def reset_parameters(self) -> None:
        self.rfl.reset_parameters()
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.rfl.output_proj:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.point_embedding.weight, std=0.02)

    def forward(
        self,
        features: Tensor,
        geometry: Tensor,
        relation: Tensor | RayFeatureRelation,
        valid: Tensor,
        query: Tensor,
        *,
        observation_usage: str = "full",
    ) -> ObservationEncoding:
        return self._forward_impl(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage=observation_usage,
            reuse_owned_rfl_delta=False,
            reuse_owned_pointwise_storage=False,
            use_dual_camera_logits=False,
        )

    def _can_reuse_owned_rfl_delta(
        self,
        features: Tensor,
        geometry: Tensor,
        relation: Tensor | RayFeatureRelation,
        valid: Tensor,
        query: Tensor,
        observation_usage: str,
    ) -> bool:
        """Recognize the hook-free compact no-grad eager observation edge."""

        modules = tuple(self.modules())
        return bool(
            type(self) is PointwiseROSA
            and not self.training
            and getattr(self, "_compiled_call_impl", None) is None
            and self.cfg.architecture_profile == "compact"
            and self.cfg.enable_rfl
            and observation_usage in ("base_support", "candidate")
            and "forward" not in self.__dict__
            and type(self).forward is _RELEASED_POINTWISE_ROSA_FORWARD
            and "_forward_compact_lean_inference" not in self.__dict__
            and type(self)._forward_compact_lean_inference
            is _RELEASED_POINTWISE_ROSA_LEAN_FORWARD
            and "_forward_impl" not in self.__dict__
            and type(self)._forward_impl is _RELEASED_POINTWISE_ROSA_IMPL
            and "_can_reuse_owned_rfl_delta" not in self.__dict__
            and type(self)._can_reuse_owned_rfl_delta
            is _RELEASED_POINTWISE_ROSA_REUSE_GUARD
            and all(
                type(value) is torch.Tensor
                for value in (features, geometry, valid, query)
            )
            and type(relation) is RayFeatureRelation
            and all(not module.training for module in modules)
            and not any(
                module._forward_pre_hooks
                or module._forward_hooks
                or module._backward_pre_hooks
                or module._backward_hooks
                for module in modules
            )
            and not torch.nn.modules.module._global_forward_pre_hooks
            and not torch.nn.modules.module._global_forward_hooks
            and not torch.nn.modules.module._global_backward_pre_hooks
            and not torch.nn.modules.module._global_backward_hooks
            and self.rfl._can_reuse_owned_delta(features, relation)
        )

    def _can_reuse_owned_pointwise_storage(
        self,
        features: Tensor,
        geometry: Tensor,
        relation: Tensor | RayFeatureRelation,
        valid: Tensor,
        query: Tensor,
        observation_usage: str,
    ) -> bool:
        """Seal Pointwise buffer reuse to the released FP32 lean eager edge."""

        modules = tuple(self.modules())
        if (
            "_can_reuse_owned_pointwise_storage" in self.__dict__
            or type(self)._can_reuse_owned_pointwise_storage
            is not _RELEASED_POINTWISE_ROSA_STORAGE_GUARD
            or _is_owned_plain_fp32_tensor is not _RELEASED_OWNED_TENSOR_CHECK
            or _mean_pairwise_camera_cosine_owned_inference
            is not _RELEASED_OWNED_CONSISTENCY
            or mean_pairwise_camera_cosine is not _RELEASED_MEAN_PAIRWISE_CAMERA_COSINE
            or any(
                "_call_impl" in module.__dict__
                or type(module)._call_impl is not _RELEASED_MODULE_CALL_IMPL
                for module in modules
            )
            or _rfl._plain_eager_no_grad_execution is not _RELEASED_PLAIN_EAGER_NO_GRAD
            or not _rfl._plain_eager_no_grad_execution()
            or not self._can_reuse_owned_rfl_delta(
                features,
                geometry,
                relation,
                valid,
                query,
                observation_usage,
            )
        ):
            return False

        if (
            type(self.feature_norm) is not nn.LayerNorm
            or "forward" in self.feature_norm.__dict__
            or type(self.feature_norm).forward is not _RELEASED_LAYER_NORM_FORWARD
            or type(self.query_proj) is not nn.Linear
            or "forward" in self.query_proj.__dict__
            or type(self.query_proj).forward is not _RELEASED_LINEAR_FORWARD
            or type(self.key_proj) is not nn.Linear
            or "forward" in self.key_proj.__dict__
            or type(self.key_proj).forward is not _RELEASED_LINEAR_FORWARD
            or type(self.value_proj) is not nn.Identity
            or "forward" in self.value_proj.__dict__
            or type(self.value_proj).forward is not _RELEASED_IDENTITY_FORWARD
            or type(self.geometry_bias) is not nn.Sequential
            or "forward" in self.geometry_bias.__dict__
            or type(self.geometry_bias).forward is not _RELEASED_SEQUENTIAL_FORWARD
            or len(self.geometry_bias) != 3
            or type(self.geometry_bias[0]) is not nn.Linear
            or "forward" in self.geometry_bias[0].__dict__
            or type(self.geometry_bias[0]).forward is not _RELEASED_LINEAR_FORWARD
            or type(self.geometry_bias[1]) is not nn.SiLU
            or "forward" in self.geometry_bias[1].__dict__
            or type(self.geometry_bias[1]).forward is not _RELEASED_SILU_FORWARD
            or self.geometry_bias[1].inplace
            or type(self.geometry_bias[2]) is not nn.Linear
            or "forward" in self.geometry_bias[2].__dict__
            or type(self.geometry_bias[2]).forward is not _RELEASED_LINEAR_FORWARD
            or type(self.point_embedding) is not nn.Embedding
            or "forward" in self.point_embedding.__dict__
            or type(self.point_embedding).forward is not _RELEASED_EMBEDDING_FORWARD
            or self.point_query is not None
            or self.point_key is not None
            or self.point_output is not None
        ):
            return False

        tensors = (features, geometry, valid, query, relation.ray, relation.valid)
        if (
            features.dtype is not torch.float32
            or geometry.dtype is not torch.float32
            or query.dtype is not torch.float32
            or relation.ray.dtype is not torch.float32
            or valid.dtype is not torch.bool
            or relation.valid.dtype is not torch.bool
            or any(tensor.device != features.device for tensor in tensors)
            or any(
                tensor.layout != torch.strided
                or tensor.is_quantized
                or tensor.requires_grad
                or not tensor.is_contiguous()
                or any(name is not None for name in tensor.names)
                for tensor in tensors
            )
        ):
            return False
        alias_check = getattr(torch._C, "_is_alias_of", None)
        try:
            if (
                alias_check is not _RELEASED_ALIAS_CHECK
                or alias_check(relation.ray, geometry)
                or alias_check(relation.valid, valid)
            ):
                return False
        except Exception:
            return False

        dims = self.cfg.embed_dims
        hidden = self.cfg.hidden_dims
        parameters = tuple(self.parameters())
        return bool(
            self.cfg.architecture_profile == "compact"
            and self.cfg.enable_rfl
            and self.cfg.enable_rosa_geometry
            and not self.cfg.uniform_camera_fusion
            and not self.cfg.uniform_point_fusion
            and all(
                getattr(module, "_compiled_call_impl", None) is None
                for module in modules
            )
            and self.feature_norm.normalized_shape == (dims,)
            and self.feature_norm.elementwise_affine
            and self.query_proj.in_features == dims
            and self.query_proj.out_features == hidden
            and self.query_proj.bias is None
            and self.key_proj.in_features == dims + self.geometry_dims
            and self.key_proj.out_features == hidden
            and self.key_proj.bias is None
            and self.geometry_bias[0].in_features == self.geometry_dims
            and self.geometry_bias[0].out_features == hidden
            and self.geometry_bias[2].in_features == hidden
            and self.geometry_bias[2].out_features == 1
            and self.point_embedding.num_embeddings == self.cfg.num_keypoints
            and self.point_embedding.embedding_dim == dims
            and self.point_embedding.padding_idx is None
            and self.point_embedding.max_norm is None
            and not self.point_embedding.scale_grad_by_freq
            and not self.point_embedding.sparse
            and all(type(parameter) is nn.Parameter for parameter in parameters)
            and all(parameter.dtype is torch.float32 for parameter in parameters)
            and all(parameter.device == features.device for parameter in parameters)
            and all(parameter.layout == torch.strided for parameter in parameters)
            and all(not parameter.is_quantized for parameter in parameters)
            and all(
                all(name is None for name in parameter.names)
                for parameter in parameters
            )
        )

    def _forward_compact_lean_inference(
        self,
        features: Tensor,
        geometry: Tensor,
        relation: Tensor | RayFeatureRelation,
        valid: Tensor,
        query: Tensor,
        *,
        observation_usage: str,
    ) -> ObservationEncoding:
        """Fuse a lean observation while reusing owned RFL/Pointwise storage."""

        pointwise_storage_reuse = bool(
            "_can_reuse_owned_pointwise_storage" not in self.__dict__
            and type(self)._can_reuse_owned_pointwise_storage
            is _RELEASED_POINTWISE_ROSA_STORAGE_GUARD
            and self._can_reuse_owned_pointwise_storage(
                features,
                geometry,
                relation,
                valid,
                query,
                observation_usage,
            )
        )
        rfl_storage_reuse = bool(
            pointwise_storage_reuse
            or (
                "_can_reuse_owned_rfl_delta" not in self.__dict__
                and type(self)._can_reuse_owned_rfl_delta
                is _RELEASED_POINTWISE_ROSA_REUSE_GUARD
                and self._can_reuse_owned_rfl_delta(
                    features,
                    geometry,
                    relation,
                    valid,
                    query,
                    observation_usage,
                )
            )
        )
        if not rfl_storage_reuse:
            # Use Module.__call__ so every public hook and interception surface
            # remains observable whenever any private precondition is absent.
            return self(
                features,
                geometry,
                relation,
                valid,
                query,
                observation_usage=observation_usage,
            )
        return self._forward_impl(
            features,
            geometry,
            relation,
            valid,
            query,
            observation_usage=observation_usage,
            reuse_owned_rfl_delta=True,
            reuse_owned_pointwise_storage=pointwise_storage_reuse,
            use_dual_camera_logits=(
                "_can_use_dual_camera_logits" not in self.__dict__
                and type(self)._can_use_dual_camera_logits
                is _RELEASED_POINTWISE_ROSA_DUAL_GUARD
                and "_dual_camera_logits_inference" not in self.__dict__
                and type(self)._dual_camera_logits_inference
                is _RELEASED_POINTWISE_ROSA_DUAL_LOGITS
                and self._can_use_dual_camera_logits(
                    features,
                    geometry,
                    relation,
                    valid,
                    query,
                    observation_usage,
                )
            ),
        )

    def _can_use_dual_camera_logits(
        self,
        features: Tensor,
        geometry: Tensor,
        relation: Tensor | RayFeatureRelation,
        valid: Tensor,
        query: Tensor,
        observation_usage: str,
    ) -> bool:
        """Seal the non-bitwise dual form to released fast FP32 inference."""

        if (
            not self._can_reuse_owned_rfl_delta(
                features,
                geometry,
                relation,
                valid,
                query,
                observation_usage,
            )
            or "_can_use_dual_camera_logits" in self.__dict__
            or type(self)._can_use_dual_camera_logits
            is not _RELEASED_POINTWISE_ROSA_DUAL_GUARD
            or "_can_reuse_owned_rfl_delta" in self.__dict__
            or type(self)._can_reuse_owned_rfl_delta
            is not _RELEASED_POINTWISE_ROSA_REUSE_GUARD
            or "_dual_camera_logits_inference" in self.__dict__
            or type(self)._dual_camera_logits_inference
            is not _RELEASED_POINTWISE_ROSA_DUAL_LOGITS
            or "_call_impl" in self.__dict__
            or type(self)._call_impl is not _RELEASED_MODULE_CALL_IMPL
            or self.cfg.embed_dims != 256
            or self.cfg.hidden_dims != 64
            or self.cfg.num_cameras != 4
            or self.cfg.num_keypoints != 13
            or self.cfg.refine_topk != 50
            or self.cfg.num_refinement_steps != 1
            or not self.cfg.enable_rosa_geometry
            or self.cfg.uniform_camera_fusion
            or any(
                type(value) is not torch.Tensor
                for value in (features, geometry, valid, query)
            )
            or any(
                value.dtype is not torch.float32
                for value in (features, geometry, query)
            )
            or valid.dtype is not torch.bool
            or features.ndim != 5
            or features.shape[-1] != 256
            or geometry.shape != features.shape[:-1] + (self.geometry_dims,)
            or valid.shape != features.shape[:-1]
            or query.shape != features.shape[:2] + (256,)
            or any(
                value.layout != torch.strided
                or value.is_quantized
                or value.device.type not in ("cpu", "cuda")
                or not value.is_contiguous()
                or value.requires_grad
                or any(name is not None for name in value.names)
                for value in (features, geometry, valid, query)
            )
            or any(
                value.device != features.device for value in (geometry, valid, query)
            )
        ):
            return False

        if (
            type(self.geometry_bias) is not nn.Sequential
            or len(self.geometry_bias) != 3
        ):
            return False
        geometry_hidden = self.geometry_bias[0]
        geometry_output = self.geometry_bias[2]
        guarded_modules = (
            self.feature_norm,
            self.query_proj,
            self.key_proj,
            self.geometry_bias,
            geometry_hidden,
            self.geometry_bias[1],
            geometry_output,
        )
        if (
            type(self.feature_norm) is not nn.LayerNorm
            or type(self.feature_norm).forward is not _RELEASED_LAYER_NORM_FORWARD
            or self.feature_norm.normalized_shape != (256,)
            or not self.feature_norm.elementwise_affine
            or type(self.query_proj) is not nn.Linear
            or type(self.query_proj).forward is not _RELEASED_LINEAR_FORWARD
            or self.query_proj.bias is not None
            or type(self.key_proj) is not nn.Linear
            or type(self.key_proj).forward is not _RELEASED_LINEAR_FORWARD
            or self.key_proj.bias is not None
            or type(self.geometry_bias).forward is not _RELEASED_SEQUENTIAL_FORWARD
            or type(geometry_hidden) is not nn.Linear
            or type(geometry_hidden).forward is not _RELEASED_LINEAR_FORWARD
            or type(self.geometry_bias[1]) is not nn.SiLU
            or type(self.geometry_bias[1]).forward is not _RELEASED_SILU_FORWARD
            or type(geometry_output) is not nn.Linear
            or type(geometry_output).forward is not _RELEASED_LINEAR_FORWARD
            or any("forward" in module.__dict__ for module in guarded_modules)
            or any("_call_impl" in module.__dict__ for module in guarded_modules)
            or any(
                type(module)._call_impl is not _RELEASED_MODULE_CALL_IMPL
                for module in guarded_modules
            )
            or any(
                getattr(module, "_compiled_call_impl", None) is not None
                for module in guarded_modules
            )
        ):
            return False

        parameters = (
            self.feature_norm.weight,
            self.feature_norm.bias,
            self.query_proj.weight,
            self.key_proj.weight,
            geometry_hidden.weight,
            geometry_hidden.bias,
            geometry_output.weight,
            geometry_output.bias,
        )
        return bool(
            all(type(parameter) is torch.nn.Parameter for parameter in parameters)
            and all(parameter.layout == torch.strided for parameter in parameters)
            and all(not parameter.is_quantized for parameter in parameters)
            and all(parameter.is_contiguous() for parameter in parameters)
            and all(parameter.device == features.device for parameter in parameters)
            and all(parameter.dtype is torch.float32 for parameter in parameters)
            and all(
                all(name is None for name in parameter.names)
                for parameter in parameters
            )
            and self.feature_norm.weight.shape == (256,)
            and self.feature_norm.bias.shape == (256,)
            and self.query_proj.weight.shape == (64, 256)
            and self.key_proj.weight.shape == (64, 270)
            and geometry_hidden.weight.shape == (64, 14)
            and geometry_hidden.bias.shape == (64,)
            and geometry_output.weight.shape == (1, 64)
            and geometry_output.bias.shape == (1,)
        )

    def _dual_camera_logits_inference(
        self,
        normalized: Tensor,
        scoring_geometry: Tensor,
        query_key: Tensor,
    ) -> Tensor:
        """Evaluate ``q @ W @ x`` without one projected key per observation.

        This changes floating-point reduction order and is intentionally not a
        bit-exact rewrite.  Its caller admits only the sealed FP32 fast-eager
        edge; every public or interceptable execution keeps the released form.
        """

        dual_query = torch.matmul(query_key, self.key_proj.weight)
        appearance_query = dual_query[..., : self.cfg.embed_dims]
        geometry_query = dual_query[..., self.cfg.embed_dims :]
        appearance_logits = torch.einsum(
            "bncpd,bnd->bncp",
            normalized,
            appearance_query,
        )
        geometry_logits = torch.einsum(
            "bncpg,bng->bncp",
            scoring_geometry,
            geometry_query,
        )
        return appearance_logits + geometry_logits

    def _forward_impl(
        self,
        features: Tensor,
        geometry: Tensor,
        relation: Tensor | RayFeatureRelation,
        valid: Tensor,
        query: Tensor,
        *,
        observation_usage: str,
        reuse_owned_rfl_delta: bool,
        reuse_owned_pointwise_storage: bool,
        use_dual_camera_logits: bool,
    ) -> ObservationEncoding:
        if not isinstance(use_dual_camera_logits, bool):
            raise TypeError("use_dual_camera_logits must be a bool")
        if use_dual_camera_logits:
            released_dual_surface = bool(
                "_can_use_dual_camera_logits" not in self.__dict__
                and type(self)._can_use_dual_camera_logits
                is _RELEASED_POINTWISE_ROSA_DUAL_GUARD
                and "_dual_camera_logits_inference" not in self.__dict__
                and type(self)._dual_camera_logits_inference
                is _RELEASED_POINTWISE_ROSA_DUAL_LOGITS
            )
            if not released_dual_surface or not self._can_use_dual_camera_logits(
                features,
                geometry,
                relation,
                valid,
                query,
                observation_usage,
            ):
                use_dual_camera_logits = False
        if observation_usage not in ("full", "base_support", "candidate"):
            raise ValueError(f"invalid observation usage: {observation_usage!r}")
        if (
            features.ndim != 5
            or geometry.shape != features.shape[:-1] + (self.geometry_dims,)
            or valid.shape != features.shape[:-1]
            or query.shape != features.shape[:2] + (self.cfg.embed_dims,)
        ):
            raise ValueError("ROSA observation tensors are misaligned")

        retain_auxiliary = observation_usage == "full"
        need_consistency = observation_usage in ("full", "candidate")
        if reuse_owned_rfl_delta:
            lifted = self.rfl._forward_lifted_owned_delta_inference(
                features,
                relation,
            )
        else:
            rfl_scale = 1.0 if self.cfg.enable_rfl else 0.0
            lifted = features + rfl_scale * self.rfl(features, relation)
        if reuse_owned_pointwise_storage:
            protected = (
                features,
                geometry,
                valid,
                query,
                relation.ray,
                relation.valid,
                *tuple(self.parameters()),
            )
            reuse_owned_pointwise_storage = _is_owned_plain_fp32_tensor(
                lifted,
                shape=features.shape,
                device=features.device,
                protected=protected,
            )
        normalized = self.feature_norm(lifted)
        query_key = self.query_proj(query)[:, :, None, None, :]
        scoring_geometry = (
            geometry if self.cfg.enable_rosa_geometry else torch.zeros_like(geometry)
        )
        if use_dual_camera_logits:
            logits = self._dual_camera_logits_inference(
                normalized,
                scoring_geometry,
                query_key.squeeze(2).squeeze(2),
            )
        else:
            camera_key = self.key_proj(torch.cat((normalized, scoring_geometry), dim=-1))
            if reuse_owned_pointwise_storage and _is_owned_plain_fp32_tensor(
                camera_key,
                shape=features.shape[:-1] + (self.cfg.hidden_dims,),
                device=features.device,
                protected=(lifted, normalized, scoring_geometry, query_key),
            ):
                torch.mul(query_key, camera_key, out=camera_key)
                logits = camera_key.sum(dim=-1)
            else:
                reuse_owned_pointwise_storage = False
                logits = (query_key * camera_key).sum(dim=-1)
        logits = logits / math.sqrt(self.cfg.hidden_dims)
        logits = logits + self.geometry_bias(scoring_geometry).squeeze(-1)
        if self.cfg.uniform_camera_fusion:
            logits = logits * 0.0

        camera_weights = masked_softmax(logits, valid, dim=2)
        values = self.value_proj(lifted)
        if reuse_owned_pointwise_storage and values is not lifted:
            reuse_owned_pointwise_storage = False
        if reuse_owned_pointwise_storage and observation_usage == "base_support":
            torch.mul(camera_weights[..., None], values, out=values)
            point_features = values.sum(dim=2)
        else:
            point_features = (camera_weights[..., None] * values).sum(dim=2)
        point_valid = valid.any(dim=2)
        point_ids = torch.arange(
            self.cfg.num_keypoints,
            device=query.device,
        )
        point_embedding = self.point_embedding(point_ids)[None, None]
        if reuse_owned_pointwise_storage and _is_owned_plain_fp32_tensor(
            point_features,
            shape=features.shape[:2] + features.shape[3:5],
            device=features.device,
            protected=(values, camera_weights, point_embedding),
        ):
            point_features.add_(point_embedding)
        else:
            reuse_owned_pointwise_storage = False
            point_features = point_features + point_embedding
        if self.cfg.architecture_profile == "compact":
            if reuse_owned_pointwise_storage:
                normalized_query = F.normalize(query, dim=-1, eps=1e-6)
                normalized_points = F.normalize(point_features, dim=-1, eps=1e-6)
                if _is_owned_plain_fp32_tensor(
                    normalized_points,
                    shape=point_features.shape,
                    device=features.device,
                    protected=(query, normalized_query, point_features),
                ):
                    torch.mul(
                        normalized_query[:, :, None, :],
                        normalized_points,
                        out=normalized_points,
                    )
                    point_logits = normalized_points.sum(dim=-1)
                else:
                    reuse_owned_pointwise_storage = False
                    point_logits = (
                        normalized_query[:, :, None, :] * normalized_points
                    ).sum(dim=-1)
            else:
                point_logits = (
                    F.normalize(query, dim=-1, eps=1e-6)[:, :, None, :]
                    * F.normalize(point_features, dim=-1, eps=1e-6)
                ).sum(dim=-1)
        else:
            point_logits = (
                self.point_query(query)[:, :, None, :] * self.point_key(point_features)
            ).sum(dim=-1)
            point_logits = point_logits / math.sqrt(self.cfg.embed_dims)
        if self.cfg.uniform_point_fusion:
            point_logits = point_logits * 0.0
        point_weights = masked_softmax(point_logits, point_valid, dim=2)
        if reuse_owned_pointwise_storage:
            torch.mul(point_weights[..., None], point_features, out=point_features)
            object_feature = point_features.sum(dim=2)
        else:
            object_feature = (point_weights[..., None] * point_features).sum(dim=2)
        if self.point_output is None:
            object_feature = F.layer_norm(
                object_feature,
                (self.cfg.embed_dims,),
            )
        else:
            object_feature = self.point_output(object_feature)

        if not need_consistency:
            consistency = None
        else:
            if reuse_owned_pointwise_storage:
                values.nan_to_num_()
                normalized_values = F.normalize(values, dim=-1, eps=1e-6)
            else:
                normalized_values = F.normalize(
                    torch.nan_to_num(values.float()),
                    dim=-1,
                    eps=1e-6,
                )
            if self.cfg.architecture_profile == "compact":
                if reuse_owned_pointwise_storage and _is_owned_plain_fp32_tensor(
                    normalized_values,
                    shape=values.shape,
                    device=features.device,
                    protected=(values, valid),
                ):
                    point_consistency = _mean_pairwise_camera_cosine_owned_inference(
                        normalized_values,
                        valid,
                    )
                else:
                    point_consistency = mean_pairwise_camera_cosine(
                        normalized_values,
                        valid,
                    )
            else:
                similarity = torch.einsum(
                    "bncpd,bnfpd->bnpcf",
                    normalized_values,
                    normalized_values,
                )
                camera_valid = valid.permute(0, 1, 3, 2)
                pair_valid = camera_valid[..., :, None] & camera_valid[..., None, :]
                upper_triangle = torch.triu(
                    torch.ones(
                        self.cfg.num_cameras,
                        self.cfg.num_cameras,
                        device=valid.device,
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )
                pair_valid = pair_valid & upper_triangle
                pair_count = pair_valid.sum(dim=(-1, -2))
                point_consistency = torch.where(
                    pair_count > 0,
                    (similarity * pair_valid.to(similarity.dtype)).sum(dim=(-1, -2))
                    / pair_count.clamp_min(1).to(similarity.dtype),
                    torch.zeros_like(pair_count, dtype=similarity.dtype),
                )
            consistency = torch.where(
                point_valid.any(dim=-1),
                (point_consistency * point_valid.to(point_consistency.dtype)).sum(
                    dim=-1
                )
                / point_valid.sum(dim=-1).clamp_min(1).to(point_consistency.dtype),
                torch.zeros_like(point_consistency[..., 0]),
            )

        if retain_auxiliary:
            valid_float = valid.to(geometry.dtype)
            valid_count = valid_float.sum(dim=3).clamp_min(1.0)

            def camera_mean(value: Tensor) -> Tensor:
                return (value * valid_float).sum(dim=3) / valid_count

            optical_angle = torch.acos((-geometry[..., 2]).clamp(-1.0, 1.0))
            camera_geometry_summary = torch.stack(
                (
                    camera_mean(optical_angle),
                    camera_mean(geometry[..., 5]),
                    camera_mean(geometry[..., 6]),
                    camera_mean(geometry[..., 7]),
                    camera_mean(geometry[..., 8]),
                    camera_mean(geometry[..., 9]),
                    valid_float.mean(dim=3),
                ),
                dim=-1,
            )
            camera_weight_mean = camera_weights.sum(dim=3) / valid_count
        else:
            camera_geometry_summary = None
            camera_weight_mean = None
        return ObservationEncoding(
            object_feature=object_feature,
            point_features=point_features if retain_auxiliary else None,
            camera_weights=camera_weights if retain_auxiliary else None,
            valid=valid,
            supported=point_valid.any(dim=-1),
            cross_camera_consistency=(
                consistency.to(query.dtype) if consistency is not None else None
            ),
            camera_weight_mean=(
                camera_weight_mean.to(query.dtype)
                if camera_weight_mean is not None
                else None
            ),
            camera_geometry_summary=(
                camera_geometry_summary.to(query.dtype)
                if camera_geometry_summary is not None
                else None
            ),
        )


_RELEASED_ALIAS_CHECK = getattr(torch._C, "_is_alias_of", None)
_RELEASED_PLAIN_EAGER_NO_GRAD = _rfl._plain_eager_no_grad_execution
_RELEASED_OWNED_TENSOR_CHECK = _is_owned_plain_fp32_tensor
_RELEASED_OWNED_CONSISTENCY = _mean_pairwise_camera_cosine_owned_inference
_RELEASED_MEAN_PAIRWISE_CAMERA_COSINE = mean_pairwise_camera_cosine
_RELEASED_MODULE_CALL_IMPL = nn.Module._call_impl
_RELEASED_LINEAR_FORWARD = nn.Linear.forward
_RELEASED_LAYER_NORM_FORWARD = nn.LayerNorm.forward
_RELEASED_IDENTITY_FORWARD = nn.Identity.forward
_RELEASED_SEQUENTIAL_FORWARD = nn.Sequential.forward
_RELEASED_SILU_FORWARD = nn.SiLU.forward
_RELEASED_EMBEDDING_FORWARD = nn.Embedding.forward
_RELEASED_POINTWISE_ROSA_FORWARD = PointwiseROSA.forward
_RELEASED_POINTWISE_ROSA_IMPL = PointwiseROSA._forward_impl
_RELEASED_POINTWISE_ROSA_REUSE_GUARD = PointwiseROSA._can_reuse_owned_rfl_delta
_RELEASED_POINTWISE_ROSA_STORAGE_GUARD = (
    PointwiseROSA._can_reuse_owned_pointwise_storage
)
_RELEASED_POINTWISE_ROSA_LEAN_FORWARD = PointwiseROSA._forward_compact_lean_inference
_RELEASED_POINTWISE_ROSA_DUAL_GUARD = PointwiseROSA._can_use_dual_camera_logits
_RELEASED_POINTWISE_ROSA_DUAL_LOGITS = PointwiseROSA._dual_camera_logits_inference
_RELEASED_MODULE_CALL_IMPL = nn.Module._call_impl
_RELEASED_LINEAR_FORWARD = nn.Linear.forward
_RELEASED_LAYER_NORM_FORWARD = nn.LayerNorm.forward
_RELEASED_SEQUENTIAL_FORWARD = nn.Sequential.forward
_RELEASED_SILU_FORWARD = nn.SiLU.forward


__all__ = [
    "PointwiseROSA",
    "masked_softmax",
    "mean_pairwise_camera_cosine",
]
