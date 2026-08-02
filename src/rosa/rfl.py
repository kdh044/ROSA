"""Spherical ray-feature lifting for sparse object-support observations.

The module is deliberately smaller than a spherical attention block.  Each
already-sampled detector observation is processed independently.  Released
legacy models retain the dense ray-appearance outer product.  Compact models
use a signed rank-16 gate over the same fixed real spherical-harmonic basis.

No keypoint interaction, camera selection, additional image sampling, or KB
Jacobian descriptor is used here.  Camera fusion remains the responsibility
of ROSA.  The output projection is zero-initialized, making the enabled module
an exact step-zero identity when its residual is added to the base DAF path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import _python_dispatch

from .types import RayFeatureRelation


def _plain_eager_no_grad_execution() -> bool:
    """Return whether private storage reuse cannot hide tensor interception."""

    if torch.is_grad_enabled():
        return False
    torch_function_mode_state = getattr(
        torch.overrides,
        "_is_torch_function_mode_enabled",
        None,
    )
    if not callable(torch_function_mode_state):
        return False
    try:
        torch_function_mode_active = torch_function_mode_state()
    except Exception:
        return False
    if type(torch_function_mode_active) is not bool or torch_function_mode_active:
        return False
    dispatch_mode_state = getattr(
        _python_dispatch,
        "_is_in_non_infra_torch_dispatch_mode",
        None,
    )
    if type(dispatch_mode_state) is bool:
        dispatch_mode_active = dispatch_mode_state
    elif callable(dispatch_mode_state):
        try:
            dispatch_mode_active = dispatch_mode_state()
        except Exception:
            return False
        if type(dispatch_mode_active) is not bool:
            return False
    else:
        return False
    if dispatch_mode_active:
        return False
    if (
        torch.jit.is_scripting()
        or torch.jit.is_tracing()
        or torch._dynamo.is_compiling()
    ):
        return False
    try:
        cuda_autocast_enabled = torch.is_autocast_enabled()
    except Exception:
        return False
    if type(cuda_autocast_enabled) is not bool or cuda_autocast_enabled:
        return False
    try:
        cpu_autocast_enabled = torch.is_autocast_enabled("cpu")
    except TypeError:  # PyTorch 2.1 compatibility.
        cpu_autocast_state = getattr(torch, "is_autocast_cpu_enabled", None)
        if not callable(cpu_autocast_state):
            return False
        try:
            cpu_autocast_enabled = cpu_autocast_state()
        except Exception:
            return False
    except Exception:
        return False
    return type(cpu_autocast_enabled) is bool and not cpu_autocast_enabled


def _mask_cast_and_add_owned_delta(
    features: torch.Tensor,
    delta: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Finish an owned RFL residual while retaining that allocation.

    ``delta`` is the private, unaliased output of the released output
    projection.  The operation order is deliberately the same as the public
    path: mask and sanitize in FP32, cast to the input dtype, then add with
    ``features`` as the left operand.  The last operation writes into the now
    dead residual allocation and therefore does not allocate another complete
    ``[B,N,C,P,D]`` tensor.
    """

    delta.masked_fill_(~valid[..., None], 0.0)
    delta.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    delta = delta.to(dtype=features.dtype)
    return torch.add(features, delta, out=delta)


class SphericalRayFeatureLifting(nn.Module):
    """Lift sparse appearance features into a low-order spherical ray basis.

    Args:
        embed_dims: Width of each sampled detector feature.
        hidden_dims: Appearance width used before the tensor product.

    Inputs:
        keypoint_features: Weighted sampled features ``[B,N,C,P,D]``.
        relation: Compact calibrated rays/validity.  Historical
            ``[B,C,N,P,19]`` relation tensors remain accepted for checkpoint
            and API compatibility.

    Returns:
        Per-keypoint residuals ``[B,N,C,P,D]``.  Invalid observations are
        exactly zero and the complete output is exactly zero at initialization.
    """

    relation_dims = 19
    num_sh_basis = 9
    compact_factor_dims = 16

    def __init__(
        self,
        embed_dims: int = 256,
        hidden_dims: int = 16,
        *,
        _factorized: bool = False,
    ) -> None:
        super().__init__()
        if int(embed_dims) <= 0 or int(hidden_dims) <= 0:
            raise ValueError("RFL dimensions must be positive")
        if _factorized and int(hidden_dims) != self.compact_factor_dims:
            raise ValueError(
                "compact factorized RFL has a fixed rank of "
                f"{self.compact_factor_dims}"
            )

        self.embed_dims = int(embed_dims)
        self.hidden_dims = int(hidden_dims)
        self.factorized = bool(_factorized)
        # Parameter-free normalization avoids a geometry-only learned offset.
        self.feature_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)
        # Bias-free projections ensure the lifted branch is a genuine
        # appearance-by-ray interaction rather than an independent ray PE.
        self.appearance_proj = nn.Linear(self.embed_dims, self.hidden_dims, bias=False)
        self.ray_proj = (
            nn.Linear(self.num_sh_basis, self.hidden_dims, bias=False)
            if self.factorized
            else None
        )
        self.output_proj = nn.Linear(
            (
                self.hidden_dims
                if self.factorized
                else self.num_sh_basis * self.hidden_dims
            ),
            self.embed_dims,
            bias=False,
        )

        self.reset_parameters()

    @classmethod
    def for_compact_profile(
        cls,
        embed_dims: int = 256,
    ) -> "SphericalRayFeatureLifting":
        """Build the fixed signed-gated CP16 topology used by compact ROSA."""

        return cls(
            embed_dims=embed_dims,
            hidden_dims=cls.compact_factor_dims,
            _factorized=True,
        )

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.appearance_proj.weight)
        if self.ray_proj is not None:
            nn.init.xavier_uniform_(self.ray_proj.weight)
        # One zero map provides both exact identity and a non-zero first-step
        # gradient.  Zeroing an inner factor would deadlock that projection.
        nn.init.zeros_(self.output_proj.weight)

    @staticmethod
    def real_sh_degree_2(unit_ray: torch.Tensor) -> torch.Tensor:
        """Return a fixed orthonormal real SH basis through degree two.

        The sign/order convention is immaterial for the learned linear output,
        but is fixed here for reproducibility.  ``unit_ray`` may have arbitrary
        leading dimensions and must end in three Cartesian coordinates.
        """
        if unit_ray.ndim < 1 or unit_ray.shape[-1] != 3:
            raise ValueError(
                "RFL unit rays must end in xyz, got " f"{tuple(unit_ray.shape)}"
            )
        ray = F.normalize(
            torch.nan_to_num(unit_ray.float(), nan=0.0, posinf=0.0, neginf=0.0),
            dim=-1,
            eps=1e-6,
        )
        x, y, z = ray.unbind(dim=-1)

        c0 = 1.0 / (2.0 * math.sqrt(math.pi))
        c1 = math.sqrt(3.0 / (4.0 * math.pi))
        c2_cross = math.sqrt(15.0 / (4.0 * math.pi))
        c2_z = math.sqrt(5.0 / (16.0 * math.pi))
        c2_xy = math.sqrt(15.0 / (16.0 * math.pi))
        return torch.stack(
            (
                torch.full_like(x, c0),
                c1 * x,
                c1 * y,
                c1 * z,
                c2_cross * x * y,
                c2_cross * y * z,
                c2_cross * z * x,
                c2_z * (3.0 * z.square() - 1.0),
                c2_xy * (x.square() - y.square()),
            ),
            dim=-1,
        )

    @classmethod
    def basis_and_valid(
        cls, relation_bncp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build SH basis and hard validity from ``[B,N,C,P,19]`` relation."""
        if relation_bncp.ndim != 5 or relation_bncp.shape[-1] != cls.relation_dims:
            raise ValueError(
                "RFL relation must be [B,N,C,P,19], got "
                f"{tuple(relation_bncp.shape)}"
            )
        relation_bncp = torch.nan_to_num(
            relation_bncp.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        valid = relation_bncp[..., 13] > 0.0
        basis = cls.real_sh_degree_2(relation_bncp[..., 8:11])
        basis = torch.where(valid[..., None], basis, torch.zeros_like(basis))
        return basis, valid

    @classmethod
    def compact_basis_and_valid(
        cls,
        relation: RayFeatureRelation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the legacy-exact basis without a sparse 19-channel carrier."""

        if (
            type(relation) is not RayFeatureRelation
            or relation.ray.ndim != 5
            or relation.ray.shape[-1] != 3
            or relation.valid.shape != relation.ray.shape[:-1]
        ):
            raise ValueError(
                "compact RFL relation must contain [B,N,C,P,3] rays and "
                "aligned validity"
            )
        ray = torch.nan_to_num(
            relation.ray.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid = torch.nan_to_num(
            relation.valid.to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).gt(0.0)
        basis = cls.real_sh_degree_2(ray)
        basis = torch.where(valid[..., None], basis, torch.zeros_like(basis))
        return basis, valid

    def forward(
        self,
        keypoint_features: torch.Tensor,
        relation: torch.Tensor | RayFeatureRelation,
    ) -> torch.Tensor:
        if keypoint_features.ndim != 5:
            raise ValueError(
                "RFL keypoint_features must be [B,N,C,P,D], got "
                f"{tuple(keypoint_features.shape)}"
            )
        bs, num_anchor, num_cams, num_pts, embed_dims = keypoint_features.shape
        if embed_dims != self.embed_dims:
            raise ValueError(
                f"RFL feature width must be {self.embed_dims}, got {embed_dims}"
            )
        original_dtype = keypoint_features.dtype
        features = torch.nan_to_num(
            keypoint_features.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        if type(relation) is RayFeatureRelation:
            expected_ray = (
                bs,
                num_anchor,
                num_cams,
                num_pts,
                3,
            )
            if tuple(relation.ray.shape) != expected_ray:
                raise ValueError(
                    "compact RFL relation shape mismatch: "
                    f"{tuple(relation.ray.shape)} vs {expected_ray}"
                )
            basis, valid = self.compact_basis_and_valid(relation)
        else:
            expected_relation = (
                bs,
                num_cams,
                num_anchor,
                num_pts,
                self.relation_dims,
            )
            if not torch.is_tensor(relation) or tuple(relation.shape) != (
                expected_relation
            ):
                observed = getattr(relation, "shape", None)
                raise ValueError(
                    "RFL relation shape mismatch: " f"{observed} vs {expected_relation}"
                )
            relation_bncp = relation.permute(0, 2, 1, 3, 4).contiguous()
            basis, valid = self.basis_and_valid(relation_bncp)

        appearance = self.appearance_proj(self.feature_norm(features))
        if self.ray_proj is None:
            lifted = basis[..., :, None] * appearance[..., None, :]
            lifted = lifted.flatten(start_dim=-2)
        else:
            lifted = appearance * self.ray_proj(basis)
        delta = self.output_proj(lifted)
        delta = torch.where(valid[..., None], delta, torch.zeros_like(delta))
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        return delta.to(dtype=original_dtype)

    def _can_reuse_owned_delta(
        self,
        keypoint_features: torch.Tensor,
        relation: torch.Tensor | RayFeatureRelation,
    ) -> bool:
        """Seal the allocation-reuse path to released compact eager inference."""

        if (
            type(self) is not SphericalRayFeatureLifting
            or self.training
            or getattr(self, "_compiled_call_impl", None) is not None
            or "forward" in self.__dict__
            or type(self).forward is not _RELEASED_RFL_FORWARD
            or "_forward_lifted_owned_delta_inference" in self.__dict__
            or type(self)._forward_lifted_owned_delta_inference
            is not _RELEASED_RFL_OWNED_DELTA_FORWARD
            or _mask_cast_and_add_owned_delta is not _RELEASED_MASK_CAST_ADD
            or not _plain_eager_no_grad_execution()
            or type(keypoint_features) is not torch.Tensor
            or type(relation) is not RayFeatureRelation
            or type(relation.ray) is not torch.Tensor
            or type(relation.valid) is not torch.Tensor
            or keypoint_features.ndim != 5
            or keypoint_features.shape[-1] != self.embed_dims
            or relation.ray.shape != keypoint_features.shape[:-1] + (3,)
            or relation.valid.shape != keypoint_features.shape[:-1]
            or keypoint_features.dtype
            not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or relation.ray.dtype
            not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or relation.valid.dtype is not torch.bool
            or not keypoint_features.is_contiguous()
            or keypoint_features.requires_grad
            or relation.ray.requires_grad
            or relation.valid.requires_grad
        ):
            return False

        tensors = (keypoint_features, relation.ray, relation.valid)
        if any(
            tensor.layout != torch.strided
            or tensor.is_quantized
            or tensor.device.type not in ("cpu", "cuda")
            or any(name is not None for name in tensor.names)
            for tensor in tensors
        ) or any(tensor.device != keypoint_features.device for tensor in tensors):
            return False

        ray_projection = self.ray_proj
        modules = (
            self,
            self.feature_norm,
            self.appearance_proj,
            self.output_proj,
            *((ray_projection,) if ray_projection is not None else ()),
        )
        if (
            type(self.factorized) is not bool
            or type(self.feature_norm) is not nn.LayerNorm
            or "forward" in self.feature_norm.__dict__
            or type(self.feature_norm).forward is not _RELEASED_LAYER_NORM_FORWARD
            or self.feature_norm.elementwise_affine
            or type(self.appearance_proj) is not nn.Linear
            or "forward" in self.appearance_proj.__dict__
            or type(self.appearance_proj).forward is not _RELEASED_LINEAR_FORWARD
            or self.appearance_proj.bias is not None
            or type(self.output_proj) is not nn.Linear
            or "forward" in self.output_proj.__dict__
            or type(self.output_proj).forward is not _RELEASED_LINEAR_FORWARD
            or self.output_proj.bias is not None
            or (
                self.factorized
                and (
                    type(ray_projection) is not nn.Linear
                    or "forward" in ray_projection.__dict__
                    or type(ray_projection).forward is not _RELEASED_LINEAR_FORWARD
                    or ray_projection.bias is not None
                )
            )
            or (not self.factorized and ray_projection is not None)
            or any(module.training for module in modules)
            or any(
                module._forward_pre_hooks
                or module._forward_hooks
                or module._backward_pre_hooks
                or module._backward_hooks
                for module in modules
            )
            or torch.nn.modules.module._global_forward_pre_hooks
            or torch.nn.modules.module._global_forward_hooks
            or torch.nn.modules.module._global_backward_pre_hooks
            or torch.nn.modules.module._global_backward_hooks
        ):
            return False

        weights = (
            self.appearance_proj.weight,
            self.output_proj.weight,
            *((ray_projection.weight,) if ray_projection is not None else ()),
        )
        return bool(
            all(type(weight) is torch.nn.Parameter for weight in weights)
            and all(weight.layout == torch.strided for weight in weights)
            and all(not weight.is_quantized for weight in weights)
            and all(all(name is None for name in weight.names) for weight in weights)
            and all(weight.device == keypoint_features.device for weight in weights)
            and all(weight.dtype == torch.float32 for weight in weights)
            and self.appearance_proj.weight.shape == (self.hidden_dims, self.embed_dims)
            and (
                (
                    self.factorized
                    and ray_projection.weight.shape
                    == (self.hidden_dims, self.num_sh_basis)
                    and self.output_proj.weight.shape
                    == (self.embed_dims, self.hidden_dims)
                )
                or (
                    not self.factorized
                    and self.output_proj.weight.shape
                    == (self.embed_dims, self.num_sh_basis * self.hidden_dims)
                )
            )
        )

    def _forward_lifted_owned_delta_inference(
        self,
        keypoint_features: torch.Tensor,
        relation: torch.Tensor | RayFeatureRelation,
    ) -> torch.Tensor:
        """Return ``features + RFL(features)`` on the sealed lean edge.

        This private entry point intentionally duplicates the short released
        RFL body.  Keeping :meth:`forward` untouched makes training, autograd,
        hooks, compilation, tracing, tensor subclasses, and public callers
        fail closed to their historical allocation and interception surface.
        """

        if (
            "_can_reuse_owned_delta" in self.__dict__
            or type(self)._can_reuse_owned_delta is not _RELEASED_RFL_REUSE_GUARD
            or not self._can_reuse_owned_delta(keypoint_features, relation)
        ):
            return keypoint_features + 1.0 * self(keypoint_features, relation)

        features = torch.nan_to_num(
            keypoint_features.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        basis, valid = self.compact_basis_and_valid(relation)
        appearance = self.appearance_proj(self.feature_norm(features))
        if self.ray_proj is None:
            lifted = basis[..., :, None] * appearance[..., None, :]
            lifted = lifted.flatten(start_dim=-2)
        else:
            lifted = appearance * self.ray_proj(basis)
        delta = self.output_proj(lifted)

        # Linear returns fresh row-major storage on either released topology.
        # Recheck that runtime fact before mutating; an unfamiliar backend
        # falls back instead of risking an alias or layout-visible change.
        if (
            type(delta) is not torch.Tensor
            or delta._base is not None
            or delta.layout != torch.strided
            or delta.is_quantized
            or not delta.is_contiguous()
            or delta.dtype != torch.float32
            or delta.device != keypoint_features.device
            or delta.shape != keypoint_features.shape
            or any(name is not None for name in delta.names)
        ):
            return keypoint_features + 1.0 * self(keypoint_features, relation)
        return _mask_cast_and_add_owned_delta(keypoint_features, delta, valid)


_RELEASED_LINEAR_FORWARD = nn.Linear.forward
_RELEASED_LAYER_NORM_FORWARD = nn.LayerNorm.forward
_RELEASED_MASK_CAST_ADD = _mask_cast_and_add_owned_delta
_RELEASED_RFL_FORWARD = SphericalRayFeatureLifting.forward
_RELEASED_RFL_REUSE_GUARD = SphericalRayFeatureLifting._can_reuse_owned_delta
_RELEASED_RFL_OWNED_DELTA_FORWARD = (
    SphericalRayFeatureLifting._forward_lifted_owned_delta_inference
)
