"""Spherical ray-feature lifting for sparse object-support observations.

The module is deliberately smaller than a spherical attention block.  Each
already-sampled detector observation is processed independently: a fixed real
spherical-harmonic basis of its calibrated camera ray is tensor-multiplied
with a low-dimensional appearance feature and projected back as a residual.

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

from .types import RayFeatureRelation


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

    def __init__(self, embed_dims: int = 256, hidden_dims: int = 16) -> None:
        super().__init__()
        if int(embed_dims) <= 0 or int(hidden_dims) <= 0:
            raise ValueError("RFL dimensions must be positive")

        self.embed_dims = int(embed_dims)
        self.hidden_dims = int(hidden_dims)
        # Parameter-free normalization avoids a geometry-only learned offset.
        self.feature_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)
        # Bias-free projections ensure the lifted branch is a genuine
        # appearance-by-ray interaction rather than an independent ray PE.
        self.appearance_proj = nn.Linear(self.embed_dims, self.hidden_dims, bias=False)
        self.output_proj = nn.Linear(
            self.num_sh_basis * self.hidden_dims,
            self.embed_dims,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.appearance_proj.weight)
        # One zero map provides both exact identity and a non-zero first-step
        # gradient.  A second zero gate would deadlock the inner projection.
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
                    "RFL relation shape mismatch: "
                    f"{observed} vs {expected_relation}"
                )
            relation_bncp = relation.permute(0, 2, 1, 3, 4).contiguous()
            basis, valid = self.basis_and_valid(relation_bncp)

        appearance = self.appearance_proj(self.feature_norm(features))
        lifted = basis[..., :, None] * appearance[..., None, :]
        lifted = lifted.flatten(start_dim=-2)
        delta = self.output_proj(lifted)
        delta = torch.where(valid[..., None], delta, torch.zeros_like(delta))
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        return delta.to(dtype=original_dtype)
