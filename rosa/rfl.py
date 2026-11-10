"""Ray feature lifting (RFL).

The calibrated viewing ray of every observation is expanded in a real
spherical-harmonic basis (degree 2, 9 terms) and gates a low-rank projection
of the sampled appearance feature.  The output projection is zero-initialized,
so the residual is zero at the start of training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import RayFeatureRelation


class SphericalRayFeatureLifting(nn.Module):
    """Residual ``W_out[(W_a LN(f)) * (W_r SH(ray))]`` per observation.

    Inputs:
        keypoint_features: sampled features ``[B,N,C,P,D]``.
        relation: unit rays ``[B,N,C,P,3]`` and validity ``[B,N,C,P]``.

    Returns:
        Residuals ``[B,N,C,P,D]``, zero for invalid observations.
    """

    num_sh_basis = 9

    def __init__(self, embed_dims: int = 256, hidden_dims: int = 16) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.hidden_dims = int(hidden_dims)
        self.feature_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)
        self.appearance_proj = nn.Linear(self.embed_dims, self.hidden_dims, bias=False)
        self.ray_proj = nn.Linear(self.num_sh_basis, self.hidden_dims, bias=False)
        self.output_proj = nn.Linear(self.hidden_dims, self.embed_dims, bias=False)
        self.reset_parameters()

    @classmethod
    def for_compact_profile(cls, embed_dims: int = 256) -> "SphericalRayFeatureLifting":
        return cls(embed_dims=embed_dims, hidden_dims=16)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.appearance_proj.weight)
        nn.init.xavier_uniform_(self.ray_proj.weight)
        # Zero only the output map so the inner factors still receive gradients.
        nn.init.zeros_(self.output_proj.weight)

    @staticmethod
    def real_sh_degree_2(unit_ray: torch.Tensor) -> torch.Tensor:
        """Orthonormal real SH basis through degree two, ``[..., 3] -> [..., 9]``."""

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

    def forward(
        self,
        keypoint_features: torch.Tensor,
        relation: RayFeatureRelation,
    ) -> torch.Tensor:
        original_dtype = keypoint_features.dtype
        features = torch.nan_to_num(
            keypoint_features.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        ray = torch.nan_to_num(relation.ray.float(), nan=0.0, posinf=0.0, neginf=0.0)
        valid = relation.valid.to(dtype=torch.float32).gt(0.0)
        basis = self.real_sh_degree_2(ray)
        basis = torch.where(valid[..., None], basis, torch.zeros_like(basis))

        appearance = self.appearance_proj(self.feature_norm(features))
        delta = self.output_proj(appearance * self.ray_proj(basis))
        delta = torch.where(valid[..., None], delta, torch.zeros_like(delta))
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        return delta.to(dtype=original_dtype)
