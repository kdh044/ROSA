"""Deterministic multiview feature sampling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DeterministicBilinearSampler(nn.Module):
    """Sample ``[B,C,D,H,W]`` feature maps at normalized fisheye pixels.

    This explicit four-gather implementation is equivalent to
    ``grid_sample(..., align_corners=False)`` while retaining deterministic
    CUDA backward behavior.
    """

    def forward(self, feature: Tensor, uv: Tensor) -> Tensor:
        if feature.ndim != 5:
            raise ValueError("feature must be [B,C,D,H,W]")
        batch, cameras, channels, height, width = feature.shape
        if (
            uv.ndim != 6
            or uv.shape[0] != batch
            or uv.shape[2] != cameras
            or uv.shape[-1] != 2
        ):
            raise ValueError("uv must be [B,N,C,S,P,2]")

        queries, samples, patch = uv.shape[1], uv.shape[3], uv.shape[4]
        camera_batches = batch * cameras
        sample_count = queries * samples * patch
        coordinates = uv.permute(0, 2, 1, 3, 4, 5).reshape(
            camera_batches, sample_count, 2
        )
        coordinates = torch.nan_to_num(coordinates, nan=-2.0, posinf=-2.0, neginf=-2.0)
        x = coordinates[..., 0] * width - 0.5
        y = coordinates[..., 1] * height - 0.5
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = x0 + 1, y0 + 1
        wx1 = (x - x0.to(x.dtype)).to(feature.dtype)
        wy1 = (y - y0.to(y.dtype)).to(feature.dtype)
        wx0, wy0 = 1.0 - wx1, 1.0 - wy1
        flattened = feature.reshape(camera_batches, channels, height * width)

        def gather(ix: Tensor, iy: Tensor) -> Tensor:
            valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            linear = iy.clamp(0, height - 1) * width + ix.clamp(0, width - 1)
            values = torch.gather(
                flattened,
                2,
                linear[:, None].expand(-1, channels, -1),
            )
            return values * valid[:, None].to(values.dtype)

        sampled = (
            gather(x0, y0) * (wx0 * wy0)[:, None]
            + gather(x1, y0) * (wx1 * wy0)[:, None]
            + gather(x0, y1) * (wx0 * wy1)[:, None]
            + gather(x1, y1) * (wx1 * wy1)[:, None]
        )
        sampled = sampled.reshape(batch, cameras, channels, queries, samples, patch)
        return sampled.permute(0, 3, 1, 4, 5, 2).contiguous()


class FusedBilinearSampler(nn.Module):
    """Sample all points with one optimized grid-sampling call per FPN level."""

    def __init__(self) -> None:
        super().__init__()
        self.fallback = DeterministicBilinearSampler()

    @staticmethod
    def _prepare_grid(
        uv: Tensor,
        batch: int,
        cameras: int,
    ) -> tuple[Tensor, tuple[int, int, int]]:
        queries, samples, patch = uv.shape[1], uv.shape[3], uv.shape[4]
        grid = (
            torch.nan_to_num(
                uv.permute(0, 2, 1, 3, 4, 5),
                nan=-2.0,
                posinf=-2.0,
                neginf=-2.0,
            )
            .reshape(batch * cameras, queries * samples * patch, 1, 2)
            .mul(2.0)
            .sub(1.0)
        )
        return grid, (queries, samples, patch)

    def forward(self, feature: Tensor, uv: Tensor) -> Tensor:
        if feature.ndim != 5:
            raise ValueError("feature must be [B,C,D,H,W]")
        batch, cameras, channels, height, width = feature.shape
        if (
            uv.ndim != 6
            or uv.shape[0] != batch
            or uv.shape[2] != cameras
            or uv.shape[-1] != 2
        ):
            raise ValueError("uv must be [B,N,C,S,P,2]")
        if feature.dtype != uv.dtype or feature.requires_grad or uv.requires_grad:
            return self.fallback(feature, uv)

        grid, (queries, samples, patch) = self._prepare_grid(
            uv,
            batch,
            cameras,
        )
        sampled = F.grid_sample(
            feature.reshape(batch * cameras, channels, height, width),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch,
            cameras,
            channels,
            queries,
            samples,
            patch,
        )
        return sampled.permute(0, 3, 1, 4, 5, 2).contiguous()

    def _sample_multiscale_prepared(
        self,
        features: Sequence[Tensor],
        uv: Tensor,
    ) -> tuple[Tensor, ...]:
        """Reuse one grid for caller-validated homogeneous FPN inputs."""

        batch, cameras = features[0].shape[:2]
        grid, (queries, samples, patch) = self._prepare_grid(
            uv,
            batch,
            cameras,
        )
        prepared = []
        for feature in features:
            channels, height, width = feature.shape[-3:]
            sampled = F.grid_sample(
                feature.reshape(batch * cameras, channels, height, width),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            prepared.append(
                sampled.reshape(
                    batch,
                    cameras,
                    channels,
                    queries,
                    samples,
                    patch,
                ).permute(0, 3, 1, 4, 5, 2)
            )
        return tuple(prepared)

    def _sample_multiscale_prepared_stacked(
        self,
        features: Sequence[Tensor],
        uv: Tensor,
    ) -> Tensor:
        """Copy one sampled level at a time into the final stacked buffer."""

        batch, cameras, channels = features[0].shape[:3]
        grid, (queries, samples, patch) = self._prepare_grid(
            uv,
            batch,
            cameras,
        )
        if patch != 1:
            raise ValueError("stacked multiscale sampling requires one patch point")
        levels = None
        for level_index, feature in enumerate(features):
            height, width = feature.shape[-2:]
            sampled = F.grid_sample(
                feature.reshape(batch * cameras, channels, height, width),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = (
                sampled.reshape(
                    batch,
                    cameras,
                    channels,
                    queries,
                    samples,
                    patch,
                )
                .permute(0, 3, 1, 4, 5, 2)
                .squeeze(4)
            )
            if levels is None:
                # Autocast may promote ``grid_sample`` independently of the
                # feature dtype.  Match ``torch.stack`` by allocating from the
                # actual sampled result instead of silently casting it back.
                levels = sampled.new_empty(
                    batch,
                    queries,
                    len(features),
                    cameras,
                    samples,
                    channels,
                )
            levels[:, :, level_index].copy_(sampled)
            del sampled
        assert levels is not None
        return levels


__all__ = ["DeterministicBilinearSampler", "FusedBilinearSampler"]
