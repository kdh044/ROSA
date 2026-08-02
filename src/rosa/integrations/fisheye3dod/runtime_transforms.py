"""Opt-in, output-exact runtime transforms for the Main126 C32 path."""

from __future__ import annotations

import numpy as np
import torch
from mmdet3d.registry import TRANSFORMS
from projects.mmdet3d_plugin.datasets.transforms.pack_sparse4d import (
    PackSparse4DInputs,
)


@TRANSFORMS.register_module()
class PackFisheye3DODUint8Inputs(PackSparse4DInputs):
    """Keep released 8-bit images compact until device preprocessing.

    The ordinary Sparse4D packer eagerly converts the image tensor to FP32 on
    the CPU.  The exact C32 preprocessor already calls ``img.float()`` after
    device transfer, so retaining uint8 here changes only where that exact
    integer-to-FP32 conversion occurs.  All non-image fields use the unchanged
    parent implementation.
    """

    @staticmethod
    def _pack_img(imgs) -> torch.Tensor:
        if isinstance(imgs, np.ndarray):
            array = imgs
        else:
            array = np.stack(imgs, axis=0)
        if array.dtype != np.uint8:
            raise TypeError(
                "PackFisheye3DODUint8Inputs requires released uint8 images, "
                f"got {array.dtype}"
            )
        if array.ndim == 4 and array.shape[-1] in (1, 3):
            array = np.transpose(array, (0, 3, 1, 2))
        if array.ndim != 4 or array.shape[1] not in (1, 3):
            raise ValueError(
                "packed fisheye images must have shape (camera, channel, H, W), "
                f"got {array.shape}"
            )
        return torch.from_numpy(np.ascontiguousarray(array))


__all__ = ["PackFisheye3DODUint8Inputs"]
