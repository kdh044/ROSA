"""Opt-in, output-exact runtime transforms for the Main126 C32 path."""

from __future__ import annotations

import numpy as np
import torch
from mmdet3d.registry import TRANSFORMS
from projects.mmdet3d_plugin.datasets.transforms.pack_sparse4d import (
    PackSparse4DInputs,
)


def _stack_released_uint8_images(imgs) -> np.ndarray:
    if isinstance(imgs, np.ndarray):
        array = imgs
    else:
        array = np.stack(imgs, axis=0)
    if array.dtype != np.uint8:
        raise TypeError(
            "compact fisheye packing requires released uint8 images, "
            f"got {array.dtype}"
        )
    if array.ndim != 4 or (
        array.shape[-1] not in (1, 3) and array.shape[1] not in (1, 3)
    ):
        raise ValueError(
            "fisheye images must map from (camera, H, W, channel) to "
            "(camera, channel, H, W), "
            f"got {array.shape}"
        )
    return array


def _pack_released_hwc_rgb_images(imgs) -> np.ndarray:
    if isinstance(imgs, np.ndarray):
        array = _stack_released_uint8_images(imgs)
        if array.shape[-1] != 3:
            raise ValueError(
                "RGB fisheye packing requires released HWC images with "
                "exactly three channels"
            )
        return np.ascontiguousarray(np.transpose(array[..., ::-1], (0, 3, 1, 2)))
    if not isinstance(imgs, (list, tuple)) or not imgs:
        raise TypeError(
            "RGB fisheye packing requires a nonempty released image sequence"
        )
    images = tuple(imgs)
    first = images[0]
    if not isinstance(first, np.ndarray) or first.dtype != np.uint8:
        dtype = getattr(first, "dtype", type(first).__name__)
        raise TypeError(
            f"compact fisheye packing requires released uint8 images, got {dtype}"
        )
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(
            "RGB fisheye packing requires released HWC images with "
            "exactly three channels"
        )
    image_shape = first.shape
    for image in images[1:]:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            dtype = getattr(image, "dtype", type(image).__name__)
            raise TypeError(
                "compact fisheye packing requires released uint8 images, "
                f"got {dtype}"
            )
        if image.shape != image_shape:
            raise ValueError("released fisheye camera images must share one HWC shape")
    height, width, _ = image_shape
    packed = np.empty((len(images), 3, height, width), dtype=np.uint8)
    rgb_chw_views = [np.transpose(image[..., ::-1], (2, 0, 1)) for image in images]
    np.stack(
        rgb_chw_views,
        axis=0,
        out=packed,
        casting="no",
    )
    return packed


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
        array = _stack_released_uint8_images(imgs)
        if array.shape[-1] in (1, 3):
            array = np.transpose(array, (0, 3, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(array))


@TRANSFORMS.register_module()
class PackFisheye3DODRGBUint8Inputs(PackSparse4DInputs):
    """Fuse released HWC BGR ordering into the required HWC-to-CHW copy."""

    @staticmethod
    def _pack_img(imgs) -> torch.Tensor:
        return torch.from_numpy(_pack_released_hwc_rgb_images(imgs))


__all__ = [
    "PackFisheye3DODRGBUint8Inputs",
    "PackFisheye3DODUint8Inputs",
]
