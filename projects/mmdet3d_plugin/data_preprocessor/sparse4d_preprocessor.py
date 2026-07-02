"""Sparse4D data preprocessor — single-frame variant of StreamDataPreprocessor.

Sparse4D handles temporal aggregation in the model (InstanceBank) so the
preprocessor sees one frame per sample: ``inputs['img']`` is (N_cam, C, H, W)
before batching or (B, N_cam, C, H, W) after the collator stacks.
"""
from typing import List, Optional, Sequence, Union

import torch
from mmengine.model import BaseDataPreprocessor

from mmdet3d.registry import MODELS


@MODELS.register_module()
class Sparse4DDataPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        bgr_to_rgb: bool = False,
        pad_size_divisor: int = 1,
        pad_value: float = 0.0,
        non_blocking: bool = False,
    ):
        super().__init__(non_blocking=non_blocking)
        self._channel_conversion = bgr_to_rgb
        self.pad_size_divisor = int(pad_size_divisor)
        self.pad_value = float(pad_value)
        if mean is not None and std is not None:
            self._enable_normalize = True
            self.register_buffer(
                "mean", torch.tensor(mean).view(-1, 1, 1), persistent=False
            )
            self.register_buffer(
                "std", torch.tensor(std).view(-1, 1, 1), persistent=False
            )
        else:
            self._enable_normalize = False

    def _process_img(self, img: torch.Tensor) -> torch.Tensor:
        if self._channel_conversion:
            img = img[..., [2, 1, 0], :, :]
        img = img.float()
        if self._enable_normalize:
            shape = [1] * (img.dim() - 3) + list(self.mean.shape)
            mean = self.mean.view(shape)
            std = self.std.view(shape)
            img = (img - mean) / std
        if self.pad_size_divisor > 1:
            h, w = img.shape[-2:]
            d = self.pad_size_divisor
            ph = (d - h % d) % d
            pw = (d - w % d) % d
            if ph or pw:
                img = torch.nn.functional.pad(
                    img, (0, pw, 0, ph), value=self.pad_value
                )
        return img

    def forward(self, data: Union[dict, List[dict]], training: bool = False):
        if isinstance(data, list):
            return [self.forward(d, training) for d in data]
        data = self.cast_data(data)
        inputs = data["inputs"]
        if "img" in inputs and isinstance(inputs["img"], torch.Tensor):
            inputs["img"] = self._process_img(inputs["img"])
        return data
