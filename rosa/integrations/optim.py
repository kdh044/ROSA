"""Optimizer wrapper that clips one parameter group's gradients separately.

Parameter groups marked ``separate_clip=True`` in ``paramwise_cfg`` (the range
uncertainty heads) are clipped with their own norm, so that their gradients do
not shrink those of the detector.
"""

from __future__ import annotations

import torch
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS

_GROUP_KEY = "separate_clip"


@OPTIM_WRAPPERS.register_module()
class SeparateClipOptimWrapper(OptimWrapper):
    def _clip(self, params: list[torch.Tensor], metric_name: str) -> None:
        grad = self.clip_func(params, **self.clip_grad_kwargs)
        if grad is not None:
            self.message_hub.update_scalar(metric_name, float(grad))

    def _clip_grad(self) -> None:
        main_params: list[torch.Tensor] = []
        separate_params: list[torch.Tensor] = []
        for group in self.optimizer.param_groups:
            target = separate_params if group.get(_GROUP_KEY, False) else main_params
            target.extend(
                p for p in group["params"] if p.requires_grad and p.grad is not None
            )
        self._clip(main_params, f"train/{self.grad_name}")
        self._clip(separate_params, f"train/rosa_{self.grad_name}")

    def load_state_dict(self, state_dict: dict) -> None:
        # Keep the group markers of the config when resuming.
        layout = [bool(g.get(_GROUP_KEY, False)) for g in self.optimizer.param_groups]
        super().load_state_dict(state_dict)
        for group, separate in zip(self.optimizer.param_groups, layout):
            if separate:
                group[_GROUP_KEY] = True
            else:
                group.pop(_GROUP_KEY, None)


__all__ = ["SeparateClipOptimWrapper"]
