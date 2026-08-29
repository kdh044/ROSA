"""Optimizer wrappers for paired native and ROSA experiments."""

from __future__ import annotations

import torch
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS

# Importing the optimizer integration is already part of every released
# config's custom_imports, so register the sparse checkpoint milestone beside
# the paired optimizer without adding another user-facing import setting.
from rosa.mmengine_hooks import ROSASparseMilestoneCheckpointHook  # noqa: F401

_SIDECAR_GROUP_KEY = "rosa_sidecar"


class _ROSAPairedClipMixin:
    """Clip native and sidecar gradients without coupling their norms."""

    def _clip_partition(
        self,
        params: list[torch.Tensor],
        metric_name: str,
    ) -> None:
        if not params:
            return
        grad = self.clip_func(params, **self.clip_grad_kwargs)
        if grad is not None:
            self.message_hub.update_scalar(metric_name, float(grad))

    def _clip_grad(self) -> None:
        native_params: list[torch.Tensor] = []
        sidecar_params: list[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            target = (
                sidecar_params
                if param_group.get(_SIDECAR_GROUP_KEY, False)
                else native_params
            )
            target.extend(
                param
                for param in param_group["params"]
                if param.requires_grad and param.grad is not None
            )

        self._clip_partition(native_params, f"train/{self.grad_name}")
        self._clip_partition(sidecar_params, f"train/rosa_{self.grad_name}")

    def load_state_dict(self, state_dict: dict) -> None:
        # Optimizer.load_state_dict replaces current parameter-group options
        # with the checkpoint's options. Preserve the config-owned partition so
        # checkpoints written before the marker existed remain resumable.
        sidecar_layout = [
            bool(group.get(_SIDECAR_GROUP_KEY, False))
            for group in self.optimizer.param_groups
        ]
        super().load_state_dict(state_dict)
        for group, is_sidecar in zip(self.optimizer.param_groups, sidecar_layout):
            if is_sidecar:
                group[_SIDECAR_GROUP_KEY] = True
            else:
                group.pop(_SIDECAR_GROUP_KEY, None)


@OPTIM_WRAPPERS.register_module()
class ROSAPairedClipOptimWrapper(_ROSAPairedClipMixin, OptimWrapper):
    """FP32 optimizer wrapper with independent native/sidecar clipping."""


@OPTIM_WRAPPERS.register_module()
class ROSAPairedAmpOptimWrapper(_ROSAPairedClipMixin, AmpOptimWrapper):
    """AMP optimizer wrapper with independent native/sidecar clipping."""


__all__ = [
    "ROSAPairedAmpOptimWrapper",
    "ROSAPairedClipOptimWrapper",
]
