"""MMEngine hooks imported only by execution paths that need MMEngine."""

from __future__ import annotations

from collections.abc import Mapping

from mmengine.hooks import Hook


class CheckpointContractHook(Hook):
    """Add the effective ROSA contract to checkpoints written by the CLI."""

    def __init__(self, metadata: Mapping) -> None:
        self.metadata = dict(metadata)

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        del runner
        checkpoint.setdefault("meta", {}).update(self.metadata)


__all__ = ["CheckpointContractHook"]
