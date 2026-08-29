"""MMEngine hooks imported only by execution paths that need MMEngine."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


class CheckpointContractHook(Hook):
    """Add the effective ROSA contract to checkpoints written by the CLI."""

    def __init__(self, metadata: Mapping) -> None:
        self.metadata = dict(metadata)

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        del runner
        metadata = dict(self.metadata)
        sealed = metadata.get("sealed_training_contract")
        if sealed is not None:
            from rosa.checkpoint_contract import checkpoint_payload_contract

            payload_contract = checkpoint_payload_contract(checkpoint)
            expected_model = sealed.get("model_state_contract")
            if payload_contract["model_state"] != expected_model:
                raise ValueError(
                    "checkpoint model state changed from the sealed runner schema"
                )
            metadata["sealed_payload_contract"] = payload_contract
        checkpoint.setdefault("meta", {}).update(metadata)


class ResumeContractValidationHook(Hook):
    """Validate built optimizer/schedulers before MMEngine loads a resume."""

    priority = "HIGHEST"

    def __init__(self, payload_contract: Mapping) -> None:
        self.payload_contract = dict(payload_contract)

    def before_run(self, runner) -> None:
        from rosa.checkpoint_contract import validate_runner_resume_contract

        validate_runner_resume_contract(runner, self.payload_contract)


@HOOKS.register_module()
class ROSASparseMilestoneCheckpointHook(Hook):
    """Protect a few resume milestones without duplicating checkpoint bytes."""

    priority = "LOWEST"

    def __init__(self, milestones: tuple[int, ...] = (8,)) -> None:
        if (
            not isinstance(milestones, (tuple, list))
            or not milestones
            or any(
                not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0
                for epoch in milestones
            )
        ):
            raise ValueError("milestones must be positive integer epochs")
        if len(set(milestones)) != len(milestones):
            raise ValueError("milestone epochs must be unique")
        self.milestones = tuple(sorted(milestones))

    @master_only
    def after_train_epoch(self, runner) -> None:
        epoch = int(runner.epoch) + 1
        if epoch not in self.milestones:
            return
        work_dir = Path(runner.work_dir).resolve()
        source = work_dir / f"epoch_{epoch}.pth"
        target = work_dir / f"milestone_epoch_{epoch}.pth"
        if not source.is_file():
            raise FileNotFoundError(
                f"checkpoint hook did not create required milestone source {source}"
            )
        if target.exists():
            source_stat = source.stat()
            target_stat = target.stat()
            if (
                source_stat.st_dev != target_stat.st_dev
                or source_stat.st_ino != target_stat.st_ino
            ):
                raise FileExistsError(
                    f"milestone target already exists with different bytes: {target}"
                )
            return
        os.link(source, target)
        directory_fd = os.open(work_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


__all__ = [
    "CheckpointContractHook",
    "ROSASparseMilestoneCheckpointHook",
    "ResumeContractValidationHook",
]
