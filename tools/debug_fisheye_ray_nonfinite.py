"""Reproduce and localize the first non-finite V2 parameter gradient."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dist import get_rank
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--detect-anomaly", action="store_true")
    return parser.parse_args()


def _finite_gradient_hook(name: str):
    def check(gradient: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(gradient)
        if finite.all():
            return gradient
        finite_values = gradient[finite]
        finite_abs_max = (
            float(finite_values.abs().max()) if finite_values.numel() else None
        )
        nonfinite_indices = torch.nonzero(~finite, as_tuple=False).tolist()[:16]
        raise FloatingPointError(
            "NONFINITE_PARAMETER_GRADIENT "
            f"rank={get_rank()} name={name} shape={tuple(gradient.shape)} "
            f"nan={int(torch.isnan(gradient).sum())} "
            f"posinf={int(torch.isposinf(gradient).sum())} "
            f"neginf={int(torch.isneginf(gradient).sum())} "
            f"finite_abs_max={finite_abs_max} "
            f"nonfinite_indices={nonfinite_indices}"
        )

    return check


def main() -> None:
    args = parse_args()
    if args.max_iters <= 0:
        raise ValueError("max-iters must be positive")
    config = Config.fromfile(args.config)
    config.launcher = "pytorch"
    config.work_dir = str(args.work_dir.resolve())
    config.train_cfg = dict(
        type="IterBasedTrainLoop",
        max_iters=args.max_iters,
        val_interval=args.max_iters + 1,
    )
    config.val_cfg = None
    config.val_dataloader = None
    config.val_evaluator = None
    config.train_dataloader.num_workers = 2
    config.train_dataloader.persistent_workers = False
    config.default_hooks.logger.interval = 10
    runner = Runner.from_cfg(config)
    registered = 0
    for name, parameter in runner.model.named_parameters():
        if parameter.requires_grad:
            parameter.register_hook(_finite_gradient_hook(name))
            registered += 1
    print(
        f"FINITE_GRADIENT_HOOKS_REGISTERED rank={get_rank()} count={registered}",
        flush=True,
    )
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    runner.train()
    print(
        f"NONFINITE_REPRODUCTION_COMPLETE rank={get_rank()} iter={runner.iter}",
        flush=True,
    )


if __name__ == "__main__":
    main()
