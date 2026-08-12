"""Run a 200-iteration DDP stability preflight for the V2 experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config
from mmengine.dist import is_main_process
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.fromfile(args.config)
    config.launcher = "pytorch"
    config.work_dir = str(args.work_dir.resolve())
    config.train_cfg = dict(
        type="IterBasedTrainLoop",
        max_iters=200,
        val_interval=1000,
    )
    config.val_cfg = None
    config.val_dataloader = None
    config.val_evaluator = None
    config.train_dataloader.num_workers = 2
    config.train_dataloader.persistent_workers = False
    config.default_hooks.logger.interval = 20
    runner = Runner.from_cfg(config)
    runner.train()
    if is_main_process():
        print(f"DISTRIBUTED_STABILITY_PREFLIGHT_COMPLETE {runner.iter}", flush=True)


if __name__ == "__main__":
    main()
