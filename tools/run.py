#!/usr/bin/env python3
"""Train or evaluate ROSA through MMEngine."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--launcher",
        choices=("none", "pytorch", "slurm", "mpi"),
        default="none",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config entries as key=value pairs.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    _add_common_arguments(train)
    train.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from a trusted local checkpoint path or auto-discover it.",
    )
    train.add_argument("--amp", action="store_true")

    evaluate = commands.add_parser("evaluate")
    _add_common_arguments(evaluate)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--dump", type=Path)

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    config = Config.fromfile(args.config)
    if args.cfg_options:
        config.merge_from_dict(args.cfg_options)
    if args.work_dir:
        config.work_dir = str(args.work_dir.resolve())
    config.launcher = args.launcher
    return config


def train(args: argparse.Namespace, config: Config) -> None:
    if args.resume:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        config.resume = True
        config.load_from = None if args.resume == "auto" else args.resume
    if args.amp:
        if config.optim_wrapper.type != "OptimWrapper":
            raise ValueError("--amp requires a plain OptimWrapper")
        config.optim_wrapper.type = "AmpOptimWrapper"
        config.optim_wrapper.setdefault("loss_scale", "dynamic")
    Runner.from_cfg(config).train()


def evaluate(args: argparse.Namespace, config: Config) -> None:
    config.load_from = str(args.checkpoint.resolve())
    config.resume = False
    if args.dump:
        output = args.dump.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        dump_metric = dict(type="DumpResults", out_file_path=str(output))
        evaluator = config.test_evaluator
        config.test_evaluator = (
            [*evaluator, dump_metric]
            if isinstance(evaluator, (list, tuple))
            else [evaluator, dump_metric]
        )
    Runner.from_cfg(config).test()


def main() -> None:
    args = parse_args()
    config = load_config(args)
    if args.command == "train":
        train(args, config)
    else:
        evaluate(args, config)


if __name__ == "__main__":
    main()
