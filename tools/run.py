#!/usr/bin/env python3
"""Train, evaluate, or benchmark ROSA through MMEngine."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import statistics
import time

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# MMEngine checkpoints store numpy objects, which torch>=2.6 rejects by default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


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
        help="Resume from a checkpoint path, or from the latest one in the work dir.",
    )

    evaluate = commands.add_parser("evaluate")
    _add_common_arguments(evaluate)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--dump", type=Path, help="Also save raw predictions.")

    benchmark = commands.add_parser("benchmark")
    _add_common_arguments(benchmark)
    benchmark.add_argument("--checkpoint", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--warmup", type=int, default=200)
    benchmark.add_argument("--iterations", type=int, default=300)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument(
        "--no-compile",
        action="store_true",
        help="Run eagerly instead of compiling backbone, neck and decoder layers.",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    config = Config.fromfile(args.config)
    if args.cfg_options:
        config.merge_from_dict(args.cfg_options)
    if args.work_dir:
        config.work_dir = str(args.work_dir.resolve())
    config.launcher = args.launcher
    return config


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_tree(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{field.name: _clone_tree(getattr(value, field.name)) for field in dataclasses.fields(value)},
        )
    return value


def _compile_for_benchmark(model: torch.nn.Module) -> list[str]:
    """Compile backbone, neck and decoder layers with CUDA graphs (reduce-overhead).

    Outputs are cloned because CUDA-graph buffers are reused between calls, and the
    RFL weight cache is built eagerly so that it is not created inside a graph.
    """

    for module in model.modules():
        if hasattr(module, "streaming_compact_weights"):
            with torch.no_grad():
                key, reliability = (t.detach().clone() for t in module.streaming_compact_weights())
            module.streaming_compact_weights = lambda key=key, reliability=reliability: (key, reliability)

    def compile_inplace(module: torch.nn.Module) -> None:
        compiled = torch.compile(module.forward, mode="reduce-overhead")
        module.forward = lambda *args, **kwargs: _clone_tree(compiled(*args, **kwargs))

    targets = []
    for name in ("img_backbone", "img_neck"):
        module = getattr(model, name, None)
        if module is not None:
            compile_inplace(module)
            targets.append(name)
    decoder = getattr(getattr(model, "head", None), "fisheye_box_decoder", None)
    if decoder is not None:
        for layer in decoder.layers:
            compile_inplace(layer)
        targets.append(f"{len(decoder.layers)} decoder layers")
    return targets


def train(args: argparse.Namespace, config: Config) -> None:
    if args.resume:
        config.resume = True
        config.load_from = None if args.resume == "auto" else args.resume
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
    runner = Runner.from_cfg(config)
    runner.test()


def benchmark(args: argparse.Namespace, config: Config) -> None:
    """Measure end-to-end single-GPU ``test_step`` latency."""

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    config.load_from = str(args.checkpoint.resolve()) if args.checkpoint else None
    config.resume = False
    if args.work_dir is None:
        config.work_dir = str(args.output.resolve().parent / "_benchmark")
    config.test_dataloader.batch_size = args.batch_size

    runner = Runner.from_cfg(config)
    runner.load_or_resume()
    model = runner.model.eval()
    compiled = [] if args.no_compile else _compile_for_benchmark(model)
    iterator = iter(runner.test_dataloader)
    timings: list[float] = []
    sample_count = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index in range(args.warmup + args.iterations):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(runner.test_dataloader)
                batch = next(iterator)
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.test_step(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= args.warmup:
                timings.append(elapsed)
                sample_count += len(output)

    total_time = sum(timings)
    result = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "compiled": compiled,
        "batch_size": args.batch_size,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "mean_iteration_ms": 1000.0 * statistics.mean(timings),
        "median_iteration_ms": 1000.0 * statistics.median(timings),
        "mean_latency_per_sample_ms": 1000.0 * total_time / sample_count,
        "samples_per_second": sample_count / total_time,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    config = load_config(args)
    {"train": train, "evaluate": evaluate, "benchmark": benchmark}[args.command](args, config)


if __name__ == "__main__":
    main()
