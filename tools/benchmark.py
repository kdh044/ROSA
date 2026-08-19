#!/usr/bin/env python3
"""Measure single-GPU end-to-end MMEngine inference latency and memory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = "none"
    cfg.load_from = args.checkpoint
    cfg.resume = False
    cfg.work_dir = str(Path(args.output).resolve().parent / "_benchmark")
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model
    model.eval()
    data_loader = runner.test_dataloader
    iterator = iter(data_loader)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    torch.cuda.reset_peak_memory_stats()
    timings = []
    sample_count = 0
    with torch.inference_mode():
        for iteration in range(args.warmup + args.iterations):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                batch = next(iterator)
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.test_step(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if iteration >= args.warmup:
                timings.append(elapsed)
                sample_count += len(output)
    total = sum(timings)
    mean_batch_size = sample_count / len(timings)
    result = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "samples": sample_count,
        "mean_batch_size": mean_batch_size,
        "mean_iteration_ms": 1000.0 * statistics.mean(timings),
        "median_iteration_ms": 1000.0 * statistics.median(timings),
        "p95_iteration_ms": 1000.0
        * float(torch.tensor(timings).quantile(0.95)),
        "mean_latency_per_sample_ms": 1000.0 * total / sample_count,
        "samples_per_second": sample_count / total,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated()
        / (1024.0 * 1024.0),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
