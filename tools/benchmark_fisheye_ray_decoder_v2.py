"""Benchmark C0 and V2 on the same native-fisheye batch.

The script intentionally measures the complete detector boundary, including
image backbone, FPN, head, and (in training mode) matching/loss/backward.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet3d.registry import MODELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    ROOT / "configs" / "rosa_r18_main126_v5_c0_sparse4d_raw_uniform_30e.py",
    ROOT / "configs" / "fisheye_ray_decoder_v2_raw_uniform_20e.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_CONFIGS),
    )
    parser.add_argument("--mode", choices=("inference", "train"), default="inference")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--compile-head",
        action="store_true",
        help="compile each complete detector head with reduce-overhead mode",
    )
    return parser.parse_args()


def build_batch(config: Config, batch_size: int, seed: int):
    loader_config = config.train_dataloader.copy()
    loader_config.batch_size = batch_size
    loader_config.num_workers = 0
    loader_config.persistent_workers = False
    loader = Runner.build_dataloader(
        loader_config,
        seed=seed,
        diff_rank_seed=False,
    )
    return next(iter(loader))


def timed_iteration(model, processed, mode: str):
    if mode == "inference":
        with torch.inference_mode():
            return model(**processed, mode="tensor")
    model.zero_grad(set_to_none=True)
    losses = model(**processed, mode="loss")
    total = sum(losses.values())
    total.backward()
    return total


def common_output_max_abs(reference, candidate):
    differences = {}
    for name in ("classification", "prediction", "quality"):
        if name not in reference or name not in candidate:
            continue
        pairs = [
            (left, right)
            for left, right in zip(reference[name], candidate[name])
            if torch.is_tensor(left) and torch.is_tensor(right)
        ]
        if pairs:
            differences[name] = max(
                float((left.float() - right.float()).abs().max())
                for left, right in pairs
            )
    return differences


def benchmark(config_path: Path, args, device: torch.device):
    torch.manual_seed(args.seed)
    config = Config.fromfile(config_path)
    deterministic = bool(config.get("randomness", {}).get("deterministic", False))
    # Match the runner contract.  In particular, V2 training must exercise
    # its deterministic four-gather sampler rather than a CUDA grid-sample
    # backward that the real experiment would reject.
    torch.use_deterministic_algorithms(deterministic)
    import_modules_from_strings(**config.custom_imports)
    init_default_scope(config.get("default_scope", "mmdet3d"))
    batch = build_batch(config, args.batch_size, args.seed)
    model = MODELS.build(config.model).to(device)
    model.train(args.mode == "train")
    processed = model.data_preprocessor(batch, training=args.mode == "train")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parity_reference = None
    if args.compile_head:
        if args.mode != "inference":
            raise ValueError("--compile-head is currently sealed to inference")
        parity_reference = timed_iteration(model, processed, args.mode)
        model.head = torch.compile(
            model.head,
            mode="reduce-overhead",
            fullgraph=False,
        )

    warmup_start = time.perf_counter()
    warmup_output = None
    for _ in range(args.warmup):
        warmup_output = timed_iteration(model, processed, args.mode)
    torch.cuda.synchronize(device)
    warmup_wall_seconds = time.perf_counter() - warmup_start
    baseline_memory = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    elapsed_ms = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        timed_iteration(model, processed, args.mode)
        end.record()
        end.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))

    peak_memory = torch.cuda.max_memory_allocated(device)
    result = {
        "config": str(config_path.resolve()),
        "model_type": config.model.type,
        "head_type": config.model.head.type,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters": parameter_count,
        "compiled_head": args.compile_head,
        "deterministic_algorithms": deterministic,
        "warmup_wall_seconds": warmup_wall_seconds,
        "latency_ms_mean": statistics.fmean(elapsed_ms),
        "latency_ms_median": statistics.median(elapsed_ms),
        "latency_ms_p95": sorted(elapsed_ms)[
            min(len(elapsed_ms) - 1, int(0.95 * len(elapsed_ms)))
        ],
        "throughput_frames_s": args.batch_size
        / (statistics.fmean(elapsed_ms) / 1000.0),
        "memory_baseline_mib": baseline_memory / 2**20,
        "memory_peak_mib": peak_memory / 2**20,
        "memory_iteration_delta_mib": (peak_memory - baseline_memory) / 2**20,
    }
    if parity_reference is not None:
        result["compile_parity_max_abs"] = common_output_max_abs(
            parity_reference,
            warmup_output,
        )
    del processed, batch, model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.1)
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency/memory benchmark")
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("invalid benchmark iteration or batch count")
    if args.compile_head and args.warmup == 0:
        raise ValueError("compiled benchmark requires at least one warmup")
    device = torch.device("cuda", 0)
    results = [benchmark(path, args, device) for path in args.configs]
    if len(results) == 2:
        results.append(
            {
                "comparison": "second_vs_first",
                "latency_ratio_mean": results[1]["latency_ms_mean"]
                / results[0]["latency_ms_mean"],
                "latency_ratio_median": results[1]["latency_ms_median"]
                / results[0]["latency_ms_median"],
                "peak_memory_ratio": results[1]["memory_peak_mib"]
                / results[0]["memory_peak_mib"],
            }
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
