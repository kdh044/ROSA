#!/usr/bin/env python3
"""Train, evaluate, or benchmark ROSA through MMEngine."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time
import warnings

import torch
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


_FAST_PROFILES = frozenset({"fast", "compact_v2_fast"})
_KNOWN_RELEASE_ARCHITECTURE = "ROSA R18 Main126"
_ROSA_TOPOLOGY_FIELDS = (
    "architecture_profile",
    "embed_dims",
    "hidden_dims",
    "num_keypoints",
    "num_cameras",
    "num_levels",
    "enable_quality_score",
)


class _CheckpointContractHook(Hook):
    """Add the effective ROSA contract to checkpoints written by this CLI."""

    def __init__(self, metadata: Mapping) -> None:
        self.metadata = dict(metadata)

    def before_save_checkpoint(
        self,
        runner: Runner,
        checkpoint: dict,
    ) -> None:
        del runner
        checkpoint.setdefault("meta", {}).update(self.metadata)


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
    evaluate.add_argument(
        "--allow-checkpoint-contract-mismatch",
        action="store_true",
    )

    benchmark = commands.add_parser("benchmark")
    _add_common_arguments(benchmark)
    benchmark.add_argument("--checkpoint", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--warmup", type=int, default=20)
    benchmark.add_argument("--iterations", type=int, default=100)
    benchmark.add_argument(
        "--baseline",
        action="store_true",
        help="Benchmark the same config with the ROSA sidecar removed.",
    )
    benchmark.add_argument(
        "--allow-checkpoint-contract-mismatch",
        action="store_true",
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


def _rosa_profile(config: Config) -> str | None:
    rosa_config = config.model.get("head", {}).get("rosa_cfg")
    return (
        rosa_config.get("profile", "explicit")
        if rosa_config is not None
        else None
    )


def _effective_rosa_config(config: Config):
    rosa_mapping = config.model.get("head", {}).get("rosa_cfg")
    if rosa_mapping is None:
        return None

    from rosa import ROSAConfig

    return ROSAConfig.from_mapping(rosa_mapping)


def _checkpoint_contract_metadata(config: Config) -> dict | None:
    effective = _effective_rosa_config(config)
    if effective is None:
        return None
    return {
        "rosa_contract_version": 1,
        "projection_contract": effective.projection_contract,
        "refinement_profile": _rosa_profile(config),
        "rosa_config": asdict(effective),
    }


def _checkpoint_state_dict(checkpoint: Mapping) -> Mapping | None:
    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, Mapping):
        return state_dict
    if checkpoint and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint.items()
    ):
        return checkpoint
    return None


def _state_dict_has_rosa(state_dict: Mapping | None) -> bool | None:
    if state_dict is None:
        return None
    return any(
        any(part == "rosa" or part.startswith("rosa_") for part in key.split("."))
        for key in state_dict
        if isinstance(key, str)
    )


def _topology_signature(config) -> tuple:
    return tuple(getattr(config, name) for name in _ROSA_TOPOLOGY_FIELDS)


def _validate_checkpoint_contract(
    checkpoint_path: Path,
    config: Config,
    *,
    allow_mismatch: bool,
) -> None:
    """Reject silent evaluation under a different projection/topology."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        return
    metadata = checkpoint.get("meta")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    rosa_mapping = config.model.get("head", {}).get("rosa_cfg")
    effective = _effective_rosa_config(config)
    target_has_rosa = rosa_mapping is not None
    checkpoint_has_rosa = _state_dict_has_rosa(
        _checkpoint_state_dict(checkpoint)
    )
    mismatches = []

    if (
        checkpoint_has_rosa is not None
        and checkpoint_has_rosa != target_has_rosa
    ):
        mismatches.append(
            "topology "
            f"checkpoint ROSA={checkpoint_has_rosa} != "
            f"config ROSA={target_has_rosa}"
        )

    checkpoint_contract = metadata.get("projection_contract")
    if target_has_rosa and checkpoint_has_rosa is not False:
        if checkpoint_contract is None:
            mismatches.append(
                "ROSA checkpoint lacks projection_contract metadata"
            )
        elif checkpoint_contract != effective.projection_contract:
            mismatches.append(
                "projection "
                f"{checkpoint_contract!r} != "
                f"{effective.projection_contract!r}"
            )

    checkpoint_rosa_config = metadata.get("rosa_config")
    if target_has_rosa and isinstance(checkpoint_rosa_config, Mapping):
        from rosa import ROSAConfig

        try:
            source_effective = ROSAConfig.from_mapping(checkpoint_rosa_config)
        except (TypeError, ValueError) as error:
            mismatches.append(f"invalid checkpoint ROSA config ({error})")
        else:
            if _topology_signature(source_effective) != _topology_signature(
                effective
            ):
                mismatches.append(
                    "ROSA learned topology differs from the checkpoint"
                )
    elif (
        target_has_rosa
        and metadata.get("architecture") == _KNOWN_RELEASE_ARCHITECTURE
        and effective.architecture_profile != "legacy"
    ):
        mismatches.append(
            "published checkpoint cannot initialize compact ROSA topology"
        )

    if not mismatches:
        return
    message = "checkpoint/config contract mismatch: " + "; ".join(mismatches)
    if not allow_mismatch:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def _compile_fast_inference(runner: Runner, config: Config) -> bool:
    """Compile only the latency-oriented sidecar after weights are loaded."""

    if _rosa_profile(config) not in _FAST_PROFILES:
        return False
    model = runner.model.module if is_model_wrapper(runner.model) else runner.model
    refiner = getattr(getattr(model, "head", None), "rosa_refiner", None)
    if refiner is None:
        raise RuntimeError("fast profile is missing its ROSA refiner")
    if getattr(refiner, "_compiled_call_impl", None) is None:
        refiner.compile(mode="reduce-overhead", fullgraph=False)
    return True


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
    runner = Runner.from_cfg(config)
    metadata = _checkpoint_contract_metadata(config)
    if metadata is not None:
        runner.register_hook(
            _CheckpointContractHook(metadata),
            priority="VERY_LOW",
        )
    runner.train()


def evaluate(args: argparse.Namespace, config: Config) -> None:
    _validate_checkpoint_contract(
        args.checkpoint.resolve(),
        config,
        allow_mismatch=args.allow_checkpoint_contract_mismatch,
    )
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
    runner.load_or_resume()
    _compile_fast_inference(runner, config)
    runner.test()


def benchmark(args: argparse.Namespace, config: Config) -> None:
    """Measure end-to-end single-GPU ``test_step`` latency."""

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    # A baseline intentionally removes the sidecar, so its latency is
    # independent of the ROSA projection/profile contract.
    if args.checkpoint is not None and not args.baseline:
        _validate_checkpoint_contract(
            args.checkpoint.resolve(),
            config,
            allow_mismatch=args.allow_checkpoint_contract_mismatch,
        )
    if args.baseline:
        head_config = config.model.get("head")
        if head_config is None or "rosa_cfg" not in head_config:
            raise ValueError("--baseline requires a ROSA config")
        head_config.type = "Sparse4DHead"
        head_config.pop("rosa_cfg")
        head_config.pop("inference_box_source_by_dimension", None)
    if args.checkpoint is not None:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        config.load_from = str(args.checkpoint.resolve())
    else:
        config.load_from = None
    config.resume = False
    if args.work_dir is None:
        config.work_dir = str(args.output.resolve().parent / "_benchmark")

    runner = Runner.from_cfg(config)
    runner.load_or_resume()
    compiled = _compile_fast_inference(runner, config)
    model = runner.model.eval()
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
    profile = (
        "baseline"
        if args.baseline
        else (_rosa_profile(config) or "none")
    )
    result = {
        "config": str(args.config.resolve()),
        "profile": profile,
        "compiled_refiner": compiled,
        "checkpoint": (
            str(args.checkpoint.resolve())
            if args.checkpoint is not None
            else None
        ),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "samples": sample_count,
        "mean_batch_size": sample_count / len(timings),
        "mean_iteration_ms": 1000.0 * statistics.mean(timings),
        "median_iteration_ms": 1000.0 * statistics.median(timings),
        "p95_iteration_ms": 1000.0
        * float(torch.tensor(timings).quantile(0.95)),
        "mean_latency_per_sample_ms": 1000.0 * total_time / sample_count,
        "samples_per_second": sample_count / total_time,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated()
        / (1024.0 * 1024.0),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    config = load_config(args)
    if args.command == "train":
        train(args, config)
    elif args.command == "evaluate":
        evaluate(args, config)
    else:
        benchmark(args, config)


if __name__ == "__main__":
    main()
