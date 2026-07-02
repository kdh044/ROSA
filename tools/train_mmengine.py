import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sparse4D with MMEngine")
    parser.add_argument("config", help="Path to the train config")
    parser.add_argument("--work-dir", help="Directory for logs and checkpoints")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from a checkpoint path, or auto-resume when omitted",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use AMPOptimWrapper if the config uses a plain OptimWrapper",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="Job launcher for distributed training",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. key=value",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    if args.resume is not None:
        cfg.resume = True
        if args.resume != "auto":
            cfg.load_from = args.resume

    if args.amp and cfg.optim_wrapper.type == "OptimWrapper":
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.setdefault("loss_scale", "dynamic")

    cfg.launcher = args.launcher

    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
