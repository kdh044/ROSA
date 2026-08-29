"""Minimal MMEngine training entry point for the KITTI-360 port.

The V49 workspace ships a much larger ``tools/run.py`` that also enforces
checkpoint contracts and captures provenance.  It is not reusable here: it
imports ``ROSAConfig``, ``ROSARefiner`` and a ``checkpoint_contract`` module
that this port tree does not contain, because the tree was assembled for the
KITTI-360 camera work rather than cloned from the V49 release.  Dragging those
modules across would re-introduce exactly the source-tree divergence this
experiment is meant to avoid, so training goes through the stock Runner and the
config alone decides the behaviour.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ROSA KITTI-360 config.")
    parser.add_argument("config")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir
    if args.resume:
        # mmengine resumes from ``load_from`` when both are set, which here is
        # the FCOS3D backbone checkpoint -- it carries no optimizer or epoch
        # state, so the run would silently restart (and crash on the missing
        # meta).  Clearing it lets the runner auto-resume from the newest
        # checkpoint in the work directory, which is what --resume means.
        cfg.resume = True
        cfg.load_from = None

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
