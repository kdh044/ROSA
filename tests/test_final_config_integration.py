from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_final_config_builds_with_pinned_sparse4d_host():
    host = Path(
        os.environ.get(
            "ROSA_DETECTOR_HOST_ROOT",
            ROOT / "external" / "detector_host",
        )
    ).resolve()
    evaluator = Path(
        os.environ.get(
            "FISHEYE3DOD_OFFICIAL_ROOT",
            ROOT / "external" / "fisheye3dod",
        )
    ).resolve()
    if not host.is_dir() or not evaluator.is_dir():
        pytest.skip("optional detector host and evaluator are not available")

    code = textwrap.dedent(
        """
        from mmengine.config import Config
        from mmengine.utils import import_modules_from_strings
        from mmdet3d.registry import MODELS
        from mmdet3d.utils import register_all_modules

        register_all_modules(init_default_scope=True)
        config = Config.fromfile("configs/rosa_r18_main126.py")
        import_modules_from_strings(**config.custom_imports)
        head = MODELS.build(config.model.head)
        assert type(head).__name__ == "FisheyeSparseBoxHead"
        decoder = head.fisheye_box_decoder
        assert decoder.config.camera_rfl_before_rosa
        assert decoder.config.sparse4d_size_head_layers == tuple(range(6))
        assert head.instance_bank.num_temp_instances == 0
        """
    )
    environment = os.environ.copy()
    environment["ROSA_DETECTOR_HOST_ROOT"] = str(host)
    environment["FISHEYE3DOD_OFFICIAL_ROOT"] = str(evaluator)
    pythonpath = [str(ROOT / "src"), str(host)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
