#!/usr/bin/env bash
# Install ROSA into the active Python environment (venv or conda).
# Needs a CUDA 12.8 toolkit (nvcc) and a GPU driver that supports it.
#
#   python3.10 -m venv .venv && source .venv/bin/activate
#   bash tools/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Activate a virtualenv or conda environment first." >&2
    exit 1
fi

pip install -U pip "setuptools<70" wheel ninja
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# OCam camera model used by the official Fisheye3DOD evaluator.
pip install "git+https://github.com/matsuren/ocamcalib_undistort.git@5c631c3220c036722e17f57a71549f5577cdf289"
# mmdet3d 1.4 pins matplotlib<3.6, whose wheels are built for numpy 1.x.
pip install matplotlib==3.10.9

# mmcv has no prebuilt wheel for torch 2.11; build it with its CUDA ops.
MMCV_WITH_OPS=1 pip install mmcv==2.2.0 --no-build-isolation

# mmdet 3.2 / mmdet3d 1.4 accept mmcv < 2.2.0 only; ROSA was run with 2.2.0.
python tools/relax_mmcv_check.py

# ROSA itself and its CUDA extension.
pip install -e . --no-build-isolation

python -c "from projects.mmdet3d_plugin.ops import rosa_point_attention_inference; print('ROSA install OK')"
