# Installation

**Requirements:** Linux, an NVIDIA GPU, the CUDA 12.8 toolkit (`nvcc`), Python 3.10.

```bash
git clone https://github.com/kdh044/ROSA && cd ROSA
git clone https://github.com/weiyangdaren/Fisheye3DOD external/fisheye3dod

python3.10 -m venv .venv && source .venv/bin/activate   # or: conda create -n rosa python=3.10 && conda activate rosa
bash tools/install.sh
```

The last line prints `ROSA install OK` when everything is in place.

`external/fisheye3dod` is the official Fisheye3DOD code; ROSA uses its
evaluator and its VoVNet-99 backbone. If you keep it elsewhere, set
`FISHEYE3DOD_OFFICIAL_ROOT=/path/to/Fisheye3DOD`.

## What `tools/install.sh` does

| Step | Notes |
|---|---|
| PyTorch 2.11 (CUDA 12.8), `requirements.txt` | mmengine 0.10.7, mmdet 3.2.0, mmdet3d 1.4.0 |
| [ocamcalib_undistort](https://github.com/matsuren/ocamcalib_undistort) | OCam camera model used by the Fisheye3DOD evaluator |
| mmcv 2.2.0, built from source | no prebuilt wheel for PyTorch 2.11; takes 10–20 minutes |
| `tools/relax_mmcv_check.py` | mmdet 3.2 / mmdet3d 1.4 refuse mmcv 2.2.0 by default; ROSA was run with it |
| `pip install -e .` | installs ROSA and compiles its CUDA kernels |

The script stops if no virtual or conda environment is active. If no GPU is
visible while building (e.g. inside `docker build`), set
`TORCH_CUDA_ARCH_LIST`, e.g. `export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"`.
After editing the CUDA sources, rebuild with `pip install -e . --no-build-isolation`.
