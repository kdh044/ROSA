# Installation

## Requirements

- Linux with an NVIDIA GPU and a driver for CUDA 12.8
- CUDA 12.8 toolkit (`nvcc` on `PATH`), used to compile mmcv and the ROSA kernels
- Python 3.10

The reported numbers were produced with:

| Component | Version |
|---|---|
| PyTorch | 2.11.0 + cu128 |
| mmcv | 2.2.0 |
| mmengine | 0.10.7 |
| mmdet | 3.2.0 |
| mmdet3d | 1.4.0 |

## Steps

```bash
git clone https://github.com/kdh044/ROSA && cd ROSA
git clone https://github.com/weiyangdaren/Fisheye3DOD external/fisheye3dod

python3.10 -m venv .venv && source .venv/bin/activate
bash tools/install.sh
```

Conda works the same way (`conda create -n rosa python=3.10 && conda activate rosa`).
The script refuses to run outside a virtual or conda environment.

`tools/install.sh`:

1. installs PyTorch 2.11 (cu128), `requirements.txt`, and
   [ocamcalib_undistort](https://github.com/matsuren/ocamcalib_undistort)
   (`ocamcamera`, used by the official Fisheye3DOD evaluator);
2. builds mmcv 2.2.0 from source with its CUDA ops (there is no prebuilt
   wheel for PyTorch 2.11; this takes 10–20 minutes);
3. lets mmdet 3.2 / mmdet3d 1.4 import with mmcv 2.2.0, whose version checks
   stop at 2.1 (`tools/relax_mmcv_check.py`);
4. installs ROSA with `pip install -e . --no-build-isolation`, which compiles
   the CUDA kernels in `projects/mmdet3d_plugin/ops`.

If no GPU is visible while building (e.g. in a container build), set
`TORCH_CUDA_ARCH_LIST`, for example `export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"`.

## Fisheye3DOD checkout

`external/fisheye3dod` provides the official Fisheye3DOD evaluator and the
VoVNet-99 implementation used for KITTI-360. If it lives elsewhere, set
`FISHEYE3DOD_OFFICIAL_ROOT=/path/to/Fisheye3DOD`.

## Check

```bash
python -c "from projects.mmdet3d_plugin.ops import rosa_point_attention_inference; print('CUDA ops OK')"
```

After changing the CUDA sources, rebuild with `pip install -e . --no-build-isolation`.
