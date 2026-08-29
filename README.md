# ROSA: Ray-aware Observation Selection and Aggregation for Surround-View Fisheye 3D Detection

Official implementation of **ROSA**.

DongHun Kim, Sang Jun Lee, HyungGi Jo<sup>†</sup> &nbsp;&nbsp;<sub><sup>†</sup> Corresponding author</sub>

A fisheye lens bends and compresses the image differently at every location,
so two samples of the same object are not interchangeable. ROSA measures the
condition of each projected observation — visibility, image location, scale,
distortion — and lets the model weight pyramid levels and camera views by it,
while the calibrated viewing ray is lifted into the feature through spherical
harmonics.

<p align="center">
  <img src="assets/figures/teaser.png" width="100%">
</p>

| | mAP | FDS / NDS | FPS |
|---|---|---|---|
| Fisheye3DOD — Sparse4D | 0.5644 | 0.6342 | 38.2 |
| Fisheye3DOD — **ROSA** | **0.6143** | **0.6647** | **43.2** |
| KITTI-360 — Sparse4D | 0.279 | 0.300 | — |
| KITTI-360 — **ROSA** | **0.311** | **0.338** | — |

---

## Installation

Requires **mmcv 2.x** — `mmcv-full` (1.x) imports `mmcv.runner` and will not
work. Full details in [docs/INSTALL.md](docs/INSTALL.md).

```bash
conda create -n rosa python=3.10 -y && conda activate rosa
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# ROSA reuses the released Fisheye3DOD evaluator and VoVNet-99 rather than
# vendoring third-party code
git clone https://github.com/weiyangdaren/Fisheye3DOD external/fisheye3dod
export FISHEYE3DOD_OFFICIAL_ROOT=$PWD/external/fisheye3dod
export PYTHONPATH=$PWD:$FISHEYE3DOD_OFFICIAL_ROOT:$PYTHONPATH
```

## Data

Download the benchmarks and point ROSA at them — see
[docs/DATA.md](docs/DATA.md) for the expected layout and for rebuilding the
KITTI-360 info files from the raw dataset.

```bash
export ROSA_DATA_ROOT=/path/to/Fisheye3DOD      # Main126 split
export ROSA_KITTI360_ROOT=/path/to/KITTI-360    # six-class, 2 Hz
```

## Train

```bash
# Fisheye3DOD — ResNet-18, 20 epochs, two GPUs
torchrun --standalone --nproc_per_node=2 tools/run.py \
    configs/fisheye3dod/rosa_r18_fisheye3dod_20e.py \
    --launcher pytorch train

# KITTI-360 — VoVNet-99, 24 epochs, two GPUs
torchrun --standalone --nproc_per_node=2 tools/train_kitti360.py \
    configs/kitti360/rosa_vov99_kitti360_24e.py \
    --launcher pytorch
```

The released configs carry the exact training recipe the reported numbers were
produced with: AdamW at `lr 2e-4` with `weight_decay 1e-3`, the image backbone
at half that learning rate, a 500-iteration linear warm-up into a cosine
schedule, gradient clipping at ℓ2 norm 25, batch size 4 per GPU, seed 3407.
Set `ROSA_SEED` to repeat a run under a different draw. Baseline configs for
both datasets are under [`benchmarks/`](benchmarks/).

## Evaluate

```bash
python tools/run.py configs/fisheye3dod/rosa_r18_fisheye3dod_20e.py \
    evaluate work_dirs/rosa_r18_fisheye3dod_20e/epoch_20.pth
```

## Benchmark runtime

```bash
python tools/run.py configs/fisheye3dod/rosa_r18_fisheye3dod_20e.py \
    benchmark --checkpoint <ckpt> --output runtime.json \
    --warmup 200 --iterations 300
```

Runtimes are batch size 1 over 300 timed iterations on a single RTX PRO 6000
Blackwell, with backbone and decoder layers compiled. `torch.compile` needs a
few hundred iterations before timings settle, so `--warmup 200` is not
optional for a fair number.

Pretrained checkpoints are on the [Releases](../../releases) page — see
[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md).

---

## Method

<p align="center">
  <img src="assets/figures/framework.png" width="100%">
</p>

Observation points generated from the current anchor are projected into the
fisheye views, where multi-level features are encoded together with the
geometry of their projection and the direction they were seen from. The
resulting ray-aware features are aggregated across camera views and then
across observation points, and the refined anchor decides where the next
decoder stage observes.

## Results

### Fisheye3DOD (Main126, ResNet-18, 20 epochs)

| Method | mAP | FDS | mATE | mASE | mAOE | FPS |
|---|---|---|---|---|---|---|
| Sparse4D | 0.5644 | 0.6342 | 0.4623 | 0.0858 | **0.3402** | 38.2 |
| **ROSA (ours)** | **0.6143** | **0.6647** | **0.4289** | **0.0788** | 0.3469 | **43.2** |

### KITTI-360 (six classes, VoVNet-99, 24 epochs)

| Method | mAP | NDS | mATE | AP<sub>car</sub> | AP<sub>ped</sub> | AP<sub>bic</sub> |
|---|---|---|---|---|---|---|
| Sparse4D | 0.279 | 0.300 | 0.589 | 0.631 | 0.246 | 0.127 |
| **ROSA (ours)** | **0.311** | **0.338** | **0.587** | **0.650** | **0.304** | **0.180** |

### Qualitative

Predictions rendered as LiDAR BEV point clouds — a sparse scene above, a dense
occluded scene below; red boxes mark a small object and objects at 44 m.

<p align="center">
  <img src="assets/figures/qualitative_bev.png" width="100%">
</p>

Fisheye image domain, Sparse4D on top and ROSA below. The three left columns
are Fisheye3DOD — a distant object, a peripheral vehicle under strong
distortion, and a large close-range object, each recovered only by ROSA. The
two right columns are real KITTI-360 recordings, where both objects lie
outside either rectified camera's field of view.

<p align="center">
  <img src="assets/figures/qualitative_fisheye.png" width="100%">
</p>

## Repository layout

```
rosa/                     core package
├── observation.py        observation cues and multi-level sampling
├── fisheye_evidence.py   condition-aware camera/level evidence
├── rfl.py                ray feature lifting (spherical harmonics)
├── attention.py          camera and point attention
├── refiner.py            gated box refinement
├── hybrid_box_decoder.py direction + log-range box state
├── fisheye_projection.py OCam projection (Fisheye3DOD)
├── mixed_projection.py   pinhole + MEI projection (KITTI-360)
└── integrations/         dataset, metric and detector bindings
configs/                  released training configurations
├── fisheye3dod/          ResNet-18, 20 epochs
└── kitti360/             VoVNet-99, 24 epochs
benchmarks/               baseline detectors, per dataset
tools/                    train / evaluate / benchmark entry points
└── data_converter/       KITTI-360 info and anchor builders
assets/                   anchor files and figures
```

## Citation

```bibtex
@inproceedings{kim2027rosa,
  title     = {ROSA: Ray-aware Observation Selection and Aggregation for
               Surround-View Fisheye 3D Detection},
  author    = {Kim, DongHun and Lee, Sang Jun and Jo, HyungGi},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications
               of Computer Vision (WACV)},
  year      = {2027}
}
```

## Acknowledgement

ROSA is built on [Sparse4D](https://github.com/HorizonRobotics/Sparse4D) and
evaluated with the [Fisheye3DOD](https://github.com/weiyangdaren/Fisheye3DOD)
benchmark. The KITTI-360 arm uses the FishBEVOD VoVNet-99 backbone released
with Fisheye3DOD. We thank the authors of both projects.
