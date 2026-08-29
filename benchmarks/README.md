# Benchmarks

The baseline detectors ROSA is compared against in the paper, kept apart from
`rosa/` because they are other people's methods. Each dataset keeps its own
schedule, backbone and info files, so the two are not interchangeable.

```
benchmarks/
├── fisheye3dod/configs/   ResNet-18, 20 epochs, OCam projection
└── kitti360/
    ├── configs/           VoVNet-99, 24 epochs, mixed pinhole + MEI
    └── heads/             the detector heads ported to the mixed-lens rig
```

Every arm is trained from scratch under the same recipe as ROSA — same
optimizer, schedule, batch size and seed — so the gap between rows reflects
the detector rather than the training budget.

Running them needs the original implementations on `PYTHONPATH`; see the
`custom_imports` block at the top of each config for which ones. The KITTI-360
heads under `heads/` are the ports that adapt those detectors to a rig mixing
two lens families, and are imported by the configs beside them.

```bash
torchrun --standalone --nproc_per_node=2 tools/train_kitti360.py \
    benchmarks/kitti360/configs/sparse4d_kitti360_6class_vov99_24e.py \
    --launcher pytorch
```
