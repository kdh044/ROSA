# Fisheye Sparse-Ray Detector V2: final 20-epoch report

## Method identity

V2 is not a Sparse4D variant name. Sparse4D C0 is the baseline and supplies
the released backbone/FPN, matcher, losses, decoder schedule, output decoder,
and evaluator infrastructure. V2 replaces the defining temporal InstanceBank,
box-conditioned 13-keypoint deformable aggregation, and Cartesian box
refinement with a `FisheyeSparseRayHead`.

V2 retains six-stage decoder-to-decoder query/state iteration, which is not
cross-frame Sparse4D query propagation. Its detection evidence is one
native-fisheye projection per sparse ray, camera, and FPN level, followed by
RFL value lifting, fisheye Jacobian/tangent geometry, joint camera-level
attention, and spherical state refinement.

## Matched 20-epoch accuracy

| Detector head | mAP | NDS/FDS | mATE down | mASE down | mAOE down |
|---|---:|---:|---:|---:|---:|
| corrected Sparse4D C0 | 0.5644 | 0.6342 | 0.4623 | 0.0858 | 0.3402 |
| Fisheye Sparse-Ray Detector V2 | 0.5288 | 0.5913 | 0.5216 | 0.1448 | 0.3723 |
| V2 minus C0 | -0.0356 | -0.0429 | +0.0593 | +0.0590 | +0.0321 |

| Class AP | C0 | V2 | Delta |
|---|---:|---:|---:|
| Car | 0.6051 | 0.5856 | -0.0195 |
| Van | 0.5682 | 0.5309 | -0.0373 |
| Truck | 0.5465 | 0.5019 | -0.0446 |
| Bus | 0.5753 | 0.4709 | -0.1044 |
| Pedestrian | 0.5129 | 0.5088 | -0.0041 |
| Cyclist | 0.5786 | 0.5747 | -0.0039 |

V2's best mAP is 0.5307 at epoch 16. Its best NDS is the final epoch-20
value, 0.5913. V2 learns much more slowly than C0 during the first eight
epochs and then approaches a lower plateau. The dominant regression is on
large vehicles and metric size/translation, while Pedestrian and Cyclist AP
nearly match C0. The first research target is therefore metric centre/extent
evidence for large objects, not reintroducing all 13 box keypoints.

The authoritative machine-readable result is:

`/home/airlab/user/kdh/ROSA/results/raw_metrics/experiments/fisheye_ray_decoder_v2_corrected_raw_uniform_fresh_e0_to_e20_seed3407/final_metrics_vs_c0.json`

## Efficiency

| Complete detector path | C0 | V2 | V2/C0 |
|---|---:|---:|---:|
| deterministic train, batch 4 | 267.00 ms | 361.25 ms | 1.353x |
| train peak allocation | 8,692 MiB | 9,872 MiB | 1.136x |
| compiled inference, batch 1 | 11.87 ms | 9.92 ms | 0.835x |
| compiled inference peak allocation | 741 MiB | 699 MiB | 0.943x |
| parameters | 29.67 M | 18.58 M | 0.626x |

The compiled deployment path is 16.5% faster and uses 5.7% less peak
allocation. Deterministic training is 35.3% slower and uses 13.6% more peak
allocation, so the training and deployment conclusions must remain separate.

Benchmark records:

- `results/raw_metrics/benchmarks/fisheye_ray_decoder_v2_deterministic_train_post20e_exact.json`
- `results/raw_metrics/benchmarks/fisheye_ray_decoder_v2_compiled_inference_post20e_exact.json`

## Verification and sealed artifacts

- Focused method/integration suite: 26 passed.
- Broad regression suite: 998 passed; only the historical near-native harness
  with an unavailable sealed external path was excluded.
- Two-GPU preflight: 200 consecutive real-data optimizer iterations passed.
- Final source snapshot SHA256:
  `1c36a3eb214bc23eacee46108e58edf4d9e9dd15e71758443f7b5fc027b83d65`.
- Epoch-20 checkpoint SHA256:
  `7c88cd3d2a5264c7a7999d6cf46027eb257c4ad966ee7089b46aa4ab53e282c8`.
- Best-mAP epoch-16 checkpoint SHA256:
  `38e9810c08476a87fdda539237f2599a17dafc4c69d1c800d6d232d73d76de31`.
- Best-NDS epoch-20 checkpoint SHA256:
  `9d9ff2b29bf929dd0aaea84b1960626c70e18aae882212785bbcc2a897b7fe3e`.
