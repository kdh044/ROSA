# Invalid experiment: bottom-z mismatch

- Status: stopped during epoch 3; do not resume or report its checkpoints.
- Started: 2026-07-31 18:03 UTC
- Stopped: 2026-07-31 18:23 UTC
- Source commit: `062d236`
- Retained checkpoints: epochs 1 and 2, for provenance only.

Main126 annotations enter the detector state at their gravity center. The
dataset constructs `LiDARInstance3DBoxes` from center-origin input and the
`Fisheye3DODGravityCenterSparse4D` wrapper explicitly writes
`boxes.gravity_center` into the regression target. The matching anchor asset
also stores center z.

This run incorrectly set `anchor_z_is_bottom=True`, shifting every native and
ROSA sampling keypoint upward by half the predicted object height. The run was
terminated as soon as the mismatch was independently verified. No validation
metric from this directory is admissible.
