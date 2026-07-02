# Sparse4D Memory Fix Code Change Memo

Date: 2026-07-01

## Current State

This is not a clean optional switch yet.

The Sparse4D source code was modified directly. The new memory-fix config has a separate
`work_dir`, but the source-level changes affect other Sparse4D runs that use this repo unless
they are reverted or wrapped with config flags.

Current experiment:

```bash
configs/sparse4d_kb20_car_ped_realvel_30e_memoryfix_github_r50_b8.py
work_dirs/sparse4d_kb20_car_ped_realvel_30e_memoryfix_github_r50_b8
```

4 epoch result so far:

```text
mAP  = 0.0771
NDS  = 0.1718
mAOE = 0.5893
mAVE = 1.0116
Car AP        = 0.1543
Pedestrian AP = 0.0000
```

Old run was much higher at epoch 4 (`mAP ~= 0.3183`), so this patch should be treated as a
debug ablation, not as confirmed improvement.

## Why This Was Changed

The suspected bug was temporal memory contamination.

Sparse4D keeps previous-frame queries/anchors in `InstanceBank`. If a scene changes, or an epoch
starts validation after training, old cached queries can still enter temporal attention unless the
memory is reset or masked correctly.

The intended fix was:

1. reset `InstanceBank` at train/val/test loop boundaries,
2. only use cached temporal query features when the previous sample and current sample are from the
   same scene,
3. skip temporal DN auxiliary loss when there is no valid previous temporal DN state.

## Files Changed

### 1. `sphere_stream_sparse4d/models/instance_bank.py`

Added scene-continuity checking:

```text
_scene_continuity_mask(...)
```

Behavior:

- compares previous `scene_token` and current `scene_token`,
- keeps temporal cache valid only when scene is the same,
- combines this with the existing timestamp gap mask.

Also changed cache confidence carry-over:

- old cached confidence is decayed as before,
- but if a batch lane is invalid because scene changed or timestamp gap is too large, previous
  confidence is zeroed before merging into current confidence.

Main touched areas:

```text
instance_bank.py:85-121
instance_bank.py:230-244
```

### 2. `sphere_stream_sparse4d/models/sparse4d_head.py`

Changed `temp_gnn` behavior.

Before:

- `temp_gnn` output could overwrite current query feature even when `InstanceBank.mask` said the
  temporal cache was invalid.

After:

- if temporal cache is valid, use `temp_gnn` output,
- if temporal cache is invalid for that batch lane, keep the current-frame query feature.

Main touched area:

```text
sparse4d_head.py:318-342
```

Also changed temporal DN handling.

Before:

- temporal DN auxiliary output could be added even when the temporal DN valid mask was missing or
  shape-invalid after memory reset.

After:

- add `temp_dn_*` targets only when `temp_valid_mask` exists and has the same shape as
  `temp_dn_cls_target`,
- otherwise skip only the temporal DN auxiliary part,
- normal detection loss and normal DN loss still run.

Main touched areas:

```text
sparse4d_head.py:436-451
sparse4d_head.py:557-568
```

### 3. `sphere_stream_sparse4d/models/detection3d/target.py`

Added a guard for temporal DN cache.

Behavior:

- if `temporal_valid_mask is None`, clear `dn_metas`,
- if `valid_mask` is missing or has an unexpected size, clear `dn_metas` and skip caching malformed
  temporal DN state.

Main touched areas:

```text
target.py:310-312
target.py:407-409
```

### 4. `sphere_stream_sparse4d/hooks/memory_reset_hook.py`

Added new hook:

```text
Sparse4DMemoryResetHook
```

Behavior:

- unwraps DDP model if needed,
- finds `model.head.instance_bank`,
- calls `instance_bank.reset()` before train, validation, and test epoch loops.

This is for Sparse4D recurrent memory, not StreamPETR memory.

### 5. `sphere_stream_sparse4d/hooks/__init__.py`

Registers the new hook module:

```python
from .memory_reset_hook import Sparse4DMemoryResetHook
```

### 6. `sphere_stream_sparse4d/__init__.py`

Imports hooks so MMEngine registry can find `Sparse4DMemoryResetHook` from config:

```python
from sphere_stream_sparse4d.hooks import *
```

### 7. `configs/sparse4d_robosense_ocam_4cam.py`

Added this globally:

```python
custom_hooks = [dict(type="Sparse4DMemoryResetHook")]
```

Important: because this is in the base config, other configs inheriting this file also get the
memory reset hook.

### 8. `configs/sparse4d_kb20_car_ped_realvel_30e_memoryfix_github_r50_b8.py`

Added a separate experiment config:

```python
_base_ = ["./sparse4d_kb20_car_ped_realvel_30e_github_r50_b8.py"]
work_dir = "work_dirs/sparse4d_kb20_car_ped_realvel_30e_memoryfix_github_r50_b8"
```

This separates output files only. It does not isolate the source-code behavior.

## How To Restore Original Behavior

To go back to the old behavior, these are the main revert points:

1. remove `custom_hooks = [dict(type="Sparse4DMemoryResetHook")]` from
   `configs/sparse4d_robosense_ocam_4cam.py`,
2. remove hook imports from `sphere_stream_sparse4d/__init__.py` and
   `sphere_stream_sparse4d/hooks/__init__.py`,
3. ignore or delete `sphere_stream_sparse4d/hooks/memory_reset_hook.py`,
4. revert the scene-token mask and confidence masking in `models/instance_bank.py`,
5. revert the `temp_gnn` invalid-lane gating and temporal DN skip logic in
   `models/sparse4d_head.py`,
6. revert the temporal DN cache guards in `models/detection3d/target.py`.

Better next step:

- do not keep this as an unconditional source change,
- add config flags such as `use_scene_memory_mask`, `gate_invalid_temp_gnn`, and
  `skip_invalid_temp_dn`,
- then run ablations one by one:

```text
A. original source behavior
B. hook only
C. scene mask only
D. temp_gnn invalid-lane gate only
E. temporal DN guard only
F. all memory fixes
```

This is needed because the current all-in-one memoryfix run has worse AP at epoch 4.
