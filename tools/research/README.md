# Research Experiment Entrypoints

이 폴더는 연구용 실행 스크립트를 한곳에 모아두는 위치다. 실제 모델 로직을 여기에 구현하지 않는다.

- `projects/mmdet3d_plugin/`: 모델, dataset transform, projection, ray matching module 등 실제 학습/추론 코드
- `configs/`: 각 실험과 ablation 설정
- `tools/research/`: 위 모델과 config를 호출해서 GT 기준 검증, curve dump, ablation 실행을 하는 바깥 entrypoint
- `tools/` 루트: train/test/dump처럼 공통 실행 엔트리

## 현재 스크립트

```text
tools/research/ray_range_gt_check.py
```

이 스크립트는 메인 방법이 아니라 보조 검증용이다.

```text
목적:
GT range 주변 후보를 만들었을 때 sparse keypoint feature score가
정답 range 근처에서 충분히 강한 peak를 만드는지 확인

현재 판단:
center one-point와 sparse keypoint 모두 direct depth/range refinement
메인 claim으로 쓰기에는 약함
```

따라서 앞으로의 메인 방향은 다음이다.

```text
direct depth/range selection: no
range aux only as novelty: no
fisheye ray geometry PE: yes
object-level overlap range auxiliary: yes
projection-quality weighted supervision: yes
visible view/keypoint support weighting: yes
projection-quality reliability: yes
temporal query update weighting: yes
```

메인 ablation은 다음 순서로 둔다.

```text
1. baseline Sparse4D
2. + query-level range aux only
3. + fisheye ray PE
4. + object-level overlap mask
5. + projection-quality loss/view weighting
```

2번만 좋아지면 novelty가 약하다. 3~5번이 같이 좋아져야 4-fisheye overlap을 안전하게 auxiliary supervision으로 바꿨다는 주장을 할 수 있다.

## 모델 코드 위치

```text
projects/mmdet3d_plugin/models/ray_matching.py
projects/mmdet3d_plugin/models/projection.py
projects/mmdet3d_plugin/models/blocks.py
projects/mmdet3d_plugin/models/sparse4d_head.py
```

## Feature 샘플링 shape

실험 스크립트는 모델 내부 shape와 같은 기준을 따른다.

```text
feature_maps[l]          = (B, N_cam, C_l, H_l, W_l)
query/keypoint 3D points = (B, N_obj, N_key, 3)
projection 결과          = (B, N_cam, N_obj, N_key, 2)
sampled feature          = (B, N_cam, N_obj, N_key, C_l)
view_support             = (B, N_cam, N_obj)
query_support            = (B, N_obj)
range_logits             = (B, N_obj, N_bins)
range_aux_weight         = (B, N_obj)
```

Range-search ablation에서는 후보 축 `K`가 추가된다.

```text
후보 keypoints       = (B, N_obj, K, N_key, 3)
후보 projection      = (B, N_cam, N_obj, K, N_key, 2)
후보 reliability     = (B, N_obj, K)
```
