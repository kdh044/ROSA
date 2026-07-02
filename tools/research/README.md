# Research Experiment Entrypoints

이 폴더는 연구용 실행 스크립트를 한곳에 모아두는 위치다. 실제 모델 로직을 여기에 구현하지 않는다.

- `projects/mmdet3d_plugin/`: 모델, dataset transform, projection, ray matching module 등 실제 학습/추론 코드
- `configs/`: 각 실험과 ablation 설정
- `tools/research/`: 위 모델과 config를 호출해서 oracle 검증, curve dump, ablation 실행을 하는 바깥 entrypoint
- `tools/` 루트: train/test/dump처럼 공통 실행 엔트리

현재 예정 스크립트:

```text
tools/research/ray_matching_oracle_check.py
```

예정 모델 코드 위치:

```text
projects/mmdet3d_plugin/models/ray_matching.py
projects/mmdet3d_plugin/models/projection.py
projects/mmdet3d_plugin/models/blocks.py
projects/mmdet3d_plugin/models/sparse4d_head.py
```
