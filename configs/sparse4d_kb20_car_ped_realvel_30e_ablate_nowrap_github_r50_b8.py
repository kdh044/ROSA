# Ablation 1: prevent per-slot timestamp wrap in SceneGroupBatchSampler.

_base_ = ["./sparse4d_kb20_car_ped_realvel_30e_github_r50_b8.py"]

randomness = dict(seed=1151096205, deterministic=False)

train_dataloader = dict(
    batch_sampler=dict(
        type="SceneGroupBatchSampler",
        allow_group_reuse=True,
        avoid_group_wrap=True,
    )
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_realvel_30e_ablate_nowrap_github_r50_b8"
