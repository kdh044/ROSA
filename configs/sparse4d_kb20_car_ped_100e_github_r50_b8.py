# Sparse4D KB20 car/ped fine-tuning from official Sparse4Dv3 R50 weights.

_base_ = ["./sparse4d_kb20_car_ped.py"]

num_epochs = 100

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    batch_sampler=dict(type="SceneGroupBatchSampler", allow_group_reuse=True),
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=2.0e-4, weight_decay=0.001),
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.5))),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=num_epochs,
        by_epoch=True,
        eta_min_ratio=1e-3,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=1)

load_from = (
    "/home/airlab/user/kdh/sphere_stream_sparse4d/"
    "ckpt/sparse4dv3_r50_kb20_compatible.pth"
)
resume = False
work_dir = "work_dirs/sparse4d_kb20_car_ped_100e_github_r50_b8"
