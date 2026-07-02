# Sparse4D KB20 car/ped fine-tuning for 100 epochs.
# Loads the previous Sparse4D checkpoint, but starts epoch/lr schedule from 0.

_base_ = ["./sparse4d_kb20_car_ped.py"]

num_epochs = 100

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
    "work_dirs/sparse4d_kb20_car_ped_fp32_lr1e4/"
    "best_fisheye_bev_mAP_epoch_24.pth"
)
resume = False
work_dir = "work_dirs/sparse4d_kb20_car_ped_fp32_lr1e4_100e_finetune_epoch24"
