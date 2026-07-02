# 3-class ablation: keep Cyclist, but remove real velocity supervision by
# using the zero-velocity pkl split. This isolates velocity labels from the
# class-set change.

_base_ = ["./sparse4d_kb20_car_ped_cyc_100e_github_r50_b8.py"]

data_root = "/home/airlab/user/kdh/incoming/new_dataset/xtreme1_5seq_mftracker/"
ann_train = data_root + "pkl/kb20_seq29_49_train.pkl"
ann_val = data_root + "pkl/kb20_seq29_49_val.pkl"

num_epochs = 30

train_dataloader = dict(dataset=dict(ann_file=ann_train))
val_dataloader = dict(dataset=dict(ann_file=ann_val))
test_dataloader = val_dataloader

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, begin=0, end=500, by_epoch=False),
    dict(type="CosineAnnealingLR", T_max=num_epochs, by_epoch=True, eta_min_ratio=1e-3),
]

train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=1)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_zerovel_30e_github_r50_b8"
