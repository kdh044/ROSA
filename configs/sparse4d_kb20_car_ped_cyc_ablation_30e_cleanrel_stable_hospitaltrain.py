# 3-class 30epoch B-cleanrel stable rerun with the hospital entrance sequence
# added to the training split. Validation stays on the original KB val split.

_base_ = [
    "./sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_stable.py"
]

data_root = "/home/airlab/user/kdh/new_dataset/kb20_plus_hospital/"
ann_train = data_root + "pkl/kb20_seq29_49_hospital_train_idvel.pkl"
ann_val = data_root + "pkl/kb20_seq29_49_hospital_val_idvel.pkl"

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=ann_train,
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=ann_val,
    ),
)

test_dataloader = val_dataloader

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_cleanrel_stable_hospitaltrain"
