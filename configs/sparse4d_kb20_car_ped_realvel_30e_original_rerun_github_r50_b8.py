# Rerun the 2-class real-velocity baseline with the restored original code path.

_base_ = ["./sparse4d_kb20_car_ped_realvel_30e_github_r50_b8.py"]

resume = False
work_dir = "work_dirs/sparse4d_kb20_car_ped_realvel_30e_original_rerun_github_r50_b8"
