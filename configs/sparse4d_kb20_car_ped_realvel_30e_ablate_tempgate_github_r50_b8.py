# Ablation 2: keep sampler unchanged, but gate temp_gnn output when InstanceBank
# marks temporal memory invalid by timestamp gap.

_base_ = ["./sparse4d_kb20_car_ped_realvel_30e_github_r50_b8.py"]

randomness = dict(seed=1151096205, deterministic=False)

model = dict(head=dict(gate_invalid_temp_gnn=True))

work_dir = "work_dirs/sparse4d_kb20_car_ped_realvel_30e_ablate_tempgate_github_r50_b8"
