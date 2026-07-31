"""C4: raw-CBGS native detector with strict-AP-fraction centerness."""

_base_ = "./rosa_r18_main126_v2_c0_cbgs_native.py"

model = dict(
    head=dict(
        loss_reg=dict(
            centerness_target_mode="strict_ap_fraction",
        ),
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v2_c4_cbgs_native_strict_centerness"

experiment_contract = dict(
    ablation="native-centerness-strict-ap-fraction",
    centerness_target="detached_strict_ap_fraction_xy",
    centerness_thresholds_m=(0.5, 1.0, 2.0, 4.0),
    changed_training_field="model.head.loss_reg.centerness_target_mode",
    denoising_scope="unchanged: DN has classification and box loss but no CNS/YNS",
)
