# 3-class 30epoch B-cleanrel stable + Fisheye view relation attention.
#
# This is the first 2+3 experiment: camera-specific object features are formed
# first, then a lightweight relation attention exchanges information between
# visible fisheye views before summing them back into the Sparse4D query.

_base_ = ["./sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_stable_viewfeat.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            geometry_attention_cfg=dict(
                use_view_relation_attention=True,
                view_relation_num_heads=4,
                view_relation_attn_drop=0.05,
                view_relation_use_support_bias=True,
                view_relation_support_bias_scale=0.5,
                view_relation_gate_bias=-2.0,
            ),
        ),
    ),
)

work_dir = "work_dirs/sparse4d_kb20_car_ped_cyc_ablation_30e_support_no_center_angle_aspect_cleanrel_stable_relattn"
