"""D6 ablation: class-balance only the three compact yaw objectives."""

_base_ = "./rosa_r18_main126_v2.py"

# Main126 class order: Car, Van, Truck, Bus, Pedestrian, Cyclist.
model = dict(
    head=dict(
        rosa_cfg=dict(
            profile="balanced",
            yaw_class_loss_weights=(0.765, 1.320, 1.509, 2.041, 1.022, 1.264),
        ),
    ),
)

work_dir = "work_dirs/rosa_r18_main126_v2_d6_yaw_class_weighted"

experiment_contract = dict(
    refinement_profile="balanced",
    ablation="D6-yaw-class-weighted-only",
    yaw_class_loss_weights=(0.765, 1.320, 1.509, 2.041, 1.022, 1.264),
    yaw_class_weight_order=(
        "Car",
        "Van",
        "Truck",
        "Bus",
        "Pedestrian",
        "Cyclist",
    ),
    weighted_objectives=(
        "wrapped_circular_yaw",
        "yaw_keep",
        "yaw_no_harm",
    ),
)
