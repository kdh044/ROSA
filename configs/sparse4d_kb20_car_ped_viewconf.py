# Fisheye geometry-aware view confidence ablation.
#
# This keeps the Sparse4D architecture unchanged except for adding a
# geometry-derived bias to DeformableFeatureAggregation attention logits.
# The bias is computed per object query / camera / sparse keypoint from:
#   - image-bound validity
#   - distance to image boundary
#   - distance to image center
#   - camera-ray incidence angle

_base_ = ["./sparse4d_kb20_car_ped.py"]

model = dict(
    head=dict(
        deformable_model=dict(
            use_fisheye_geometry_attention=True,
            geometry_attention_cfg=dict(
                # KB20 images were configured with a loose 360-degree mask.
                # The attention prior uses a realistic half-FOV instead.
                max_incidence_deg=95.0,
                boundary_margin=0.08,
                boundary_weight=1.0,
                center_weight=0.5,
                angle_weight=1.0,
                bias_scale=1.0,
                min_confidence=1.0e-4,
            ),
        ),
    ),
)
