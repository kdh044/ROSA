# Fisheye geometry-aware view confidence ablation.
#
# This enables fisheye-aware view confidence in DeformableFeatureAggregation.
# The geometry relation is computed per object query / camera / sparse keypoint from:
#   - image-bound validity
#   - distance to image boundary
#   - distance to image center
#   - camera-ray incidence angle
#   - relative projected footprint
#   - projected short-axis size
#   - projected aspect/anisotropy
#
# It also adds a lightweight object/query-level ray-range auxiliary loss.
# GT boxes are used only as matched training targets, not as model inputs.

_base_ = ["./sparse4d_kb20_car_ped.py"]

model = dict(
    head=dict(
        ray_range_aux=dict(
            enabled=True,
            # Applied at every decoder layer; 0.1 was too strong and made the
            # query/classification branch unstable during warm-up.
            loss_weight=0.01,
            min_range=0.2,
            max_range=40.0,
            beta=0.1,
            # KB20 pkl stores z as box-bottom; use geometric center for ray range.
            target_z_is_bottom=True,
            # Auxiliary range supervision is only for overlap objects.
            min_visible_views=2,
            min_projection_quality=0.02,
            max_incidence_deg=95.0,
            boundary_margin=0.08,
            boundary_weight=1.0,
            center_weight=0.0,
            detach_geometry=True,
            projection_type="kb_fisheye",
            # Keep auxiliary supervision independent from the fusion support so
            # learned confidence / anisotropy cannot down-weight its own loss.
            use_fusion_support=False,
            multiply_target_quality=False,
        ),
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
                # Keep area as a weak cue; clipped/anisotropic views can have
                # misleadingly large area, so short-axis and aspect carry the
                # stronger distortion-aware prior.
                relative_area_weight=0.25,
                short_axis_weight=0.75,
                # Aspect/anisotropy can be noisy because pose and truncation also
                # change the footprint. Start it nearly off and let the model
                # learn whether to use it in fusion.
                aspect_weight=0.0,
                learnable_quality_weights=True,
                max_learnable_quality_weight=1.0,
                min_quality_component=1.0e-4,
                short_axis_ref=0.04,
                max_aspect_ratio=4.0,
                use_learned_view_confidence=True,
                use_view_feature_confidence=False,
                use_geometry_feature_adapter=False,
                detach_geometry=True,
                learnable_bias_scale=True,
                positive_bias_scale=True,
                bias_scale=1.0,
                learned_confidence_init=0.95,
                learned_confidence_ramp_start=0,
                learned_confidence_ramp_iters=1000,
                min_confidence=1.0e-4,
                min_support=1.0e-4,
            ),
        ),
    ),
)
