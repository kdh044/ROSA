# Geometry-conditioned feature adapter ablation.
#
# The main viewconf config can keep the CUDA deformable aggregation path.
# This ablation disables the fused CUDA op because the adapter needs the
# per-camera sampled view features before aggregation.

_base_ = ["./sparse4d_kb20_car_ped_viewconf.py"]

model = dict(
    use_deformable_func=False,
    head=dict(
        deformable_model=dict(
            use_deformable_func=False,
            geometry_attention_cfg=dict(
                use_view_feature_confidence=True,
                use_geometry_feature_adapter=True,
            ),
        ),
    ),
)
