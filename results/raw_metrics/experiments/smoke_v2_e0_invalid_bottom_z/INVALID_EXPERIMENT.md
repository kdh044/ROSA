# Invalid smoke: bottom-z mismatch

This one-optimizer-step smoke test used source commit `062d236` with
`anchor_z_is_bottom=True`. It demonstrated that the training path executed,
but it does not validate the corrected Main126 gravity-center contract and
must not be used as metric or resume evidence.
