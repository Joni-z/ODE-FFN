# Comparison units and audit scope

The user's fixed protocol is training seed0. Comparisons are deterministic descriptive endpoint differences; no significance test, between-training confidence interval or error bar is computed. Five FM times and512 anchors per time are repeated evaluations of each checkpoint, not independent training replications. Checkpoints on a learning curve also are not replications. Relative percentages in the readout are calculated from the full stored values, before rounding.

Primary endpoint: random-time validation oracle MSE. Supporting metrics: common-input velocity MSE, mean squared Frobenius Jacobian error, validation sampling SW2 and recorded costs. Small and full SwiGLU are both retained. The normalization in the endpoint plot uses the smaller oracle MSE of the two references separately on each dataset; it does not remove the other reference.

Remote audits independently verify source/data hashes, all90 checkpoint identities/configuration hashes, all75 saved oracle arrays, recorded initial-tensor pairing and the final validation oracle MSE. Source/weight replay of the saved common fields is an additional check, documented separately when complete. The saved sampling metric is not presented as an independently regenerated trajectory.

For the scalar response diagnostic, floor/error ratios are ratios of summed squared residuals across the same512 anchors, not averages of per-anchor ratios. The exact projector uses the measured response matrix with the documented numerical rank threshold. Its correction-span invariant has maximum element discrepancy below7e-15.
