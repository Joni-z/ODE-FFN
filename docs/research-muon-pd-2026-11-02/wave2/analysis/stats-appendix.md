# Units and comparisons

The comparison unit is a registered training configuration at the fixed20k
endpoint. There are36 first-wave configurations, all model seed0. All rates and
both ordinary control widths are retained in the complete table. No checkpoint
is selected on test SW2. Field MSE uses validation; independent fixed test
sampling is reported separately.

This bundle reports descriptive fixed-run differences and ratios. It does not
assign confidence intervals or inferential p-values from checkpoints, layers or
projection directions: those are repeated measurements of a run, not independent
training replicates. Means in the spectral figures average six matrices; they
are not uncertainty estimates. No multiplicity-corrected significance claim is
made.

Product16 scalar Muon0.02 versus scalar AdamW0.001: field error decreases28.7%,
test SW2 decreases19.4%. Against SwiGLU192 AdamW0.0003, scalar Muon field error is
52.2% higher and SW2 is11.2% higher. These are arithmetic effect sizes, not
confidence statements. Differences between selection metrics are retained.

All saved-sample checks use the exact test directions reconstructed by the
registered seed and independent double-precision NumPy projections/sorting.
The regenerated sample arrays are also checked pointwise. Field replay uses the
same frozen model on the same device and the already-audited analytic oracle,
with independent float64 accumulation. This validates execution consistency;
it is not a second independent derivation of the population oracle.

The gate saturation fraction uses the archived condition on sigmoid probability
below0.01 or above0.99. The Jacobian summary is the maximum among12 local ODE-layer
probe values. Sampling floors are finite-sample reference comparisons, not exact
zero-error targets. No result is described as an image FID result.
