# Numerical comparison units and limits

This is a deterministic approximation qualification, not a population
performance comparison. Six fixed trained checkpoints are retained, with80
registered inputs each and six layers; all2880local states are evaluated at
all five resolutions and four nonidentity modes. No points are removed.
Maxima and violation counts are the relevant statistics because every point
must meet the registered gate. Means, p-values and confidence intervals cannot
replace that universal finite-cohort check and are not used for admission.

Bounds are fixed in hidden-surrogate-protocol.md: local1e-9+1e-3times native
refinement norm; output1e-10+1e-3times full-reference response norm; Ferror
<=0.0005at each layer, bank and whole model. Every failing level is retained.
Fuses each method's own reference displacement and is not generation quality.
Eight input banks are repeated inputs at fixed weights, not independent
training replicates. No training-seed inference is made.

The common resolution is selected across every registered model/layer, rather
than by a favorable score. Zero response denominators remain unresolved.
Native Torchand a separate NumPymidpoint implementation must agree within
1e-11absolute/relative. The source DOP853references have their own prior,
unchanged two-tolerance qualification. No test of improved sampling occurs.
