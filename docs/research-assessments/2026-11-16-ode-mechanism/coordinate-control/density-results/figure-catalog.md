# Figure definitions and inspection

All three PNG renderings were visually inspected. Corresponding PDFs were
exported from the same Matplotlib figures. Values retain full precision in statistics.json;
text tables display three decimals.

## terminal-density

- Purpose: compare all five families on each task and frozen field.
- Source: statistics.json summaries; each cell has1,024qualified Gaussian paths.
- Variables: family and estimated mean terminal log-density ratio in nats.
- Bars: within-bank95%bootstrap intervals, conditional on the trained weights.
- Notice: Ring8 estimates overlap; Product16 anchored stages are below the new
  release, with Free ISRU lower in point estimate.
- Interpretation: use the paired figure to assess differences, rather than
  decide a paired result from overlap of marginal intervals.

## paired-coordinate-density

- Purpose: show all eight registered new-minus-reference comparisons.
- Source: statistics.json contrasts; path/bank identities are paired.
- Variables: task, original/projected field, reference and density difference.
- Bars: 99.375% nominal percentile intervals; positive favors the reference.
- Notice: only the two Product16new-minus-anchored intervals exclude zero.
- Interpretation: coordinate choice changes this learned density comparison;
  the new control does not establish a shared-P joint-density advantage.

## product16-density-width

- Purpose: examine a concrete discrepancy behind the Product16density results.
- Source: product-geometry.json plus the preserved reference product-geometry;
  native target calibration is exactly identical. Statistics/geometry density
  means agree within 1e-12 before plotting.
- Variables: projected KL, normalized conditional variance and pairwise
  posterior-sign covariance RMS. KLbars show one source Monte Carlo SE;
  the moment panels are descriptive point estimates.
- Notice: new variance ratio 1.227 versus anchored 1.146; pairwise dependence
  remains near the same target-sample floor.
- Interpretation: widths differ, while these moments alone do not explain
  all joint KL. Neither dimension count nor model count is treated as an
  independent sample size.

The point estimates use the continuous-flow density identity after numerical
qualification. They do not claim an exact finite-Heun-map likelihood. No
image experiment, model selection or held-out time profile is represented.
