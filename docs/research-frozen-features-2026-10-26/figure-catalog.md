# Figure interpretation

`figures/head-refit-comparison.png` and `.pdf`

- Purpose: test whether a new linear head on the same learned final features removes most of the original ODE deficit.
- Data: all four frozen direct backbones, native/shared-mean/shared-pair/separate-mean heads; primary contrasts against all three generic references.
- Encoding: points are two-fold means of five-time paired-redraw MSE. Whiskers and brackets span the two fold values, not confidence intervals. The right panel retains the fixed50% majority threshold.
- Observation: ODE and generic risks change little relative to the large native gap; closure stays below1.4% on every fold.
- Decision: a majority head-only accessibility explanation is unsupported in the registered probe. This is not a proof that changing training cannot improve representations or heads.

`figures/all-feature-probes.png` and `.pdf`

- Purpose: expose all48 fixed settings and the two rival site/time explanations.
- Data: all three sites, both temporal scopes and both label definitions for every backbone.
- Encoding: exact pooled MSE in each cell; logarithmic color scale. Separate-time probes use five heads; mean labels are privileged. Native-head results remain in the main figure/table.
- Observation: reading before final normalization or fitting per-time heads does not yield a strong ODE recovery. Affine input probes agree across model families, as expected from their common input function class.
- Decision: inspect nonlinear feature formation/propagation rather than nominate a changed readout from these results. Every cell remains visible; no selected minimum is treated as a new model result.
