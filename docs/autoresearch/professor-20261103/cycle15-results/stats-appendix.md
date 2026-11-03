# Statistical and numerical scope

Six frozen cases: three models times15k/20k, all from model seed0. The primary
common-path source is preselected cycle14bank0,4096particles, native Heun128.
The33time profiles,128steps,48endpoints and256Jacobian probes are dependent
readouts of those fixed models. They are not training replicates. All finite
outcomes and original controls are retained; no inferential p-values are used.

The bank0checkpoint reversal triggered explicitly post-outcome precision
analysis using all eight saved banks. Publish all24paired within-model bank
changes and54pool/omission scores. Full pools contain32768samples; omissions
contain28672and overlap. Their range measures evaluation sensitivity, not a
training confidence interval. The original mean-of-banks endpoint is unchanged.

Sector counts refer to angular Voronoi regions around the known equal-radius,
equal-weight Ring8 means. Their true probability is1/8by symmetry. Low-posterior
points are retained. Report both analytic-weight and empirical-target L1; the
latter can reward cancellation with finite target counts. Component moment
summaries use equal component weights when comparing centroid/covariance error;
source row counts remain available. Transformation-based SW2 diagnostics use
the unchanged formula/directions but are separate distributions, not additive
pieces of the original score.

Common-source endpoint MSE is a particular transport-map comparison to a
numerical oracle path. It is not the infimum over distribution couplings.
Report reference128/256/512precision. The discrete step identity is exact on
saved native float32states with differences recomputed in float64. Analytic
oracle derivatives are separately checked by central differences. Model
Jacobians are source-bound to the native nonzero derivative witness.

The sector-flux response is first order in an oracle/model perturbation under
stated regularity, not a measured causal explanation of every trained field.
Its calibration is the next selected test; all completed controls stay visible.
