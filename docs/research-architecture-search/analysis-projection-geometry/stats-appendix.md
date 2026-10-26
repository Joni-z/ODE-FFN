# Statistical and mathematical scope

Training is the existing seed0 experiment. This analysis runs no optimizer
updates and uses four frozen models, three snapshots,4096 uniform-time
calibration points, five512-point fixed-time panels and128 response anchors
per population. Layers and anchors are diagnostic units, not independent
training replications. Tables give exact descriptive medians/ranges; no
significance tests or across-run confidence intervals are inferred.

The teacher on bank-derived inputs is the analytic *population* velocity.
It is a counterfactual target and differs from the empirical-bank
conditional mean. Saved sampled-pair gradients remain the actual
unclipped pair-label gradients. Neither is an Adam update.

G is the uncentered feature second moment. Numerical positive eigenvalues
exceed max(1e-12,1e-10*lambda_max); raw matrices are retained. The first
fixed-time input rank bound is input_dim+2. Uniform-time pooling and deeper
nonlinear features need not obey that bound. Response moments average
W^TW and are not complete Gauss--Newton operators: W and h correlate.

The post-hoc standardized-gradient quantity is defined in analysis-report.md.
Its lowest-quarter share uses the same positive eigensystem as the raw
quantity. Division by small eigenvalues can amplify sample error;4096 inputs
estimate G while128 estimate gradients. No claim is made about a population
tail or a convergence rate. Parameter-gradient fractions are coordinate
dependent and must not be interpreted as invariant capacity allocations.

Full-network gradient reconstruction and finite-difference tests validate
the local derivative algebra, not whether a new parameterization will help
training. Source/data/checkpoint/array digests are checked before analysis;
actual nonpersistent runtime buffers are restored for cross-platform replay.
