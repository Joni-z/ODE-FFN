# Statistical and numerical scope

The units are four fixed trained models at three deterministic numerical
meshes. Meshes are repeated computations of those models, not new trained
repetitions. They share all 1,024 starts, 2,048 reference images and Inception
weights. Every generated start is retained; no best checkpoint or mesh is
selected. The official CIFAR test set remains unused.

Report finite-cohort differences only. No p-values, confidence intervals over
training variability, or significance claims follow from the number of images,
pixels, grids or saved checkpoints. [comparison.csv](results/comparison.csv)
and [contrasts.csv](results/contrasts.csv) preserve all registered scores and
contrasts. FID has finite-sample bias and uses an exposed validation reference;
it cannot be compared with FID-50K or an unopened final-test result.

At 128 steps, K2-Swi129 is -8.319871 FID points (-4.333% relative to Swi129),
compared with -10.705332 (-5.515%) at 32. K1-Swi129 changes from -9.192454
to -6.646395. K2-K1 changes from -1.512878 to -1.673476. These are descriptive
paired-model contrasts. Cycle17 weighted/clean regression losses and its
fitting/order gates are unchanged by sampling resolution.

RMS endpoint differences average all 1,024 x 3 x 32 x 32 raw coordinates;
maximum differences use those same coordinates. Values remain in normalized
pixel units, without display clamping. The successive-change ratios 4.917–5.171
describe only these three meshes. They neither estimate an exact-flow error
nor prove a global order. No adaptive reference solver or extrapolated output
is added. At N steps, the sampler makes 2N model calls per image.

The low-rank empirical FID formula is unchanged. The producer uses NumPy SVD;
the independent implementation uses unnormalized cross products and SciPy's
gesvd driver. Maximum score disagreement is 6.54e-13 against the preregistered
1e-8 limit; same-backend feature replay is bitwise. No epsilon, covariance
shrinkage, feature change or numerical tolerance is fitted to outcomes.

The scalar cap and Gaussian-acceleration fixtures in interpretation-fixtures/
are separate analytic checks. They are not additional trained observations,
image FID measurements, or evidence identifying the cause of this ranking.
