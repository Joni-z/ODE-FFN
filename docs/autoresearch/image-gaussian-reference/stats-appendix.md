# Methods and verification

## Comparison unit and metric

The unit is a registered model endpoint evaluated on fixed banks, not an
individual pixel, image, time panel or checkpoint treated as an independent
training repeat. The full table retains 34 endpoints; 16 primary rows include
14 neural 20k endpoints and two train-only statistical fits. All formal model
initializations use seed 0. Descriptive differences and ratios are reported;
no inferential tests, p-values, confidence intervals or multiplicity-adjusted
claims are warranted from these endpoint comparisons. Evaluation rows are
paired by saved inputs, not independent experimental replications.

For error `e = prediction - clean_target` and the saved denominator
`delta = max(1-t, 0.05)`, clean MSE is `mean(e*e)` and weighted MSE is
`mean((e/delta)^2)`, averaged over all coordinates and saved states. Keep the
original 1,024 validation pairs and five 64-pair time panels. FID uses 1,024
generated images, 2,048 unchanged real references and the pinned Inception
extractor/conversion. The finite exposed screen is not an external test of
transfer or a population FID estimate with an uncertainty interval.

The fixed quality gate requires FID at most 0.95 times both Swi86/SiLU131
20k references, both fitting metrics at most 1.05 times the better respective
reference, and strict default SiLU64 FID ordering. The FID cutoff is 176.59322.
Both new references fail every gate. See [decision.json](results/decision.json)
and [all ratios](results/primary-ratios.csv); no threshold is changed.

## Fit and mathematical object

The saved original split has 45,000 training and 5,000 validation indices,
with zero overlap. Fit only training images. Normalize in native Torch
float32 as `pixels/127.5-1`, then promote to float64. Estimate the mean and
covariance by a two-pass batch-512 calculation with MLE denominator 45,000.

PPCA59 stores an orthonormal 3072-by-59 basis, 59 leading covariance
eigenvalues, the mean, and the average discarded variance 0.03595957516486257.
The training covariance trace is 761.8266649563974, of which the retained
directions account for 85.77810347739245%. Isotropic uses the same mean and
trace/3072 variance 0.24799045083216062. No eigenvalue clipping or noise/rank
search is used. Rank 59 and its scalar-count rule were fixed before fitting.

The PPCA representation stores 184,380 float64 values (1,475,040 bytes),
with generic family dimension 182,610 after orthonormality constraints.
Isotropic stores 3,073 float64 values. These are not equal degrees of freedom,
precision bytes, optimization budgets or training costs to the neural models.
This is Gaussian density maximum likelihood, not minimization of the neural
FM objective. Exact conditioning means exact for the fitted Gaussian prior;
it need not be the real-image conditional mean.

Prediction uses the [derived formula](theory.md) in float64 and returns
float32. Sampling remains the original float32 CPU Heun128 with denominator
cap 0.05, batch 64, the original Gaussian starts, no state clipping and no
substitution of exact Gaussian endpoint samples. Raw tail fractions are
preserved in [comparison.csv](results/comparison.csv). Feature input conversion
clips/quantizes only as the original extractor does; raw arrays are retained.

## Independent numerical checks

- Primitive readiness: 21 fixed state/time cases, independent dense and
  Woodbury conditioning, AD derivatives, MLE and physical prior save/load.
  Maximum discrepancy 1.78e-15. The primitive pass recorded one minor
  environment-name mismatch; its original receipt/source remains preserved.
- Full readiness: a new fresh agent executed the complete documented command
  once, actual session 43297, terminal exit 0 and no mismatches. It covered
  actual 64-image/3072D/rank59 native Heun/physical restore, the independent
  sampler, the measured kernel cost, normalization/MLE, actual-shape fake
  covariance fitting, 32 historical feature rows and all 32 old FIDs.
- Real fit: an independent raw-moment calculation in batch 257 checks all
  training inputs, mean/covariance, complete eigenvalue spectrum and residual
  variance. Maximum covariance entry discrepancy is 2.45e-15; PPCA eigen
  residual is below 8.00e-14. No eigenvector-sign comparison is used.
- Predictions and sampling: an independent Woodbury posterior/manual Heun
  checks 1,024 validation pairs, every time panel and 1,024 new endpoints per
  arm. Registered bounds are prediction relative/absolute 3e-6, sample max
  below 3e-5 and RMS below 3e-6. PPCA sample max is 1.1920928955078125e-7,
  RMS 5.632383001991602e-10; isotropic endpoints agree bitwise. Do not call
  the PPCA sample replay bitwise.
- Features: all 2,048 new and 2,048 original reference rows replay bitwise.
  All 34 FIDs independently recompute with SciPy gesvd within 1e-8. No
  covariance square-root approximation or alternate image bank is selected.
- Closure: verify public CSV values, report rounding, all fixed cases/figure
  hashes, source registration, actual data and parent seals. Submitted Report1,
  src/synthetic/configs/scripts, raw source and all prior runs are unchanged.

## Reporting-only PCA decomposition

Added after quality outcomes and visual inspection, using the already fitted
training basis U and mean mu. For each original output, let `q=x-mu`.
Retained energy is `mean(norm(U^T q)^2)/3072`; complement energy is
`mean(norm(q-U U^T q)^2)/3072`. RMS is the square root of each energy.
An independent closure calculation uses total-minus-projection energy and
checks all 17 rows against original arrays, including the original 2,048
real references. Pythagorean reconstruction error is below 3.89e-16 in the
producer. Paired differences use each model's same-start sample minus PPCA.
The PPCA self-difference share is undefined and is left blank, not zero.

The aggregate retained correlation centers each retained coefficient over
the 1,024 paired starts, then uses the Frobenius inner product divided by
the two norms. Floating arithmetic gives 1+9e-16 for the PPCA self-check;
the raw value is preserved. It is not an image-wise similarity score, a
comparison with real paired targets, or a FID decomposition. The basis is
global image PCA, not a measured manifold tangent. Similar marginal energies
do not establish the same distribution or residual dependencies.

The reporting producer's original terminal tool receipt was unavailable
after a context-truncated observation. Its complete files and absent process
were inspected without relaunch. No original exit code is inferred. The
archive's independent original-array verification has its own observed
terminal result. This reporting limitation does not change the six-stage
formal controller's consumed terminal exit 0 (session 35918), nor the core
curator's consumed exit 0 (session 78800).

## Resource and cost scope

The six monitored phases used about 191 seconds total. Peak sampled CPU RSS
was 1,061,486,592 bytes in the fit audit, below 1 GiB. Per-stage timing/RSS is
in [resource-summary.csv](results/resource-summary.csv). Statistics and sample
generation used one CPU thread; the warm MPS extractor handled image features.
No environment installation, GPU training or remote host was used.

PPCA's counted matrix work is 362,496 MACs per image, in float64, for its
complete two-projection predictor, excluding pointwise operations. Isotropic
has no matrix multiplication. Neural costs in the CSV are FFN-only float32
MACs over 64 tokens and three blocks, excluding the common backbone and
pointwise work. These scopes and precisions differ: do not use this table
as an equal-compute or equal-wall-time architecture comparison.

Public copied JSON receipts may use a single final newline; parsed values
equal the raw private originals. Scientific CSVs and figure bytes are copied
unchanged. Original raw receipt bytes, including terminal blank lines, remain
in the private runtime. No Obsidian write-back is selected for this repository.
