# Units, comparison scope and numerical checks

The cohort is four frozen interventions of two existing image models, not
four new trained repetitions. All twelve local durations are reused from
cycle 19. The nine other endpoints are reused under their original runtime
seals. The primary comparison is exp versus folded polynomial for each
existing model. Folded versus literal constant-clock prediction is the
arithmetic-order control. All three contrasts per model are retained.

Every endpoint uses the same 1,024 validation image/noise/time pairs and 1,024
Gaussian sampling starts. There are five fixed panels of 64 examples per
model. Images, pixels, panels, matrices and checkpoints are not independent
trained repetitions. No training-variance test, p-value or population FID
ordering is claimed. Prediction/sample RMS values compare paired arrays,
not unpaired distribution distances. No new trained method is selected.

All FIDs use Heun128, 1,024 generated/2,048 reference counts, the same pinned
Inception weights/MPS feature path and uint8 conversion. Small-sample bias
and shared validation exposure remain. No best mesh, checkpoint, duration,
matrix scale, subset, feature extractor or polynomial order is selected.
Raw integration states are not clipped; image conversion clips to [-1,1].
Raw out-of-range fractions are computed from each actual 128-step bank.

Each new local map is constructed in float64 then cast once to float32.
The exp and folded variants use identical single-matrix application code.
The mathematical reference is a matrix exponential, evaluated numerically;
it is not exact real arithmetic. Independent SciPy expm checks retain the
registered 1e-11 relative/absolute tolerance. Cached float32 maps must match
their float64-to-float32 casts bitwise. SciPy and Torch float32 rounding are
not claimed to be bitwise identical.

The 48 operator records span all six P matrices in both models at
0/1000/2500/5000 updates. All use final calibrated c retrospectively. Their
spectra describe recorded matrices, not performance uncertainty or the
original time-varying clock history. Semigroup defects concern the frozen
linear map, not the whole nonlinear FFN or predictor. Norm bounds and
eigenvalues do not certify task quality or full-model contraction.

Preflight physically replays native and literal constant-clock validation
predictions, and verifies ten promoted whole-network polynomial witnesses.
Separate numerical and analytic fixtures cover zero, negative diagonal and
nonnormal nilpotent matrices. The initial preflight failed before model
outcomes because its interpreter lacked einops. The preserved v1 source and
failure are not overwritten. V2 separates existing image and SciPy-only
environments without adding packages or changing scientific tolerances.

Post-generation verification must replay every new prediction, all 20 new
time panels and 4,096 generated images using the cached maps and a separate
Heun expression. Matrix construction/history is independently checked with
NumPy/SciPy and the original checkpoint tensors. Fresh feature verification
must reproduce 4,096 generated and 2,048 reference rows bitwise and independently
recompute all thirteen FIDs using gesvd within the unchanged 1e-8 limit. The
final analysis records actual terminal audit results; this appendix states
the verification scope rather than substituting for those receipts.
