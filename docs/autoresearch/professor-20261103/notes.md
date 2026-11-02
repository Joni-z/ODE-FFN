# Evidence and decisions

Prior36+30 trajectories are complete and replayed. Muon produces higher-rank updates
but larger P; matched-magnitude momentum does not explain all field gains. Selected
matrix decay0.1/0.5 harms fitting. PD/DP/sandwich ordering alone does not beat SwiGLU.
Prior Stage8 tested wider-positive and affine DP atK1/5k/AdamW3e-4, with negative
results (docs/research-architecture-search/clock-range-readout.md). It did not test
PD/K2/Muon or replacing the nonlinear controller with a locally matched affine one.
The new test is an interaction, not a rediscovery of affine gates.

145 GPU5 is now occupied.3November fresh snapshot shows GPU0/1/2/3 free; choose GPU0
UUID GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 after another check before launch.
Lenovo GPU0 remains desktop-only415MiB. No environment changes are needed.

Additional read-only decomposition registered before execution: archived scalar P
at0/1k/2.5k/5k/10k/15k/20k, both AdamW rates and both Muon rates, Ring8/Product16.
Normalize rows and columns separately and measure resulting spectral norms alongside
row/column magnitudes. P=diag(r)R gives ||P||2<=max(r)||R||2; this is an inequality,
not an exact multiplicative attribution. Checkpoint/config hashes are retained.
This asks whether our P growth resembles magnitude drift discussed by Muown, without
introducing Muown or any new optimizer trick. CPU-only readout; training is unchanged.

Before candidate outcomes, identified a controller-rank confound: full affine can
learn rank64; original32-hidden MLP logit Jacobian has rank<=32. Record this explicitly,
then distinguish using a factorized-linear control if affine suggests a lead.
Do not label a win as evidence that nonlinear gate features are unnecessary.

Cycle2 readout found very large local raw-D hidden Jacobians/term ratios, despite
better Ring8 field fitting. Register a frozen precision case before interpretation:
selected-by-validation raw-D rate (Ring8Muon.02/Product16Muon.006), original scalar
best rate, and compute-matched SwiGLU128 best rate. Same checkpoints, fixed validation
bank, whole-model float32 versus float64 predictions/field MSE and complete vector-field
Jacobians at16 existing probes. This includes time-embedding precision; it is not a
pure FFN-precision isolation or a retrained comparison. No sampling/metric replacement.

Register a native-Muon coordinate diagnostic before execution: final Product16
cycle1 affine-PD/Muon.006 checkpoint; identical held validation pairs; three copied
models with original coordinates, uniform P'=P/3,D'=3D at same LR, and the same
rescaling with LR/3. Transform the P momentum covariantly (M'=3M). Freeze auxiliary
parameters and omit clipping for this isolated P-only step. Measure pre-step exact
function/gradient covariance, native five-NS update directions, conditional DeltaA
norms at common hidden states and post-step predictions. This is not a complete
training step, an architecture improvement, or a new optimizer proposal; no actual
checkpoint/run is modified. CPU float64 forward with native BF16 NS is deliberate.
