# Frozen image clocks: does the FID lead need their input variation?

## Question and prior stops

Original image ODE K1/K2 have lower FID than both SwiGLUs on the three tested
meshes, while K2 fails the fixed fitting/order gates. Test whether replacing
all six local input clocks by fixed scalars removes that finite-checkpoint
FID lead. This diagnoses the already fitted image functions. It trains no
model, rescues no stopped setting and nominates no architecture.

Earlier synthetic gate-necessity and vector-clock-collapse results show that
conditioning can matter. They do not answer this image question. Small clock
SD alone does not justify a constant-map approximation. No retrainability,
individual-layer necessity or unique ODE mechanism follows from this test.

## Frozen cohort

Use the original four cycle17 final5,000 checkpoints and runtime buffers.
Image parent `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2`,
seal62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172.
Resolution parent `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18`,
seal0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108.
Reuse all four native Heun128 score/sample/feature records under these hashes.
Do not rerun training or relabel reused controls as new runs. Model seed0only.

Two new intervention cells: ode_k1_constant and ode_k2_constant. For each,
restore its native model and run exactly the first64saved validation inputs
and their saved times through the unchanged native model. For each of the
six ODELayer inputs, compute its native scalar durations and average all
64images x64tokens in float64, then store one float32 scalar. No target,
label, oracle, FID, time bin, fit or alternate calibration is used. This is
an exposed validation calibration, not a fresh holdout or generalization test.

Replace every local clock simultaneously by its one stored constant. Preserve
P, middle SiLU, attention, external time/AdaLN conditioning, normalization,
readout and every native tensor. Compute the literal recursive polynomial
in the original float32 order. All downstream states are recomputed.
This is a complete finite intervention, not an anchored-Jacobian ablation.

## Evaluation and resources

Paired weighted/clean MSE uses all original1,024validationpairs. Keep all five
64-image time panels .1/.3/.5/.7/.9 and the same first eight generated starts
for examples. Generate all1,024original Gaussian starts, in64-image chunks,
uniform Heun128 on[0,1], velocity `(D-x)/max(1-t,.05)`. No state clipping,
new solver, extra grid or best score selection. The finest tested grid is
fixed because cycle18 found a coarse-grid contribution; it is not exact flow.

CPUfloat32, at most two one-thread sampling children,1200s per child. Audit
phase1200s; feature/feature-audit phases600s each. Monitor PID/create-time,
terminal exits and5sCPU/RSS. No automatic retry. Private runtime
`/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle19`; expect<250MBextra.
Use unchanged cycle17 training/metrics spec475428fb/dc729a50and their witnessed
kernels/doc-pass receipts. No package installation or environment doc change.

FID remains1,024generated/2,048validationreferences with the same Inception,
weights, MPSbackend and uint8 conversion. All six endpoints appear together.
Reuse the four sealed native feature banks/scores; extract both new banks.
Report clean/weighted losses, FIDs, mean/covariance terms, raw out-of-range
fractions and each native-to-constant difference. No inferential tests over
images, layers, grids or model repetitions. No additional quality metric.

## Checks and interpretation

Before intervention outcomes: verify all parent seals/sources, native first64
predictions bitwise, six-calibration coverage, range/finiteness and exact
unchanged parameters/buffers. Test literal constant-clock vs folded-matrix
values in float64 on data-free fixtures and promoted whole-network inputs;
full-model outputs must agree at maxabs1e-9. This is algebra validation, not
an architecture-performance result. Preserve the original float32 sampling.

After execution: independent literal-clock forward and manual Heun replay
reproduce every new prediction/sample bitwise; all tensors and source hashes
remain unchanged. Fresh features for both new banks/reference must reproduce
bitwise; independently recompute all six FIDs with SciPygesvd, difference<1e-8.
Four native feature banks retain their cycle18 independent audit and seals.

For each ODE arm, compare its constant-clockFID to BOTHordinary native FIDs.
If it no longer beats the better ordinary control, its frozen input variation
was necessary for that measured lead under this intervention. If its lead
persists, input variation was not necessary to retain that ordering at these
fixed weights/constants. Either sign is local: constants can be a distribution
shift, and a different calibration or a retrained model is untested. Also
report all loss changes and partial gap retention, without a new continuation
gate. Do not attribute the whole lead uniquely to the clocks or to ODE order.

Stop after this one intervention per model. No clock-range, temperature,
calibration, subset, layer, time-dependent-clock, training or solver rescue.
Cycle17 gates and all previous stopped-family decisions remain unchanged.
