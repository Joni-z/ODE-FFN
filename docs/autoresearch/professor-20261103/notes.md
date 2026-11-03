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
# Cycle3 closure and next question

All22 completed and independently replayed from14ab235; generated arrays exact. Original scalar K1 improves field error in8/8 pairings, sampled distance in5/8. RawK1 Ring8Muon.02 reaches.0109129 but SW2.177924 versus matched Swi64.150384; Product16 rawK1.0237107 is worse than Swi64.0187811. The field margin against strongest Ring8 ordinary is only2.8%. Treat this as quadratic-feature evidence, not an accepted ODE method. Frozen Heun32/64/128 retains all six-case rankings; no coarse-solver explanation for the Product16 gap. Complete CSVs/figures in cycle3-results and solver-results. A redundant process poll returned Unknown process id after normal completion; no evaluation was retried.


Table audit correction: Product16 Swi128Adam.0003 has field.0168548645/SW2.0818476662; Swi192Adam.0003 has.0156676032/.0792062049. The cycle3 prose table initially mislabeled the former field value with the latter; corrected directly against the replayed source CSV. Raw endpoints, plots, candidate decisions and validation-selected solver checkpoints were unaffected.

Cycle4 complete18, exact sample replay; no ordinary field win. Constant Product16
Muon.02 MSE.0189632/SW2.0755200 is an ordinary bias-free SiLU FFN, not an ODE
method. Symmetry diagnostic original scores replay within4.8e-10 and identity
closes1.11e-16. Product16 rawK1 even fraction48.9%, Ring46.3%; ordinary controls
also have appreciable even error. Selected next: diag(abs(h)) at K1, holding
conditional singular values matched at common P,h, plus known bilinear64 ordinary
reference with the same projections as native Swi64. Full predictors are not
forced odd; no general image/FM-specific claim.

Historical cross-check: research-architecture-search/next-direction-evidence.md
already separates target-Jacobian legality from approximation, and the
research-conservative-readout-2026-10-26/results/analysis-report.md stops the
known potential readout despite large parent-relative improvements. The earlier
state-clock midpoint setting also completed without nomination. The new PD/raw
nonlinear-flow note specializes an old distinction, not a novel restart. No
previous signed-square/absolute-diagonal/bilinear64 entry was found outside
the current study in the scoped Markdown search. This is not a novelty survey.

Cycle5 negative: absK1 best validation Ring.0123905/Product.0265689, versus raw
.0109129/.0237107. All6 rate-matched field comparisons worsen. Ring even energy
.006027 exceeds raw.005208; Product even energy.011044 only slightly below
raw.011520 while odd error rises.015536 versus.012049. No parity-feature rescue.
Bilinear ProductAdam.0003 has field.0682647/SW2.0705403, motivating terminal
structure inspection rather than assuming either field MSE or random SW2 is
sufficient alone. All14replays and four parity receipts pass. GPUs free.

Sample audit proves neither random SW2 nor latent-axis averaged SW2 certifies
all mixture coordinates. BilinearAdam.0003 mean gap9.06%, exactpopulation.52%;
axis0gap32.45% versus3.04%. Allmodels have visible imperfections. C6 selects
longer matched budgets for retained raw/Swi references using earlier validation
rates; no secondary-metric model selection. Auditor gains explicit endpoint-step
(default20k unchanged) for80k, with configuration/data/checkpoint integrity guards.

Cycle6 closed: all8 final80k trajectories, exact20k numerical prefixes and native
endpoint replays pass; all generated arrays reproduce exactly. Raw K1 retains
a matched-Swi64 field tradeoff but loses sampling and wider ordinary controls.
See docs/autoresearch/professor-20261103/cycle6-results/findings.md. Next: register
a frozen Frobenius-preserving P-shape intervention against matched-operator-norm
scalar contractions; no new training method is accepted.

Cycle7 closed54frozen cases, six exact replay anchors, all finite. Norm/path
checks pass independently. Only the smallest RawK1/Ring20k shape change improves
slightly; it worsens at80k and all stronger flattening paths worsen fitting.
Post-hoc flattening is not a repair. Next bounded question: can the native model
adapt when weak spectrum relaxation is imposed throughout training? Register
one relaxation time with untouched and paired scalar-contraction controls;
no new gate, loss, schedule or rate sweep.


## Cycle10 interpretation fixed before final outcomes

At a common hidden state, radial normalization scales A=P diag(h) uniformly;
it does not flatten its singular-value shape. Its full Jacobian changes by a
rank-one feedback term. Static scaling has the same function class as raw K1
under P reparameterization, but paired stored tensors do not imply paired
initial hidden functions or identical optimizer geometry. These distinctions
prevent crediting a scalar-coordinate gain to new dynamics.

Native validation/test separation was checked directly in the pinned
reference/synthetic/train.py: fixed_eval_data always constructs field inputs
from validation.npz; sampling_split changes only terminal sampling targets,
source draws and directions. The final CUDA audit therefore recomputes the
same validation field and separate test sampling score used in the protocol.
No evaluator changes were made.

The scalar-log reader passed on all six native16-step145 smoke outputs
(36ODE-layer records; ordinary models have no ODE records). Earlier collector
layer-key mismatch was fixed before use. Final evidence still awaits all12
formal endpoints. Two completed145native anchors already match archived
numerical states and samples; the first partial Lenovo receipt had zero ready
anchors and is explicitly not a closure result.
