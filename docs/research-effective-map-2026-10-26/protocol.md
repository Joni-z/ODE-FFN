# Paired matrix-coordinate experiment

Fixed before new outcomes, 26 October 2026. This is a local-function diagnostic, not a generative-method nomination. Project owner: Zhizhe Zhang; processing stays local on synthetic-model artifacts.

## Evidence and question
The [preceding diagnostic](../research-data-curvature-2026-10-26/analysis-report.md) finds target-dependent clock activity and a fitting deficit remaining in the learned SiLU core. Clamping SwiGLU-target students' clocks globally to 1 retains most of that deficit. A specific P differential can be ill-conditioned, but the screen does not identify a universal spectral cause.

Can direct effective-matrix coordinates learn a known representable function substantially better than polynomial P coordinates when the two learners start from exactly the same function? A separate unchanged SwiGLU target tests whether conclusions depend on representability being known.

## Two matched cores
At each of the three positions take the two P matrices from the historical cold original-ODE constructor, verified against its step-0 checkpoint. Define

```text
Q(P)=I+P+(1/2)P²
P learner: F(h)=Q(P2) SiLU(Q(P1)h)
W learner: G(h)=W2 SiLU(W1h), initialized Wi=Q(Pi)
```

Both learn exactly two 64-by-64 matrices, 8192 parameters, with no bias, clock, extra activation or normalization. The P implementation forms its effective Q matrices, then uses the same two linear operations as the W learner. This permits bitwise paired initial float32 functions. Separately verify equivalence to the original sequential constant-duration K2 solve in float64. The experiment concerns the unit-duration core, not the unchanged original adaptive-clock model.

Initialization changes no trained-teacher weights. The W learner's initial matrices are a deterministic transformation of the same cold P tensors. Training comparisons intentionally differ in parameter coordinates and their optimizer geometry. They need not have equal parameter gradients or equal function updates after optimization starts.

## Six targets and references
For each position use:

1. **Fixed-flow target.** Apply the same folded-Q P core to the already frozen original-ODE teacher's two projection matrices. Targets are evaluated in the defined folded convention, not silently substituted from a different floating-point evaluation order. The P teacher matrices and their effective W matrices provide explicit zero-risk representers in both learner classes. Verify this separately without initializing a cold student from them.
2. **SwiGLU128 target.** The existing unchanged local SwiGLU teacher on the same common inputs. Representability in either fixed-width core is unknown. Its native teacher is an exact representer for the SwiGLU control.

Include the unchanged native SwiGLU128 student at every target/position. Train twelve new P/W cells and three new SwiGLU cells on the fixed-flow targets. The three SwiGLU-on-SwiGLU cells may be reused from the sealed local-FFN study only if constructor tensors/buffers, optimizer, data/target bytes, all consumed index/input/label digests and saved settings match. Label those rows historical, not new runs. If equivalence fails, stop and document it before deciding on a fresh control; do not silently reuse.

## Inputs and budget
Reuse exactly the previous common feature inputs: 100000 training Z rows per position, all 20000 validation rows and all five 512-row time panels. Prefix features remain fixed. All new cells sample the identical 512-index sequence with a dedicated generator seeded 0. No fresh test set is opened.

Use 5000 updates, AdamW learning rate 3e-4/default moments, weight decay 0, global gradient clip 1 and no schedule. The update function is the unchanged prior local-study implementation. No parameter-group scaling, different clipping, teacher strength, width, extra supervision or post-result continuation is allowed.

## Outcomes and interpretation
Primary: empirical validation MSE, normalized by target energy, for the complete six-target-by-three-student table. Secondary: five time panels, local input-Jacobian error on the first 128 validation inputs, every saved learning curve, gradient/clipping windows, all learned P/effective-W spectra, and determinants/origin-Jacobian signs.

A substantial known-representer coordinate difficulty is operationalized as: at least two fixed-flow targets have P relative MSE above 1%, W at or below 0.25%, and P at least four times W. Report all continuous values even when this pattern fails. This is a descriptive criterion, not a statistical test or architecture acceptance gate. If it appears, absence of the target from the P class cannot explain that control's residual. It does not identify the optimizer mechanism.

If both arms learn the fixed-flow targets but P loses on SwiGLU targets, class restriction and target-dependent optimization remain confounded. If both struggle on SwiGLU targets relative to the ordinary control, the shared nonlinear core is still a plausible limitation. A W advantage alone does not constitute an ODE method.

The map Q has nonnegative determinant for every real P. The direct W core can have a negative determinant for its Jacobian at zero. Record whether learned matrices/functions cross this boundary, using signed log determinants and singular values; do not treat origin behavior as an on-distribution lower bound. Initial functions are paired and originate in the P class. A direct W learner can leave that matrix-image class as it trains.

## Exact checks and costs
Verify the adjoint identity

```text
grad_P loss = G + (1/2)(P^T G + G P^T), G=grad_Q loss
```

against native autograd at cold and teacher matrices, along with input Jacobians and complete parameter-gradient checks in float64. Verify all exact representers, paired initial values/input derivatives, nonzero post-update effective-map differences, and three-step versus two-plus-resume histories for all fifteen new cells. Save all optimizer/global/sampling RNG and actual consumed digests at steps 0/1000/2500/5000; metrics every 500 and status every 100.

Count actual matrix work at batch 512, including forming Q and backpropagating through P². The P model's effective matrices can be cached after training for constant-clock inference; report raw formation cost separately from folded inference. Parameter equality does not mean identical training work. No latency claim is made from these counts.

Run at most two one-thread CPU children with five-second actual CPU/RSS monitoring. Freeze source/data/protocol bindings, launch once and preserve failures. Final independent audit must replay cached labels, all new and reused checkpoints/streams, native outputs and Jacobians, P/effective-W spectra, gradient identities and control provenance. No restart on an observation timeout.

## Workflow
Planning-with-files, hypothesis-generation, experimental-design, daily-coding and results-analysis guide the work. Procedural provenance is [Kassis, Agarwal, He, Patel and Brueckner (2026), Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065), with metadata checked in the preceding primary-source review. It is not evidence for the model hypothesis.
