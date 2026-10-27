# Fixed-readout sampling invariant: registered verification

This read-only study follows cycle07; its aggregate errors/samples have already been viewed. Register this source and protocol before computing the fitted null spaces or inspecting their projected sample identities. It changes no training, accepted endpoint, sampler step count, cap or architecture. The useful-ODE successor objective remains open.

## Cohort and exact prediction

Use all18 final cycle07 models: thin/thick, original professor K2/Swi128/Swi192, clean/shifted/direct. At the actual final LayerNorm and head define `C=I-11^T/64`, `B=W diag(gamma) C`, `c0=W beta+b`. An orthogonal projector P onto ker(B^T) satisfies `N_theta P=c0 P`. The [preceding derivation](../readout-flow-invariant.md) gives the exact projected continuous and Heun dynamics.

Compute32-step Heun scalars R,d before reading any fitted projected samples, so that row-wise `generated P=R*source P+d*c0 P`. Direct gives R=d=1. Both skip heads have the same R, but different bias coefficients d. Compute a second projector onto `ker(B^T) intersect ker(Q^T)` for the target-normal consequence. Use double-precision SVD with rank threshold `1e-10*max(sigma_max,1)` and retain actual ranks and projection residuals; do not select a favorable subspace dimension.

On all five64-anchor cycle07 panels, check projected **raw** outputs against c0; on all2048 saved sources/generated samples check the source-wise Heun identity for both projectors. Compare empirical covariances with R² times the same source covariance, rather than comparing finite samples with identity. Report population normal-space full-W2 lower bounds separately from empirical SW2. These are different metrics.

For every original-ODE endpoint (six models), set only its six P matrices to zero in a local copy. Retain clocks, biases, all other weights/buffers and the fitted readout. Generate with the same sources/Heun32; record the whole-sample displacement and both projected identities. This is a fixed intervention, not a coefficient sweep or a candidate. Do not evaluate/select its FID/SW2 or train it. A displacement below1e-7 RMS makes that individual intervention uninformative; retain it rather than omit it.

Two older Swi128 controls, Ring2D and Product16, check the dimension boundary using their actual saved heads. Their output ranks/null dimensions determine whether this obstruction is present there; do not claim the128D theorem explains older failures automatically. Bind both checkpoint hashes against the prior registered references.

## Verification gates and scope

- Before fitted outcomes: verify centering algebra, rank-deficient and full-row-rank cases, literal unfactored nonlinear decoders and all Heun coefficients in float64. Include gamma=0 and zero bias cases.
- Projector orthogonality/idempotence and B/Q residuals must be<=1e-8; literal fixture source identities<=1e-11.
- Fitted float32 raw-output projection error max<=2e-5; saved and zero-P source-wise identity errors max<=1e-4 per coordinate. Empirical covariance relative Frobenius discrepancy<=1e-4. Tolerances are fixed before the new outcomes and do not change cycle07's scientific thresholds.
- Independently build projectors from an orthonormal centered64D basis and QR algebra, and obtain R,d through literal scalar Heun updates. Match projectors within1e-8. Replay every zero-P generated batch, verify unchanged non-P tensors/buffers, and preserve every prior source/data/checkpoint/seal.

The theorem assumes finite frozen parameters, positive LayerNorm epsilon and a raw predictor locally Lipschitz in state. Its final-normalization bound makes the raw output bounded; the capped skip adds at most linear growth. Hence the finite-horizon flow exists globally and its Gaussian-source projections have finite second moments. Rank/projection and Gaussian-W2 calculations are conditional on that fixed predictor and the specified populations, not an image-generation or ODE-specific performance theorem.

Acceptance means these identities are verified, not that a new method works. An analytic skip is already a generic control. No learned radial skip, wider/time-varying output head or new training is selected. The old2D/16D controls and all earlier stopped settings remain explicit.

Use one single-thread Mac CPU process,600s total execution ceiling, per-model PID/time/RSS status, no GPU/MBP/torch dependency. Save raw projectors/arrays/logs outside Git, publish source/small tables/figures. Keep all failures. Freeze source before outcomes; independently audit before interpretation.
