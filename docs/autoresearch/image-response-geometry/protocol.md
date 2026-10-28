# Observed-coordinate response: fixed diagnostic

Zhizhe Zhang · October29,2026. Register before new image derivatives.

## Question
Does Swi86 exhibit materially more negative response than K1 in directions
observed by both models? Such responses violate a necessary property of the
best conditional-mean predictor under Gaussian corruption. A measured
difference would only motivate a later controlled hypothesis, not explain FID
or establish that repairing it improves quality. Old covariance-head,
conservative-readout, spectral, frame and rotation settings stay stopped.

## Correct target and actual input information
For a fixed learned linear input map B, let U span its row space with
orthonormal columns. For Z=tX+(1-t)e and independent standard Gaussian e,
the ideal predictor using that observation is m_B(z,t)=E[X | U^T Z=U^T z].
Its derivative satisfies

```text
J_z m_B = t/(1-t)^2 Cov(X | U^T Z) U U^T
U^T J_z m_B U = t/(1-t)^2 Cov(U^T X | U^T Z), symmetric PSD.
```

This does not require knowing the population prior. The full first matrix
need not be symmetric or PSD. Do not constrain full-space derivatives after
lossy encoding, transfer this property to hidden P, or equate PSD with gain<=1.
The empirical image-bank reference is not used.

Inspect the actual native two-convolution patch encoder, disabled adaptive/
multiscale branches and raw-input routes. Form its learned64-by48 patch map
B=proj2*proj1 in float64. Its numerical rank uses relative SVD cutoff1e-10,
fixed before outcomes. Use its actual row-space basis and the nonoverlapping
4-by4 patch layout. For the K1/Swi86 comparison compute the exact numerical
intersection of these two patch row spaces, again cutoff1e-10. Verify both
projection residuals<=1e-10 and orthogonality<=1e-11. If the intersection is
empty, report that comparison as unavailable rather than replacing its space.

## Fixed models, states and response sketches
Reuse the sealed image-state-risk inputs and exact frozen20k models: native
K1, cold fixed06, matched-init SiLU64, Swi86 and SiLU131. All are included on
paired validation, paired training, K1-generated and Swi86-generated states,
at Heun indices13,38,64,90,115. Use images0and1, never select by response.
This gives200own-observation rows plus80K1/Swi shared-observation rows.

Derivatives use each frozen model's float64 real extension at its stored
float32 input. Native RMSNorm and attention force float32 internally. In the
isolated real extension remove only those casts, retaining literal norm and
attention equations and the time-embedding dtype hook. First require this
literal implementation in float32 to reproduce the native output bitwise.
Then verify promoted/native forwards with rtol1e-4/atol1e-5 and replay
all five original64-image native prefixes bitwise. No attention-backend change.

Use24orthonormal Krylov directions in observation coordinates, one starting
normal vector from Generator(seed0). Products are first-order VJPs only.
For own spaces, repeatedly apply the observed J^T and reorthogonalize twice.
For the shared comparison, use one identical basis for both models: evaluate
both observed J^T actions on every new vector and alternate which action
supplies the next vector. On norm<1e-12breakdown draw the next deterministic
random vector and report it; do not increase the dimension or try another seed.

With basis Q, save M=Q^T U^T J^T UQ, its symmetric/skew parts and singular
values. Record the smallest eigenvalue of sym(M), its unit physical direction,
an explicit VJP Rayleigh value, negative symmetric energy, skew fraction and
orthogonality residual. The smallest Ritz value is an upper bound on a full
minimum symmetric eigenvalue, not an estimate with a claimed global accuracy.
A negative witness proves a violation in that direction numerically; absence
of a witness does not prove PSD. No trace/population-rate inference is allowed.

## Prespecified decision
Use only the shared basis for the main K1/Swi comparison. A premise worth
examining requires, on BOTH generated-state origins: Swi86's mean negative
hinge max(0,-Rayleigh) >=.05, at least3of10Swi86rows below-.05, and K1's mean
hinge <=.75times Swi86. Every time/domain and all ordinary controls remain
reported. These are diagnostic priority rules, not model-acceptance gates.

If this conjunction fails, do not prioritize a PSD/monotonicity constraint
from this study. If it passes, first determine whether ordinary controls share
the contrast and formulate a separate falsifiable architectural hypothesis.
Either result leaves old quality claims and the main successor objective open.
No extra Krylov size, state cohort, threshold or spectral penalty rescue.

## Verification and resources
Preflight: source/data/parent hashes; all five native64-image replays; actual
encoder maps/observed spaces; projection/lift identities and a null-input
perturbation showing observation sufficiency; fixed Gaussian/atomic-prior
conditional-derivative identities; a counterexample to full-space PSD; known
matrix sketch checks; actual nonzero native VJPs and central differences.
A fresh documented agent must run it once without fixes before formal work.

Independent audit checks all280compressed24-by24Jacobian blocks by batched
central forward differences at h=1e-4, and all280Rayleigh directions at BOTH
h=1e-4and2e-4. Fixed rtol3e-5/atol3e-7; also verify stored basis, projection,
eigenvalue, Rayleigh and metric arithmetic. Audit does not reconstruct the
full3072-by3072Jacobian or certify unprobed directions. Check unchanged model
tensors/buffers and formal-file hashes. Consume actual terminal exits before
analysis. Preserve any failed runtime; no automatic retry.

One one-thread CPU worker on Mac mini, no GPU reservation or package change.
Monitor actual CPU/RSS every5seconds and progress each model/group. Measurement
ceiling1800s, audit1800s, expected new storage below300MB. Re-poll a live handle
on observation expiry. All old sources/data/runs and Report1 remain unchanged.
