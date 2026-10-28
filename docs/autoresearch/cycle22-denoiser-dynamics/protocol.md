# Objective-level dynamics check before another architecture

This is a mathematical/numerical study, not new training or another hidden
solver upgrade. The current useful-ODE-successor objective remains open.
Prior cycle21 is complete and sealed, with no process live. Do not reopen
Taylor training, recurrence, spectral constraints or hidden covariance under
a new name. The question is whether proposed dynamics actually describe the
conditional mean supervised by the existing objective.

## Frozen claims

The [derivation](derivation.md) states U0–U8. Check the heat/denoiser PDE in
Gaussian-observation and our FM coordinates, the nonzero material drift of
the ideal denoiser, and the third-moment term in precision homotopy. Include
wrong-prior and invariant-endpoint counterexamples, so a differential identity
is not mistaken for correct denoising or generation quality.

Use formula-derivation to organize the mathematical object and proof-checker
for a fresh full-scope review. Source units U0–U8 are fixed before extraction;
one read-only shard covers all units, with no recursive delegation. Record
proof obligations, coverage, hypotheses, counterexamples and review scope.
Native review is same-family/provisional, not cross-family proof acceptance.
Numerical agreement is not a proof substitute. No candidate training is
selected by reviewer scores or by passing the numerical fixtures.

## Existing laws, fixed probe cohort

Use gaussian_aniso_2d and gmm_ring8_2d from the existing synthetic/v1 data,
and rotated_product_bimodal_16d from product16_v1. Verify their distribution
and validation files against their existing manifests. Use the first eight
validation clean samples, Gaussian fixture noise drawn once with seed0, and
fixed times .1/.3/.5/.7/.9. There are120point/time cases. These probe points
test identities, not empirical risk, training or generated-image quality.
The official test arrays remain unused. Do not change the source laws or
compare these new deterministic fixtures with old training scores.

Compute complete float64 posterior means analytically: finite-component
Gaussian conditioning for the first two laws, and exact latent-factorized
conditioning for the rotated product law. Compute J, spatial Laplacian and
variance/time derivatives by automatic differentiation, retaining every
case. Compare clean means to the separate existing NumPy marginal-velocity
oracle, using D=z+(1-t)u without the sampler cap. Require mean discrepancies
<=1e-9. PDE residuals must be <=1e-8*(1+largest term magnitude) per case;
covariance/Jacobian discrepancies <=1e-9, with symmetric PSD tolerance1e-10.
For each case retain complete derivative arrays and each PDE term, not just
the maximum residual. No random trace estimator or selected direction.

The closed-form one-dimensional controls are not new datasets: X~N(0,1),
the wrong-prior v=2 denoiser, its true invariant endpoint map, and two finite
posterior moment examples in U8. Their equations and numeric values are fixed
before execution. No data generator, optimizer, metric or model is replaced.

## Resources and verification

Run once on one Mac mini CPU thread using the existing check interpreter:
torch2.10.0/numpy2.2.6/scipy1.18.1. No new package, GPU job or remote host.
Register source before target outcomes. Save source/data hashes, exact command,
PID/create-time, terminal exit and CPU/RSS telemetry. Ceiling300s, expected
disk below20MB. Preserve failures; observation expiration is not termination.

After a successful run, independently replay all mean values with NumPy,
closed-form Gaussian/atomic cases and finite differences at h=1e-4/5e-5.
Use all120cases, without selecting a favorable step. Finite differences
validate the saved derivatives within declared truncation-aware tolerances,
not as an exact second AD implementation. Save both step results. Complete
For every derivative array and both step sizes require max absolute difference
<=2e-3*(1+max absolute AD derivative). Finite-difference PDE residuals must
be <=1e-2*(1+largest participating derivative-term magnitude), separately
at both steps. These gates are fixed before execution and are deliberately
looser than the float64 AD identities because second differences have
truncation and subtraction error. Do not choose the better step afterward.
AD and numeric fixtures must pass before curating evidence. No test on
learned image-model PDE residuals, new regularizer, architecture or training
run is selected here.

Write a short evidence-led analysis, a proof audit with provisional semantic
status and a case figure. Preserve submitted Report1 and all sealed parents.
