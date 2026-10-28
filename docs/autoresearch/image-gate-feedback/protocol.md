# Native gate feedback: fixed checkpoint and state analysis

Owner: Zhizhe Zhang. Register before fitted-state measurements. The previous finite clock study is complete, published and sealed; no new frozen-clock replacement is selected.

## Question and mathematical object

At 20k, mean-clock collapse removes the native image lead while changing aggregate regression relatively little. It changes clock values and derivatives together. Measure the native response without changing either: separate finite between-time/image/token variation, then compare complete local/FFN derivatives with their pointwise frozen-value algebraic references. The [derivation package](../../../DERIVATION_PACKAGE.md) fixes the equations and their limits.

This is a read-only diagnostic, not a new quality experiment. It cannot causally separate time and image information, decompose FID, certify the complete denoiser or select a new model from a Jacobian statistic. Smaller singular values, invertibility or contraction are not automatic quality goals. No new training, data coupling for quality evaluation, loss, calibration, time-only-clock replacement or hyperparameter sweep.

## Exactly eight native cells

Families: original K1, original K2, Swi86 and SiLU131; actual checkpoints at 5k and 20k. K1 is the practical image lead, K2 the order context, Swi86 the equal-FFN-work/near-parameter control, SiLU131 the near-parameter ordinary-core control. No new widths or selected checkpoints. Model seed remains0.

Restore model and nonpersistent buffers using frozen budget/native constructors. Budget parent seal `ac381ff444384e052c4cc48e9f11550f00ab410a233154972177e97fe543a48b`; completed 20k clock parent seal `825980fab2105dce825380ddfa0c349b5a1f179cb68dd7e44bcd5510916a6f9b`. Bind the original native17, resolution18 and ordinary20 parent seals as well.

## Fixed inputs, times and capture rules

Use the first64original Gaussian sampling starts and the first64clean targets in the original validation bank. Their bytes are immutable. Define paired probes by float32 linear interpolation of these fixed arrays. This derived diagnostic bank is not substituted for any training, validation or FID bank.

Query the exact native Heun128 indices13,38,64,90,115, at times13/128,38/128,64/128,90/128,115/128. Two domains per cell:

- `paired`: the fixed linear interpolant at that exact external time;
- `generated`: the native predictor-stage state at the start of that original Heun step, not its Euler trial/corrector state.

Reuse the entire canonical64-image batch. All128steps and arithmetic remain unchanged; extra paired queries must not alter the generation path. Every final64-image bank must reproduce its already-published native prefix bitwise: budget20k for every family, resolution18 at5k for K1/K2/Swi86, ordinary20 at5k for SiLU131.

Capture actual FFN inputs/outputs and both inner ODELayer inputs where applicable. Record all64×64native scalar clocks per local layer/time/domain for variance. Jacobians use every image at the fixed token positions0,21,42,63:256states per module/time/domain. No state is chosen by error, gradient, norm, image quality or later outcome. Ordinary controls have full FFN derivatives; their clock-variance fields are not applicable rather than zero.

## Measurements and interpretation fixed before outcomes

For each native ODE clock/domain, use the balanced five-time array `s[t,image,token]`. Report population variance as the exact sum of between-time mean variance, within-time between-image mean variance and within-image token variance. Center each clock separately. Preserve all clock-level values; any pooled plot weights components by their raw variance sums, without introducing between-layer mean variation. Fractions describe the fixed cohort, not information importance or model uncertainty.

Compute the float64 real extension at captured native float32 states. Record native/promoted forward differences. For every ODELayer probe retain base/full singular values, rank-one correction norm, clock gradient norm, P norm and the determinant multiplier when the base is numerically invertible. For all families retain complete FFN singular values. For ODE FFNs also retain the pointwise frozen-value reference and feedback-difference norm/rank residual. This reference changes derivatives algebraically, not forward outputs, and is not a sampled variant.

Report continuous medians,95thpercentiles, extremes and explicitly counted numerical-rank exclusions for the fixed cohort; no p-values, best-layer selection or new architecture-acceptance threshold. Pooling must be defined. Checkpoint/domain differences do not identify a causal training effect. Native state coordinates and residual/time modulation differ across models, so ordinary-core spectra are context rather than a whole-predictor stability ranking.

## Verification and resources

Before formal measurement, test the complete local state/P differentials, bounded-clock counterexample, FFN rank-at-most-two identity and Heun chain rule against float64 automatic differentiation. Verify literal native-core forwards, actual checkpoint steps, all parent/source hashes, derived input bytes and unchanged parameters/buffers. Fresh documented preflight must replay the eight canonical64-image endpoints bitwise before launch; preserve failure and stop without automatic retry.

Independent audit must regenerate native state/clock captures from the unchanged paths, verify all endpoints and inputs, and check analytic Jacobians with an independent AD expression on prespecified states. Numerical exclusions and cross-precision errors are reported, not silently discarded. No tolerance or quality bank changes to pass an audit.

Mac mini CPU, at most two one-thread workers; no GPU reservation and no package installation. Reuse image-training@475428fb; matrix-only checks/plots use the existing check interpreter without constructing image models there. Per-cell ceiling900seconds, full analysis ceiling4,000seconds, independent audit ceiling2,000seconds. Monitor actual PID/create-time and primed five-second CPU/RSS. Estimated additional storage below300MB. Runtime `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gate-feedback`.

Stop after this fixed cohort and its audit. Retain all native quality results under their old seals; no new FID is calculated or used for selection. Any proposed constraint or architecture requires its own task-relevant hypothesis and controlled quality experiment later.
