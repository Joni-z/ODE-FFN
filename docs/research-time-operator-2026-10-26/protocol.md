# Time-dependent operator: a bounded checkpoint diagnostic

Registered2026-10-26 before computing the new gradients or interventions. No training or architecture is selected. Previous recurrence, anchored-corrector, spectral, clock-coordinate and activation settings stay stopped.

## Question and rivals

Do the original learned P matrices retain substantially different useful descent directions at different external times, after accounting for finite calibration noise and ordinary-FFN alternatives? Concentrated middle-time error and parameter distances do not establish that claim. The earlier literature/evidence review explicitly left this question unmeasured.

Rivals are: a usable common matrix adjustment suffices; a temporal direction merely uses additional tangent capacity; its fitted direction overfits calibration inputs; the apparent direction depends on noisy pair labels; or the same effect is ordinary conditional computation equally useful to SwiGLU. A favorable diagnostic cannot prove a trained architecture gain, efficient low-rank realization, or necessity of time-varying P.

## Fixed inputs

Use the completed frozen recurrent-depth study: Ring8 and Product16, arms ode_k2_native_final, swiglu192_native_final, swiglu128_native_deep. The latter is a strong practical comparator, not a matched-loss control. All checkpoints are at5000updates. Use all512 saved inputs at each of five times .1/.3/.5/.7/.9; keep the same source, actual nonpersistent buffers and oracle arrays. No test bank, new samples, retraining, generation or model selection.

Two cross-fit orientations use indices0:256 to calibrate and256:512 to evaluate, then swap the roles. These are exposed analysis subsets, not fresh holdouts or independent experimental repetitions. Report both and their pooled mean, never only the favorable orientation.

Native float32 output replay must match all30 saved prediction arrays bitwise before analysis. Then restore a separate float64 model. Its predictions are the reference for finite perturbations; do not compare perturbed double outputs directly with unperturbed float outputs. Bind checkpoints, runtime source, panel inputs, data manifest and validation targets, native arrays, script and protocol hashes.

Calibrate each model with two label definitions separately:

- Population-oracle velocities: a low-variance diagnostic of evaluation risk, not original training labels.
- Pair velocities from the same cached validation targets used to construct each panel. Recover source=(x-t*target)/(1-t), pair=target-source. These are finite validation draws of the same pair-label law, not the conditional mean of the original training bank. Evaluate all interventions against the oracle on the opposite half.

## Parameters and metric

For ODE, vary only the six original P weights. For SwiGLU, vary only w12/w3 weights in its three FFNs. Keep all biases, controllers, surrounding parameters and buffers fixed. State the selected dimensions; these blocks allocate their budgets differently. This is not a coordinate-invariant class comparison.

For each matrix i, fix n_i=||W_i||F from the native checkpoint and define its scaled displacement b_i=delta W_i/n_i. All n_i must be nonzero. Concatenate these coordinates into b. The perturbation budget is the mean over the five times of ||b(t)||². Normalizing per matrix avoids silently allocating every displacement to the largest matrix; it does not establish a unique natural metric.

Let g_k be the calibration-loss gradient in b coordinates at time k; g_k,i=n_i*grad_Wi L_k. Let phi_k be the centered, unit-RMS linear function of the five actual saved time values, so mean(phi)=0 and mean(phi²)=1. Define

```text
g0 = mean_k g_k
g1 = mean_k phi_k*g_k
N  = sqrt(||g0||² + ||g1||²)
```

With fixed rho=.01, compare exactly three displacements:

```text
shared:       b_k = -rho*g0/||g0||
time-linear:  b_k = -rho*(g0+phi_k*g1)/N
time-reverse: b_k = -rho*(g0-phi_k*g1)/N
```

Every nondegenerate displacement has mean_k||b_k||²=rho². Time-reverse uses the identical fitted coefficients and parameter budget but reverses their temporal association. Reset the complete native matrices before each evaluation and recompute the full downstream network, including state-dependent clocks. No linearized output is substituted for this finite intervention. Zero gradient norms produce a declared zero displacement and an indeterminate record, not a division by zero or a new scale.

The first-order equal-time calibration loss changes are

```text
shared:       -rho*||g0||
time-linear:  -rho*N
time-reverse: -rho*(||g0||²-||g1||²)/N.
```

The extra temporal tangent dimension guarantees an apparent calibration advantage at first order. That alone is uninformative about transfer. Record all per-time gradient norms, the full cosine matrix, norm of the pooled gradient, mean gradient norm, and ||g1||/||g0||. A small pooled gradient or negative cosines does not by itself show that time-specialized training will improve the task.

## Verification and outcomes

Verify orthonormal time coordinates, perturbation norms, analytical first-order identities, and central finite-difference derivatives on the actual full models. For derivative verification only, use epsilon=1e-5 multiplying the declared unit temporal direction; require relative2e-4/absolute1e-8 agreement per panel. This is not a second quality setting or a step sweep. If verification fails, report the failure and investigate it before interpreting interventions.

Save each calibration/evaluation prediction and oracle error before and after all three finite changes. Retain every arm, dataset, calibration teacher, fold and time, with exact risk differences. Report whether the time-linear direction improves opposite-half oracle risk relative to both shared and reversed directions, whether the sign survives both folds and both calibration teachers, and how it compares with the generic rivals. Use descriptive results without p-values or treating folds/times as training repetitions. Make numerical tables and gradient/risk figures.

No automatic architecture nomination follows. Only a coherent transfer pattern can motivate a separately specified, cost-controlled time-dependent-P design and its generic counterpart. Weak, inconsistent or generic-only evidence stops this diagnostic question without a rank, time basis, step, temperature or optimizer search. In particular, extra tangent freedom is not an ODE-specific contribution. Submitted Report1 and every frozen experiment remain unchanged.
