# Mean, radius and direction in the original normalized backbone

Registered2026-10-26 before new checkpoint traces or counterfactual outcomes. This is a read-only diagnosis. No new training is selected, and earlier stopped parameter/clock/time-adapter settings remain stopped.

## Exact identities and their limits

For one pre-LayerNorm residual state h of width d, center it as c=h-mean(h)*1. Decompose its complete conditioned FFN output f into

```text
f = m*1 + beta*c + q
m = mean(f)
beta = dot(c,f-m*1)/dot(c,c)
dot(q,1)=dot(q,c)=0.
```

At c=0 set beta=0,radial=0,q=f-mean(f). This is a declared finite evaluation boundary; it does not make the projector globally smooth there. Log zero/near-small actual variances rather than assuming them absent. The original LayerNorm epsilon is unchanged throughout.

With the existing residual step a=1/sqrt(3), the exact centered next state is

```text
c_next = (1+a*beta)*c + a*q.
```

The uniform branch component is exactly invisible to all subsequent pre-LayerNorm blocks and the final LayerNorm in exact arithmetic. Removing it from every FFN output preserves the complete function, all input derivatives and common-parameter gradients. This remains true with finite epsilon, affine normalization and time FiLM. Float64 evaluation must verify the identity, not claim bitwise equality after floating-point subtraction.

The radial component is generally not invisible. It changes the radius and the relative angular step; if1+a*beta is negative it can reverse direction. With sigma=sqrt(mean(c²)+epsilon), the exact LayerNorm input derivative is

```text
J_LN = diag(gamma)/sigma * [I-11^T/d-c*c^T/(d*sigma²)]
J_LN*1 = 0
J_LN*c = gamma*c*epsilon/sigma³.
```

Time FiLM adds its own fixed diagonal scale. Radial insensitivity is approximate when variance dominates epsilon, not an exact finite-epsilon invariance. It does not follow that removing radial drift improves prediction.

There is a counterexample inside the actual professor scalar K2 family. Set each second-solve generator P2=lambda*u*u^T, where u=1/sqrt(d)*1. Every power P2^j maps into the uniform direction. The second solve equals its input plus a sample-dependent uniform shift for arbitrary original scalar clocks. Thus the complete pre-LN network function is independent of lambda in exact arithmetic, while sigma_max(P2)=|lambda| and the nonzero eigenvalue is lambda. This is a constructed identifiability counterexample, not a claim that trained P is rankone or that extreme floating-point values are harmless. It blocks a universal inference from individual P spectra to task error.

## Fixed checkpoints and data

Use the completed recurrent-depth source-v2 snapshots,5000steps, on Gaussian2D,Ring8,Product16. Three checkpoint types per task: original scalar ODE K2 native_final, SwiGLU192 native_final and cheaper SwiGLU128 native_deep. The first two share loss and nearly executed matrix cost; the latter is a practical frontier with a different supervision history. Do not treat it as a causal loss-matched comparison.

Use the existing full uniform-time validation field bank and all512 inputs at .1/.3/.5/.7/.9. No new test targets, training labels, sample generation or optimization are part of the base diagnosis. Bind all native weights, runtime buffers, source/data/arrays and analysis code. Native float32 prediction/Jacobian replay must reproduce saved panels; subsequent finite differences are measured against a separately restored native float64 model.

## Fixed interventions

Recompute all downstream states and clocks for each variant:

- native;
- drop_uniform on all three full FFN branches (function-preserving control);
- drop_radial on all three branches;
- drop_tangent on all three branches;
- drop_radial separately at block0,1,2;
- drop_tangent separately at block0,1,2.

That is ten variants per checkpoint. None changes either internal solve, controller, SiLU, learned parameter or residual scale. No coefficient is fitted. Global variants receive full validation-field and five-panel Jacobian evaluation. Single-block variants receive all five panels; they are exploratory localization, not eligible layer-selection candidates. Save all predictions, derivatives and per-input risk changes.

For each global variant retain native/altered hidden traces. Report full versus centered state RMS, update energy in uniform/radial/tangent components, reconstruction/orthogonality defects, beta, radial multiplier and sign reversals, centered radius changes, feature-angle changes and actual epsilon/variance ratios. Use ratios of summed energies and retain individual-input records. These quantities are mechanisms/proxies; their size alone does not show useful or harmful computation.

## Checks and decisions

Actual-model tests must cover the centered-update identity, finite-epsilon LayerNorm derivative, original-K2 spectral counterexample, full native/wrapper identity, drop_uniform full-function/input-derivative invariance, zero-centered handling and finite-difference derivatives of the projected variants. Tests already pass before this run.

On every actual panel, the drop_uniform control must match the native float64 field within absolute/relative1e-10 and full input Jacobian within absolute/relative1e-9. Check the energy decomposition and exact risk polarization independently. If a structural/identity check fails, stop interpretation and investigate it; do not weaken a quality threshold or modify old runs.

Only a global ODE drop_radial intervention may become a candidate for further evaluation, and only if the same setting improves full primary field MSE by10% over every generic native/global-variant comparator on at least one nonlinear task, stays within5% on the other tasks, improves its own parent on that nonlinear task, and has nonlinear full Jacobian error within10% of the best generic value. This nominates a frozen inference modification, not a trained architecture or a causal diagnosis of the original gap.

A qualifying intervention first requires matched Heun32 validation SW2 checks against ordinary projected/native controls, then a separately registered training/compute/fresh-holdout study. No candidate or component-energy pattern alone completes the research goal. If no global ODE intervention qualifies, stop this setting without projector coefficients, epsilon, width, residual-scale, selected-layer or norm sweeps. A useful finding may instead be that observed norm growth is partly invisible, radial drift is useful, or geometry effects are generic. Preserve those outcomes explicitly.

Local execution uses one thread and process/memory telemetry. Lenovo is offline. No GPU is reserved. Submitted Report1 and all immutable training sources remain untouched.
