# Antithetic risk and newly created odd activation responses

Registered before new x/-x model evaluations, 26 October 2026. The internal-feature probe did not expose a strong single-site affine residual reservoir. The separate algebra note identified a distinction between SiLU and multiplicative SwiGLU, but its bias-free assumptions have not been tested against the learned network states.

## Immutable predictors and common inputs

Primary models: original scalar K2 ODE and direct SwiGLU128/192/384 from the final conservative-readout study. Add the final historical sigmoid-GLU128 as a secondary known activation control. Its five native float64 panels already replay exactly under the immutable model implementation. Its historically matched Swi128 has weights and runtime buffers bitwise equal to current Swi128, so that is an alias, not another independent model or duplicate forward run. Recheck the complete key sets and values before the new study.

Use all4096 saved A population rows at each time(.1,.3,.5,.7,.9) from coordinate_dependence_20261026. For physical x=A*rotation^T evaluate x and its exact negation. The analytic population velocity is odd. Reproduce old positive predictions for the four primary models, the original20 float32 panels, and the GLU's five old float64 panels. No checkpoint or time selection, model training or new target law. This is exposed diagnostic population data, not a fresh final test. Cached training-bank symmetry is not assumed.

## Exact finite risk decomposition

Let O=(f(x)-f(-x))/2 and E=(f(x)+f(-x))/2. The paired risk equals odd fitting error plus even leakage:

```text
T = (||f(x)-v*(x)||²+||f(-x)+v*(x)||²)/2
  = ||O-v*||² + ||E||².
```

At each time, project the odd error O-v* onto the raw linear-input span, without an intercept. An antithetic odd function has zero paired mean; this is a diagnostic linear subspace, not a fitted architecture. Split its risk into linear odd error L and residual nonlinear odd error N. Thus T=L+N+even exactly on the finite panel. Use all16 input coordinates and all outputs, with identical projection operators across models. No target nonlinear basis is supplied.

Also project O and the oracle separately. Record the residual nonlinear signal's relative RMS and its coefficient along the oracle residual: <O_res,v_res>/||v_res||². These are finite-panel descriptive measurements, not proof that a nonlinear feature is absent or a realizable model correction.

Generate256 paired row-bootstrap resamples with generator seed0, preserving all coordinates/times/models of each row. Refit the weighted raw-linear projection in every resample. Weighted moment systems are16-dimensional and use no tuned penalty; direct SVD/QR must confirm the projection. Save weights, projected coefficients, component risks and every bootstrap result.

Primary comparisons are ODE minus each of the three direct Swi references. For positive total excess compute signed even-excess and nonlinear-odd-excess fractions. For each explanation separately: supported as a majority only if all three95% percentile interval lower bounds are at least.50; rejected as a majority only if all three upper bounds are below.50; otherwise mixed/inconclusive. Do not clip negative or above-one fractions. GLU comparisons remain secondary and cannot replace a primary outcome.

## Actual activation measurements

For the original ODE record each controller hidden SiLU and each middle SiLU, nine sites. For SwiGLU record each gate preactivation, value projection and actual product entering its output projection. Record the analogous sigmoid gate/product for GLU. Preserve actual hidden widths; units are averaged within each input row, not treated as independent samples.

For every standard SiLU pair p+=m+d,p-=m-d, measure

```text
activation_odd = (SiLU(p+)-SiLU(p-))/2
base = d/2
R = activation_odd-base
|R| <= .75*|m|*min(|d|,1).
```

Store per-row mean-square energies of m,d,activation_odd,R and the pointwise bound, plus maximum bound violation. Measure eta=R_RMS/base_RMS and zeta=bound_RMS/base_RMS. These need not be fractions below1. The three original middle SiLUs at all five times are the fixed local question: a uniformly small added-odd response requires eta<=.10 in all15 cases; a small-center certificate additionally requires zeta<=.10 in all15. If any eta exceeds.10, the uniformly small-contribution account is rejected. If eta is small but the bound is large, report an uncertified observation, not the certificate. Do not select a different layer or threshold after outcomes.

Also retain the paired even/odd duration moments for all six original solves, to check the separate constant-clock idealization. These are descriptive and cannot replace the fixed middle-SiLU criterion.

For SwiGLU the product's zero-even-center reference at the observed odd projections du,dv is .5*du*dv*tanh(du/2); for sigmoid-GLU it is dv/2. Record its energy, the actual product-odd energy and their cross term. This is local algebra, not a downstream intervention, causal contribution or measured benefit from removing learned centers. Nonlinear dependence already present in d must not be credited entirely to the activation.

## Verification and resources

Before outcomes verify the parity/risk identities, affine and nonlinear-odd field witnesses, weighted projector equivalence and the SiLU bound. Actual activation hooks must preserve native predictions and reconstruct actual products/SiLU outputs. Bind the original checkpoints, runtime buffers, configuration, population arrays and source hashes.

An independent manual traversal must reproduce all measured fields and per-row activation statistics. Independently weighted QR projection, including a latent-coordinate rotation check, must reproduce every original component and all256 bootstrap results/classifications. No symmetrized field is nominated as a model: its idealized error subtraction alone does not establish a quality/cost advantage.

Use one Torch/BLAS CPU thread on Mac mini, with process/time/memory telemetry and bounded arrays. Save full paired field outputs and per-row sufficient activation statistics, avoiding large duplicate hidden-tensor archives. No GPU allocation, model optimizer steps or new training. Lenovo remains offline; local progress is possible. Either outcome needs separate design evidence before any architecture experiment.
