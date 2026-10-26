# Two exact links to the objective, and what they would permit us to test

Notation: y is clean data, e is independent N(0,I), x=(1-t)e+t y, U=y-e, 0<t<1. These are algebraic consequences of our specified path, not new theorems or empirical findings.

## Conditional labels, field geometry and optimization noise

At a fixed x,t, e=(x-t y)/(1-t), hence

```text
U=(y-x)/(1-t)
v*=E[U|x,t]
Cov(U|x,t)=Cov(y|x,t)/(1-t)^2 = V(x,t).
```

The standard Gaussian posterior derivative identity gives

```text
J_v* = -I/(1-t) + t Cov(y|x,t)/(1-t)^3
     = [-I+t V(x,t)]/(1-t).
```

The covariance identity is established background, discussed in [On the Guidance of Flow Matching](https://arxiv.org/html/2502.02150v3) and [Lipschitz regularity](https://arxiv.org/html/2604.06065v1). It makes the relationship precise without equating the quantities: derivative, conditional label variance and neural optimization curvature are different objects.

For loss `ell=0.5 ||f_theta(x,t)-U||^2`, let H be the output-by-parameter Jacobian `D_theta f`. At a fixed x,t and parameter state,

```text
g = H^T(f-U)
E[g|x,t] = H^T(f-v*)
Cov(g|x,t) = H^T V H.
```

Thus two parameterizations can represent the same f but have different H, induced parameter updates, and label-induced gradient noise. The polynomial derivative `L_A` from the mathematical audit participates in H. This is an actual route connecting the ODELayer parameterization to the FM training objective. It does not show the resulting geometry is better or make AdamW equal to Euclidean gradient flow.

If a future function-matched P-versus-B comparison shows a material optimization difference, an oracle-label control could help explain it. On Gaussian/Ring8, replace the random paired label U by the analytic v* at the **same** sampled x,t. The expected squared-loss minimizer is unchanged, while conditional-label noise is removed. Random input/batch variation remains; oracle training is not noise-free optimization. This would be a mechanism diagnostic restricted to analytic targets, not a proposed practical training recipe or another default experiment. Do not run it before the parameterization question warrants it.

A Gaussian sanity check prevents an inappropriate transfer of empirical-distribution claims. For scalar y~N(0,q), q>0, let c=(1-t)^2+t^2 q. Then

```text
v*(x,t) = [t q-(1-t)] x/c
V(x,t) = q/c
J_v* -> 1 and V -> 1 as t -> 1.
```

So neither divergence of the marginal velocity derivative nor vanishing label variance near clean data is universal for a smooth full-support target. Conditional-path slopes, an empirical finite dataset and a smooth population distribution require different arguments.

## Why a clean-data bridge matters, but is not an ODE advantage by itself

Let D predict the clean-data posterior mean. Its converted velocity and derivative are

```text
v_D=(D-x)/(1-t)
J_vD=(J_D-I)/(1-t).
```

At the same x,t and without endpoint clamping,

```text
||v_D-U||^2 = ||D-y||^2/(1-t)^2.
```

Therefore comparing direct velocity against direct clean-data prediction under one common velocity loss is possible. Comparing unweighted MSE for both would also change time weighting. JiT additionally changes time sampling and clamps the denominator; matching its practical setup requires documenting those choices, including how the reference label is clamped. Our repository uses the same clamped denominator for prediction and label in `src/denoiser.py`. None of these changes should be mixed into the first ODE mechanism experiment.

For a mathematical limiting example, suppose y=Q z with Q having r orthonormal columns in dimension D. The posterior mean stays in span(Q); its derivative has rank at most r. On the perpendicular subspace, the ideal velocity has derivative `-I/(1-t)`. A bottlenecked direct-output network may therefore have a different representational burden depending on the prediction target. This is a limitation of prediction parameterization, shared by ODE and SwiGLU; it is not evidence for ODE-FFN. A singular subspace-supported target also introduces endpoint singularity, so its behavior must not be attributed solely to architecture.

If the hidden-block mechanism survives the simple controls, a useful later bridge is the same known mixture in a larger, fixed orthogonal embedding, with a declared small positive perpendicular variance if smooth endpoints are required. It retains analytic Gaussian-mixture targets, exposes denoising/ambient-direction demands, and avoids adding arbitrary datasets. Compare the same block choices under the same loss and time sampling, reporting errors parallel/perpendicular to the signal subspace. Keep one chosen ambient dimension initially. A positive perpendicular variance removes the exact rank-deficiency claim above; the exact oracle must use that variance.

This is a contingent transfer test. It should follow a reason to retain the block, not serve as a new benchmark engineered to manufacture a favorable result.
