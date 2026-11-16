# Next diagnostic: field error that changes density

The completed [symmetry intervention](sharing-symmetry/analysis-report.md)
rejects a single selected-symmetry explanation of sharing. Ring8 reverses
after projection; Product16 retains a shared-generator lead even though the
released field has lower projected oracle MSE. Stop response-function search.
The next mathematical object should describe distribution change directly.

## Exact identity

Let p_t be the known native FM marginal, v* its oracle velocity, v the model
velocity, rho_t the model-generated density, e=v-v*, and delta=rho-p.
Subtract the two continuity equations, using the same initial Gaussian:

```text
partial_t delta + div(v delta) = -div(p e),     delta_0 = 0.

div(p e) / p = div(e) + e dot grad(log p).
```

The identity is transport calculus, not a new theorem. The research question
is how the learned architectural constraint affects its source term.
This is an exact identity where the densities are smooth, not a first-order
approximation in e. The scalar source term includes both divergence and
the alignment of field error with the oracle score. Average velocity MSE
discards both pieces. If div(p e)=0 everywhere and the transport problem is
unique, the correct p_t also solves the model continuity equation despite
nonzero velocity error.

An isotropic Gaussian example makes the distinction concrete. A rotational
error e=aJx with skew-symmetric J has zero divergence and is orthogonal to
the radial score, so the density source vanishes. A radial error e=ax can
have the same mean squared magnitude while changing the density. This is
a mathematical example, not an explanation already measured in our models.

## Execution plan and current disposition

1. Derive oracle score and velocity divergence from the existing native
   mixture laws. Their log_density and velocity functions already provide
   independent finite-difference references. Keep the native implementation
   unchanged; place differentiable diagnostics in the study adapter.
2. Verify the continuity identity on deterministic Gaussian rotational/radial
   examples and on the actual Ring8/Product16 marginals. Check analytic or
   autograd derivatives against central differences at decreasing steps.
3. Use the same eight frozen model/task cells and a newly bound evaluation
   stream. Measure exact full-field divergence, score-aligned error and
   density-source residual by time and data region, for original/projected
   fields. Trace exactly in dimension 2/16; no stochastic trace estimator.
4. Treat residual norms as a diagnostic, not a guaranteed SW2 bound. Register
   one time-local oracle-replacement intervention only if a common feature
   of the remaining Product16 gap supports it. Include ordinary controls,
   identical sources, two outer resolutions, and retain every prediction.
5. Accept the proposed localization only if the frozen intervention changes
   the paired distribution comparison in the predicted direction. Reject
   it if residual ordering disagrees, the effect disappears under solver
   refinement, or the ordinary controls show the same unexplained behavior.

The source term alone does not determine the terminal error: transport by
v can amplify or cancel it. Earlier local variational transport failed as
a full learned-model explanation; this test must not recycle that claim.
No image architecture or training is promoted from these measurements.

## Derivation package: along-trajectory density ratio

Status: **COHERENT AS STATED**, for the exact continuous flow under the
assumptions below. The estimator is validated on the registered numerical pilot
and completed all-source population comparison. Product16 rejects the proposed
sharing density explanation; Ring8 intervals include zero. The derivation
below is unchanged. [Completed analysis](density-transport/population-analysis.md).

**Target and invariant object.** Follow the log density ratio
L(t)=log(rho_t(X_t)/p_t(X_t)) on a model trajectory X'=v(X,t). This keeps
the object of interest the evolving distribution, rather than substituting
a field-error norm. Define e=v-v*, s=grad(log p), and a=div(e)+e dot s.

**Assumptions.** Positive smooth native Gaussian-mixture marginals; a C1
model field admitting a unique invertible flow on [0,1]; equal initial
densities rho_0=p_0=N(0,I). Integrability is additionally required when
taking expectations. Native LayerNorm has positive epsilon and the tested
activations are smooth. Numerical Heun maps are not automatically exact
continuous-flow maps or globally invertible at an arbitrary step size.

**Derivation map.** Continuity -> material log-density derivative -> oracle
log-density derivative on the model path -> log-ratio integral -> expectation.

1. Identity: along X'=v, d log(rho_t(X_t))/dt = -div(v)(X_t,t).
2. Identity: oracle continuity gives
   d log(p_t(X_t))/dt = -div(v*)(X_t,t)+e(X_t,t) dot s(X_t,t).
3. Identity: subtracting yields dL/dt=-a(X_t,t), L(0)=0, hence
   L(1)=-integral_0^1 a(X_t,t)dt.
4. Proposition under integrability: E[L(1)] over exact model-generated
   endpoints is KL(rho_1||p_1). A finite sample average need not be nonnegative.
5. Numerical approximation: integrate a and div(v) at the actual native
   Heun stages, keeping its generated state unchanged. Cross-check
   log(p_0(X_0))-integral div(v)-log(p_1(X_1)) against -integral a.

The instantaneous change-of-variables identity is established background,
not a proposed contribution: [Neural ODE, Theorem1](https://papers.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf),
[FFJORD, Section2.2](https://arxiv.org/html/1810.01367v2). We use exact traces
in dimensions2/16 instead of Hutchinson estimation.

**Numerical boundaries.** The formula is for the continuous field. A coarse
Heun endpoint paired with a divergence integral is not an exact likelihood
of the discrete sampling map. Also accumulate the actual step-Jacobian
log-determinants as a numerical consistency diagnostic, without assuming
global discrete-map injectivity from a positive determinant on observed
points. Refine state and log-ratio jointly; retain unresolved cases.

**Interpretation and falsification.** Test whether shared/released models
differ in the signed density-error accumulation, and where along time it
arises. This can explain a distribution difference only if the estimate is
resolved and a prespecified frozen-field intervention supports the predicted
effect. KL and SW2are different metrics; agreement is not assumed. The
identity applies to ordinary FFNs as well and does not establish novelty
by itself. If derivative/solver checks fail, no model ranking is reported.

## Completed numerical qualification

The oracle derivatives, actual native sampler observer and independent full-
map finite differences pass. All sixteen original/projected numerical cells
qualify: six at 512steps, ten at 1024. Local source RMSdoes not supply the
existing across-model sampling ordering. [Measured results and figures](density-transport/analysis-report.md).
The four-bank population-density comparison is [complete](density-transport/population-analysis.md):
all sixteen original model/field cells qualify independently. The subsequent
mean-preserving control has completed training, solver evaluation and all-source
density qualification. Its full paired analysis is complete.
[Current control](coordinate-control/analysis-report.md),
[completed density analysis](coordinate-control/density-results/analysis-report.md).
