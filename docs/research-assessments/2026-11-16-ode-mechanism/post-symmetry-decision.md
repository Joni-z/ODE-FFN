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
error e=aJx with skew-symmetric Jhas zero divergence and is orthogonal to
the radial score, so the density source vanishes. A radial error e=ax can
have the same mean squared magnitude while changing the density. This is
a mathematical example, not an explanation already measured in our models.

## Bounded execution plan — prepared, not launched

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
4. Treat residual norms as a diagnostic, not a guaranteed SW2bound. Register
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
