# From hidden-operator diagnostics to sampled distributions

This is a conditional mathematical explanation, not a numerical certificate for
our checkpoints. The current synthetic network predicts velocity directly; it
is not the JiT image-prediction parameterization.

Let u(x,t) be the population marginal velocity, transporting the source to p_t,
and v(x,t) the learned field. Couple exact flows X_t,Y_t with the same initial
Gaussian sample, using u and v respectively. Assume well-posed flows, finite
second moments, and a global spatial Lipschitz bound L_t for v, with integrable L.
Write

```
e_t = sqrt(E ||v(X_t,t)-u(X_t,t)||^2)
r_t = sqrt(E ||Y_t-X_t||^2).
```

Insert and subtract v(X_t,t) in the difference of the two vector fields. The
Lipschitz inequality and Minkowski's inequality give the integral Gronwall bound

```
W2(Law(Y_1),p_1) <= r_1
r_1 <= integral_0^1 exp(integral_t^1 L_s ds) * e_t dt.
```

For a discrete Heun trajectory Y^h coupled to the same source, add the numerical
term sqrt(E||Y^h_1-Y_1||^2) by the triangle inequality. No numerical order bound is
asserted without the additional spatial/time regularity and constants it requires.
Increasing NFE is an empirical check, not a proof that the final resolution is exact.

Three practical consequences follow.

1. An unweighted average field error need not order final sampling quality. Where
   the error occurs in time/state space and how errors propagate can matter.
2. The relevant L_t concerns the complete predictor, not an isolated P or one
   hidden-layer Jacobian. Our16 complete-field probes do not estimate a certified
   global L_t and must never be inserted as such into the exponential bound.
3. Exact-field approximation and discrete integration error are distinct. The
   registered frozen Heun32/64/128 readout tests the latter without retraining or
   replacing the primary Heun32 protocol.

Our reported oracle MSE averages coordinates, so a population quantity with that
convention contributes e_t=sqrt(dim*MSE_t), not sqrt(MSE_t). Finite cached-bank
validation is only an empirical proxy for this expectation. The finite-projection,
finite-sample sliced-W2 readout is also not the population W2 in the displayed
bound; its reference-sampling floor remains part of the experimental report.

The [earlier exact marginal-field derivation](../../research-architecture-search/complete-field-constraints.md)
characterizes u for the independent linear coupling. It does not imply that a
hidden P must be symmetric, contractive or spectrally flat. This is why the current
logic chain separates P/update spectra, conditional PD, complete predictor behavior,
and final sampled distance instead of equating any two of them.
