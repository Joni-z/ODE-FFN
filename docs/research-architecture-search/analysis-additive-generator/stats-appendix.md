# Measurement scope

All comparisons use the new Mac studies' controls. Previous Windows endpoints
are historical motivation, even when they closely agree numerically.
All arms use the existing validation pairing and five-time diagnostic arrays.
The [preceding measurement definitions](../analysis-identity-location/stats-appendix.md)
apply unchanged: coordinate-mean oracle MSE, full squared-Frobenius Jacobian
error and fixed-stream Heun32 validation sliced-Wasserstein comparison.
Training uses bank-pair labels, whereas the analytic oracle is the population
conditional mean. Neither teacher nor generation stream is a new holdout.

`screen-decision.json` applies the declared thresholds to both additive arms
and both baselines. Endpoints, counts, matrix MACs and wall times for every
cell are in `comparison.csv`. These are descriptive results under the stated
setup; no confidence intervals, p-values or population effect sizes are inferred.

The local mathematical risk floor concerns one latent scalar generator across
the five specified times, with unrestricted positive controllers. Numerical
quadrature values and ideal constructive witnesses are not neural training
results. The exact finite-controller witness establishes a class difference,
and the converse example establishes non-inclusion. Neither result establishes
a bound for the actual dense hidden multi-block velocity model.

Paired initial functions and predicted non-controller gradients match exactly
in the full-width float32/float64 preflight. The zero initial velocity head
is not used to hide initialization defects: a shared nonzero head is tested.
Frozen predecessor defaults also preserve three-update optimizer trajectories.
The additive trainer's resumed three-update state matches uninterrupted
weights, optimizer, runtime buffers and RNG states exactly.
