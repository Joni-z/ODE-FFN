# Comparison scope

Four arms, three populations and the registered5000-update seed0 protocol.
Primary values are exact descriptive run endpoints. The five evaluation
times, hidden layers and validation anchors are not independent training
replications. No significance tests or confidence intervals are constructed.

The parent/candidate have identical trainable tensors at initialization and
identical data RNG states at every saved snapshot. Internal hidden functions
differ for vector durations; this is not a same-function reparameterization.
Both initial full output functions are zero. Frozen-generator spectral
similarity is asserted only for the same D/P, not for separately trained
models. Positive sigmoid durations make D invertible.

Validation oracle MSE, common fixed-time field error, Jacobian error and
Heun32 validation SW2 have different evaluation laws. Do not substitute one
for another or compare SW2 directly to a marginal-Wasserstein metric.
The analytic oracle refers to the target population; pair training samples
the same finite training bank in all arms. That distinction is unchanged.

Wall times are observations from this queue, not isolated hardware benchmarks.
MACs count linear operations and exclude pointwise/control overhead. Both
are reported because these architectures have different numbers of small
operations. No test-set tuning, additional seeds or fresh holdout is used
for this failed candidate.
