# Statistical scope

The comparison unit is the registered model trajectory on a fixed task and
budget. Checkpoints along one trajectory are repeated observations, not
independent replications. Image methods share actual minibatches and noise;
the synthetic table reuses its audited historical controls where recorded.

This draft reports descriptive endpoint values and complete available curves.
There are no p-values, confidence intervals, significance claims, or invented
run-variation estimates. Model seed0 and the separately named data generators
are recorded in the protocol/config. No inference about seed instability is
made or added to the report.

Image FID uses standard2048-dimensional Inception features and empirical
sample covariances with denominator n−1. The independently checked identity is
`||meanX-meanY||² + ||A||F² + ||B||F² - 2*sum(svd(A B^T))`, with centered,
normalized sample factors A/B. Numerical agreement is below0.000024 on all
completed rows. This verifies a metric calculation; it is not a quality result.

Validation uses1024fixed held-out noisy examples under the registered time
distribution. These losses differ from the oracle-field errors of synthetic
tasks and should not be compared numerically across the two benchmarks.

Matrix bands are the observed minimum/maximum across six blocks. They are
not uncertainty bands. The entropy effective rank uses normalized singular
values, rather than squared values; a maximum of192 applies to each square
ISRU map. These statistics do not bound the entire normalized residual model.
