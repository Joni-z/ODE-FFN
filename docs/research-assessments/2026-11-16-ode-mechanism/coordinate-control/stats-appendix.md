# Paired-bank statistical appendix

The comparison unit is a common 4,096-source bank at a fixed checkpoint.
These intervals describe sampling-bank variation conditional on the trained
functions. They do not estimate training-run variation. Bank pairing is
retained. All scores are lower-is-better; differences are new minus reference.

The empirical paired bootstrap distribution is computed using all 6,435
occupancy vectors for eight draws from eight bank differences, with their
multinomial weights. Enumeration is exact; percentile-interval population
coverage is approximate. A small n=3 independent ordered enumeration and a
constant-difference case pass. The two-sided exact sign test assumes
independent, exchangeable nonzero bank signs under its null. Holm corrects
all four displayed tests. Shapiro checks are descriptive at n=8; the sign
test does not rely on normal differences. Percentile intervals are pointwise,
not familywise-adjusted. Every distribution and effect size is stored in
comparison.json.

| Task / reference | Mean difference | SD | 95% percentile interval | New better banks | Paired standardized difference | Sign p | Holm p | Shapiro p |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Ring8 / shared | 0.007 | 0.011 | [-0.001, 0.014] | 3/8 | 0.616 | 0.727 | 1.000 | 0.324 |
| Ring8 / anchored | -0.006 | 0.010 | [-0.012, 0.000] | 7/8 | -0.640 | 0.070 | 0.211 | 0.213 |
| Product16 / shared | 0.006 | 0.003 | [0.004, 0.008] | 0/8 | 1.703 | 0.008 | 0.031 | 0.656 |
| Product16 / anchored | -0.001 | 0.003 | [-0.003, 0.001] | 5/8 | -0.279 | 0.727 | 1.000 | 0.411 |

The Product16 shared-versus-new contrast survives the four-test correction.
The remaining tests do not resolve a directional difference under that rule.
Bank mean differences need not equal pooled-score differences.
