# Statistical unit and exact uncertainty

The primary observation is a deterministic summary on128 paired interpolation
paths at fixed weights. Each pair is reused at five times. Average times within
each pair before resampling; do not treat640 rows as independent. The two-task
decision uses the registered point-gap direction, not a post-hoc p-value gate.

Conditional paired bootstrap:10000 resamples of128 source/target pair indices,
analysis RNG0, percentile intervals. The97.5% intervals use a two-contrast
Bonferroni family. SD describes point-pair differences. These intervals concern
input sampling at fixed checkpoints, not independent-training uncertainty.
No parametric normality assumption, t-test, across-model significance or
training-repeat confidence interval is used. No p-values are claimed.

- Ring8: paired gap -0.013, SD 0.182; conditional95% interval ['-0.049', '0.013']; two-contrast interval ['-0.056', '0.017'].
- Product16: paired gap 0.012, SD 0.114; conditional95% interval ['-0.007', '0.032']; two-contrast interval ['-0.010', '0.035'].

All twelve models and sixty time rows are present. Existing pooledSW2 values
retain their original32768-sample common-bank identity. Derivative states use
the selection cohort; SW2 is the separate preexisting confirmation cohort.
Different task scores and historical cohorts must not be pooled.
