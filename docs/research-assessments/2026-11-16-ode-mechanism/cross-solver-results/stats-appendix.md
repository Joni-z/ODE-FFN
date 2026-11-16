# Comparison units and exact paired statistics

Four retained 20k weight sets, training seed 0. Eight new full evaluations:
four crossed solvers by selection/confirmation. Each uses eight common4096-source
banks, 256 directions, Heun128 and 32,768 pooled samples. Native original-solver and
ordinary control metrics are reused only with matching asset/checkpoint hashes.
All eight new metric reconstructions have actual terminal 0, as do all evaluations.

Paired bank difference is crossed minus native SW2, keeping source/target and
directions identical. Enumerate 6,435 multisets with exact multinomial weights for
all 8^8 ordered bank resamples. Report marginal 95%, four-comparison-adjusted
intervals per split, and 8-comparison sensitivity over both splits. These describe
conditional evaluation-bank variation of the retained weights. Pooled SW2 and
the mean of bank SW2 are distinct statistics; both directions must satisfy the
prebound prediction. No arbitrary performance margin or best-arm selection.
