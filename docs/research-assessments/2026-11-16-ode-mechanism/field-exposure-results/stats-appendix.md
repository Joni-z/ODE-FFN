# Numerical and statistical units

Four checkpoints, training seed 0/20k; no new training. Each case retains 4,096 IID
field inputs, all six local solves and six forward modes. Original-identity replay
is exact. Two complete DOP853 tolerances, per-component/layer gates, energy closure,
and independent state-only RK45 on differently packed 64-point chunks qualify every
continuous local output. Numerical risk resolution is the measured difference
between complete tolerance passes and its Cauchy-Schwarz energy bound; it is a
resolution diagnostic rather than a rigorous certificate for the unknown exact
solution. Risk uses vector squared norm; displayed MSE divides by physical dimension.

Primary comparison: own-solver loss minus other-solver loss for each weight set,
two tasks by two training solvers. Conditional paired bootstrap over 4,096 original
IID field points keeps all solver predictions aligned. Evaluation RNG 0, 10,000
resamples, batches 100; percentile 95% and Bonferroni four-comparison intervals.
No fabricated sampling banks or independent training runs. Full per-state
contrasts and bootstrap arrays are source-bound outside Git. Both native and
both independent commands have actual terminal 0.
