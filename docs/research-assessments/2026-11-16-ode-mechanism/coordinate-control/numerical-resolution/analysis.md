# Completed numerical controls for the mean-preserving release

All four numerical workers exited0. Independent NumPy reconstructs all12
new pooled scores and96bank scores within2.78e-17. Each row uses all32,768
original common-bank sources. Source and inputs stay bound to fd7f09b.

| Task / model | Internal4, Heun128 | Internal16, Heun128 | Internal4, Heun512 |
|---|---:|---:|---:|
| Ring8 / Shared P | 0.091 | 0.091 | 0.091 |
| Ring8 / Anchored stages | 0.111 | 0.103 | 0.111 |
| Ring8 / Mean-preserving | 0.104 | 0.105 | 0.104 |
| Product16 / Shared P | 0.064 | 0.069 | 0.064 |
| Product16 / Anchored stages | 0.073 | 0.082 | 0.073 |
| Product16 / Mean-preserving | 0.072 | 0.080 | 0.072 |

The outer-solver changes are small. Inner refinement changes the anchored
release more than the new release. On Ring8, the coarse point ordering
between the two releases reverses at16internal steps; the shared model
remains lower in the pooled score. The paired-bank evidence remains necessary
for any directional claim. This is a numerical comparison of fixed functions,
not training with16steps or a newly selected architecture.

The original ordinary controls remain in the primary20kcomparison. They
do not have an internal ODE solve. Do not substitute this three-model
numerical table for the full architecture-performance comparison.

[Scores](scores.csv), [bank scores](banks.csv), [independent audit](COMPLETE.json).
Population-density qualification remains in progress.
