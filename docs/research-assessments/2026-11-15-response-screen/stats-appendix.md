# Statistical units and reproducible comparisons

## Units and protocol

There are 36 trained trajectories: 32 focal response/order/task cells plus
four ordinary controls. Each uses training seed0. Selection evaluates15k and
20k, yielding72 metric rows. The frozen shortlist confirms12 final20k
checkpoints using a different evaluation stream. No checkpoint is selected
using confirmation results.

Each metric stream/task contains4,096 exact-oracle field states and eight
source/target sample banks of4,096 points with256 projection directions.
Sampling is Heun128. Pooled SW2 is computed after concatenating all32,768
samples. It is not the arithmetic average of bank SW2. Ring8 and Product16
have different geometries and dimensions; compare models within each task,
not raw SW2 magnitudes across tasks.

Training stream and evaluation streams are distinct. Confirmation means
fresh evaluation arrays for the same trained weights; it is not a new fit.
The eight-bank standard deviation describes sampling/evaluation variation
conditional on those weights. It does not measure training variability.

## Paired differences against SwiGLU

The unit here is a common evaluation bank. Compute
`delta[b] = SW2(candidate,b) - SW2(SwiGLU,b)` for the same bank and directions.
Negative values favor the candidate. The intervals below are pointwise
Student-t intervals of the eight-bank mean difference (`df=7`). They are
conditional descriptive comparisons, without multiplicity adjustment or a
population-level claim about retrained architectures. They are not intervals
for the nonlinear pooled SW2 statistic.

| Task | Candidate | Mean paired difference | SD of differences | Conditional95% interval | Banks improved |
|---|---|---:|---:|---|---:|
| Ring8 | ISRU K1 | +0.002383 | 0.006650 | [-0.003177, +0.007942] | 4/8 |
| Ring8 | ISRU K8 | -0.002631 | 0.015947 | [-0.015963, +0.010701] | 4/8 |
| Ring8 | Arctan K1 | -0.010619 | 0.007354 | [-0.016767, -0.004471] | 8/8 |
| Ring8 | Arctan K8 | -0.008260 | 0.013746 | [-0.019752, +0.003232] | 6/8 |
| Product16 | ISRU K1 | +0.018686 | 0.002528 | [+0.016572, +0.020799] | 0/8 |
| Product16 | ISRU K8 | +0.017035 | 0.004201 | [+0.013522, +0.020547] | 0/8 |
| Product16 | Arctan K1 | +0.016615 | 0.003139 | [+0.013991, +0.019239] | 0/8 |
| Product16 | Arctan K8 | +0.014998 | 0.003824 | [+0.011801, +0.018195] | 0/8 |

Exact bank values, arithmetic and relative pooled changes:
[paired-confirmation.json](results/paired-confirmation.json).
All model bank means/SDs and field MSEs are in
[confirmation-metrics.csv](results/confirmation-metrics.csv), with raw bank
scores in [confirmation-banks.csv](results/confirmation-banks.csv).

## Bindings and checks

- Every pipeline completed20k training and its required selection evaluations;
  all36 exits are zero and no cell is missing or unstarted.
- Selection consists of36 cells/72 rows; confirmation has12 cells/12 rows.
  Original generated samples independently reproduce metrics in NumPy, with
  maximum difference5.551115123125783e-17 for each stream.
- The shortlist SHA256 remains
  `214045c0a46b283e4ed5cb5a012ce76c6f604f5ca497a96360cf9fafd4da2e2f`.
  Metric verification binds each checkpoint, configuration and asset manifest.
- Training source is `b5fe7aa`; independent analysis source is `0e78ce7`.
  Runtime worktrees and raw samples remain outside Git. The curator copies
  only aggregate evidence and figures into this directory.

[Verification](results/verification.json),
[confirmation verification](results/confirmation-verification.json),
[shortlist](results/shortlist.json), [cost measurements](results/latency.json),
[completion manifest](completion.json).
