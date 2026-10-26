# Low-rank state allocation: no practical lead

The fixed allocation fails. Factoring the original projections at the same
parameter and dense-MAC budget already worsens both nonlinear tasks. Spending
that budget on a larger state does not recover the loss. None of the three
low-rank arms passes the registered screen, so Phase B is not selected.

The previous clock-collapse diagnostic showed that conditional clocks help
the trained parent. This study retained both K1 solves, their original clocks
and the middle SiLU, while replacing P by two factors U and V. The proposed
model pads 64 inputs to 128 coordinates before SiLU and returns the first 64
coordinates after the second solve. Unused zero columns and discarded rows
are removed exactly. Narrow controllers and rank-21 projections pay for the
extra state. There is no new head, objective, label law or solver order.

The controls separate several practical questions. Factor64/q32 has exactly
the dense parent's budget. Narrow64/q21 shares the expanded model's common
initial tensors and initial function within floating-point tolerance, but
costs less. Their parameter responses differ once a nonzero output head can
receive the extra features. Dense and factorized models also have different
initial maps and feasible projection sets. These are allocation comparisons,
not a pure optimization or augmentation-only proof.

All values below are final primary oracle MSE. Asterisks mark previously audited
Stage23 outcomes, retained in the generic comparison without new training.

| Arm | Gaussian2D | Ring8 | Product16 |
|---|---:|---:|---:|
| Dense ODE parent | 0.003772 | 0.019930 | 0.266155 |
| Factor64/q32 | 0.004302 | 0.032344 | 0.310046 |
| Narrow64/q21 | 0.004014 | 0.038695 | 0.346669 |
| Expanded128/q21 | 0.003992 | 0.037952 | 0.358879 |
| SwiGLU128 | 0.003927 | 0.015296 | 0.092326 |
| SiLU128 | 0.004384 | 0.033714 | 0.153892 |
| SwiGLU96* | 0.004188 | 0.018258 | 0.109161 |
| GLU128* | 0.003717 | 0.027148 | 0.158060 |

Factor64/q32 raises Ring8 error by 62.29% and Product16 error by 16.49%
relative to the dense parent. Expansion then raises those errors another
17.34% and 15.75% relative to Factor64/q32. Against the cheaper narrow model,
expansion improves Ring8 by only 1.92% and worsens Product16 by 3.52%.
It remains 2.48 times worse than SwiGLU128 on Ring8 and 3.89 times worse on
Product16. Its Jacobian and SW2 ratios also fail the nonlinear guards.

The narrow model does save resources: on Product16 it uses 20.21% fewer full-model
parameters and 20.42% fewer dense MACs than SwiGLU128. Its substantially higher
error rejects the registered quality-preserving cheapness claim. Concurrent
training wall times are retained but are not latency measurements.

The expanded path did learn. On Product16, averaged across the three blocks,
the extra first-solve output RMS is 1.158 times that solve's input RMS, and the
extra input to the second solve is 0.798 times its full input RMS. Extra V2
columns grow from zero to mean Frobenius norm 2.231; extra second-controller
input columns reach 5.817. All twelve tested extra factor/controller slices
already move in the three-update preflight. An uninitialized or permanently
dead extra branch is therefore inconsistent with these records. Activity alone
does not establish that the branch is useful; no post-training removal was run.

Across six solves, Product16's mean largest singular value of the full padded P is 3.894 in the
expanded model, versus 1.674 in the dense parent. Its mean probe-clock VDU
spectral radius is 0.538. These are different internal operators and widths,
not evidence that a full velocity field is stable or unstable. Factor norms
depend on factor coordinates, and the reduced spectrum omits clock derivatives.
All P, factor, clock and extra-path trajectories remain available.

Stop this allocation without rank, width, controller, initializer or clock
rescue. The results do not isolate rank restriction, SVD initialization and
optimization as separate causes, and do not prove a general limit of ODE
architectures. They do reject this concrete way of retaining clocks while
expanding the nonlinear state at the existing budget. The original useful-ODE
goal remains open.

[Complete metrics](comparison.csv), [audit details](stats-appendix.md),
[registered decision](screen-decision.json), and [figures](figure-catalog.md)
preserve every control and negative outcome.
