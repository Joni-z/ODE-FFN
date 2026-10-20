# Figure catalog

All plots use real committed numerical evidence. No inferential error bars are
shown because independent training replicates are absent. Rebuild with
`python scripts/analyze_frozen_order.py`. The same PDFs are placed in Report 1.

| Figure | Purpose and variables | Source | Observation and interpretation limit |
|---|---|---|---|
| `09-frozen-order.pdf` | Full 4x5 SW2 matrix for each target; shared log color scale; blue outline marks trained order | `frozen-order-summary.json` | Low-order exact substitution worsens scores; K8 is close to exact. Rows are frozen checkpoints, not retraining. |
| `10-fixed-input-error.pdf` | Mean relative action error versus evaluation order; curve identifies trained order | `probes-145.json` | Numerical approximation improves with order. Six-layer means on 16 fixed inputs per layer; does not imply improving generation or a global error bound. |
| `11-ring8-examples.pdf` | Target and five model/order endpoint clouds, identical axes | `reports/report1/data/ring8-frozen-examples.csv`; original hashed arrays | K2 exact substitution visibly changes distribution. First 1,000 indices, no visual selection; no claim of exact mode counts or statistical coverage. |
| `12-geometry-and-training.pdf` | Matched-matrix gains versus chain shape; K2 learned sigma-max and stable-rank trajectories | `controlled-operators.json`; previous stage report values | Same eigenvalues/norm permit different gains; learned P concentrates. Constructed matrices and trained networks are distinct evidence, not one causal experiment. |
| `13-solver-check.pdf` | Ring8 SW2 versus Heun steps for all 12 selected modes | `solver-check-summary.json` | Intervention directions survive finer resolution. NFE equals twice steps. Post hoc target selection; no convergence proof. |

Captions must state seed 0, fixed endpoint 5,000 updates and fixed evaluation set
for learned-model comparisons. The operator example has no training seed.
The matrix figure prints values for color-independent interpretation. Curves
also use markers and line styles. Scatter plots have common limits and scales.

Before interpreting the figures, check:

1. Native order and evaluation order have not been conflated.
2. Exact means a direct matrix-exponential action, not exact integration of the
   external learned velocity field.
3. Local captured gates are distinguished from recomputed full-network gates.
4. No rounding of the main matrix is used to claim exact equality.
5. Baseline and finer-resolution controls remain in the accompanying evidence.
