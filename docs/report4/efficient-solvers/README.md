# Efficient internal solvers on Mac mini

Adams–Bashforth improves low-budget Product16 generation. It scores 0.049 in the selection stream and 0.055 in the confirmation stream, versus 0.057 and 0.062 for two-step midpoint at the same training cost. It wins seven of the eight confirmation banks at this budget.

| Short Product16 budget | Drift calls | Updates | Selection SW2 | Confirmation SW2 |
|---|---:|---:|---:|---:|
| Two-step midpoint | 4 | 10,000 | 0.057 | 0.062 |
| Two-step Ralston | 4 | 9,944 | 0.060 | |
| SSPRK3 | 3 | 12,275 | 0.065 | |
| Midpoint then Adams–Bashforth | 3 | 12,282 | **0.049** | **0.055** |

The lead changes at the full training budget. Adams–Bashforth takes about 19% fewer training operations per update than two-step midpoint. Its sampling time is about 13% lower on the Mac mini. The full-budget score falls between the existing two- and four-step midpoint scores on both datasets.

| Full-budget model | Ring8 confirmation SW2 | Product16 confirmation SW2 |
|---|---:|---:|
| Four-step midpoint | **0.071** | 0.052 |
| Two-step midpoint | 0.138 | **0.042** |
| Midpoint then Adams–Bashforth | 0.084 | 0.048 |

The three-call solver is a useful option for a short Product16 training budget. Selection and confirmation both favor full-budget two-step midpoint on Product16. The original Ring8 checkpoint using four steps remains the best of these trained models. Benchmarks use the median of six interleaved Heun32 runs of 512 samples on one CPU thread.

Averaging the three existing two-step checkpoints reduces Ring8 confirmation SW2 from 0.138 to 0.104 and sector-mass error from 0.053 to 0.038. Selection SW2 also improves from 0.154 to 0.116. On Product16, averaging reduces selection SW2 from 0.048 to 0.039 and increases confirmation SW2 from 0.042 to 0.044. Both tests use the same saved training trajectory.

All 14 pooled SW2 values were reproduced independently from the saved samples. [Results and checkpoint hashes](results.json) · [Training budgets](budget.json) · [Latency](benchmark.json) · [Independent verification](verification.json)

This fixed-budget autoresearch cycle keeps the full-width ISRU drift and tests whether three evaluations of it are enough. The Product16 short screen selects a solver. Its full-budget models then use the separate confirmation arrays.

The two-step Adams–Bashforth variant starts with one explicit midpoint step. Its second step uses 1.5 times the current slope minus 0.5 times the initial slope, already computed during the first step. With two half-steps this makes three matrix calls per local solve. Ralston uses two steps and four calls. SSPRK3 uses one step and three calls. The external flow-matching sampler stays Heun128 for SW2.

A follow-up tests checkpoint averaging as a separate run on the original two-step models. It equally averages the parameters at 25k, 30k and the registered full-budget endpoint, with no additional training. The earlier Ring8 result showed a better velocity fit and worse sector balance late in training. If averaging suppresses late oscillations, confirmation SW2 and sector balance should improve.

[Protocol and trainer](../../../studies/muon_pd_20261102/efficient_solvers_20261122/)


## Reproduce

Use the existing project synthetic Python environment. Run `cycle.py <work-dir>` on the dedicated study branch with the original work directories available. Run `cycle.py <work-dir> --confirm-short-budget`, `average.py <work-dir>`, then `benchmark.py <work-dir> <output-json>`. The cycle writes live phase and step records, per-candidate logs and an untracked `results.tsv`.
