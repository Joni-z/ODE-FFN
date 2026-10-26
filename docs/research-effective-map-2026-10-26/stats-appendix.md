# Statistical and computational scope

The cohort contains six fixed target functions at three positions and three student families: fifteen new seed-0 training cells and three verified historical controls. Positions, times, output coordinates and validation rows are dependent conditions, not independent training replications. All results are descriptive empirical risks; no training-population significance test or confidence interval is asserted.

Primary MSE averages all 64 outputs and all 20000 existing validation rows. Relative MSE divides by the corresponding target's mean squared output. Secondary derivatives use the fixed first 128 common validation inputs, comparing complete 64-by-64 module Jacobians. Five time panels have 512 inputs each. No new test set is used. Teacher energies and every absolute risk are in comparison.csv; cross-target amplitude normalization does not imply equal difficulty.

The substantial known-representer coordinate-difficulty rule was fixed before outcomes: at least two fixed-flow targets with P error above 1%, W at or below 0.25%, and P at least four times W. It does not occur. This is an operational descriptive criterion, not a hypothesis-test p-value or architecture acceptance rule. No learning-rate, bias, width or continuation rescue was added.

## Actual work

| Learner | Parameters | Batch-512 forward MACs | Batch-512 forward/loss-backward MACs |
|---|---:|---:|---:|
| P core | 8192 | 4718592 | 12058624 |
| W core | 8192 | 4194304 | 10485760 |
| SwiGLU128 | 16554 | 8355840 | 19496960 |

The P counts include Q formation and differentiation through P². Pointwise work is additional. A trained constant-duration P core can fold its effective matrices, giving the same 8192 matrix MACs per inference input as W; raw per-call formation is separately exported. These are dispatch counts, not latency measurements. The ordinary control is not parameter matched to the two smaller cores.

## Pairing and representability
All three initial P/W functions and input Jacobians match bitwise in float32. The independent sequential-polynomial expression is checked in float64. Six cold/teacher P-gradient adjoint preflight checks pass, as do six exact fixed-flow representers, three exact SwiGLU representers, fifteen exact resume histories and three historical-control three-step replays. Stored teacher weights enter only procedural representer checks, never a cold student's initialization.

The three reused controls keep their original checkpoints, arrays and logs. Their native constructors, optimizer state, data/target files and complete consumed index/input/label chains match this study. Reuse is recorded in output/reused/ and the audit cohort. New fixed-flow labels are evaluated through the declared folded-Q convention; they are not relabeled native adaptive-clock targets.

## Independent audit
The corrected audit replays all 367680 fixed-flow target rows, six complete 5000-batch consumed streams, 72 checkpoint forwards, 90 endpoint time panels and eighteen endpoint Jacobians. NumPy independently checks all 192 raw/effective/control matrix records, signed determinants and origin Jacobians. Twelve endpoint P-gradient adjoint checks pass. Separate closed/sequential forwards and Jacobians agree with native differentiation within 1.78e-14 and 2.89e-15; adjoint maximum discrepancy is zero. All 178 bindings and 74 previous documentation files retain their hashes.

Audit attempt 1 failed before numerical work because `time` was shadowed by a loop variable. That source remains in audit-source/. The renamed-loop audit is frozen separately in audit-source-v2/; training source and results are unchanged. The passing receipt is `/Users/mr.z/ode-workspace-prep/architecture-search/effective_map_20261026/audit.json`.

A successful numerical audit is not a cause or capacity theorem. The determinant result concerns a smooth real-weight function at zero, which is not a data-distribution lower bound. A known-representer fitting contrast can speak to finite-budget learning, while a contrast on an unknown-representability target cannot isolate optimization from class restrictions.

## Complete exports
- comparison.csv: 18 rows.
- time-panels.csv: 90 rows.
- learning-curves.csv: 72 rows.
- training-metrics.csv: 198 rows.
- costs.csv: 3 rows.
- matrix-trajectories.csv: 192 rows, including every raw P/effective W and control matrix.
- origin-jacobians.csv: 72 rows, including all saved steps and all reused controls.
