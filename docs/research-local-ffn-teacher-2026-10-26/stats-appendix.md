# Statistical and verification scope

## Units and estimands
The comparison unit is a fixed local teacher function, position and cold student. There are eighteen seed-0 training runs, six target functions and three positions from the same two trained backbones. Positions, time panels, coordinates and validation rows are not independent training replications. No p-values, training-population confidence intervals or unsupported mean-plus-standard-deviation claims are reported.

The primary estimand is empirical mean squared output error on all 20000 fixed validation inputs. Relative MSE divides by the same teacher's mean squared output. Output coordinates are averaged within each input. Comparisons are within a target function; normalization does not justify an equal-difficulty assumption across teachers. The derivative estimand is the mean squared Frobenius Jacobian error on the first 128 common validation inputs, divided by teacher derivative energy. Fixed-time evaluations use five times and 512 inputs each. No test set was opened.

The 5%/1%/fourfold local-gap pattern was declared before student outcomes. It is a descriptive operational criterion, not a statistical test or an architecture acceptance rule. All cells are reported; no budget, learning-rate, target-strength or width setting was selected after outcomes. Nominal SwiGLU widths preserve the existing 2/3 convention: actual hidden widths are 85 and 128.

## Actual local cost

| Student | Parameters | Forward matrix MACs | Forward plus loss-backward MACs |
|---|---:|---:|---:|
| Original ODE | 16642 | 24704 | 65920 |
| SwiGLU128 | 16554 | 16320 | 38080 |
| SwiGLU192 | 24896 | 24576 | 57344 |

Counts come from actual dispatch for the isolated FFN and its parameter gradients. Input features are fixed, so no upstream prefix gradient is computed. Pointwise work is excluded. ODE and SwiGLU128 are near parameter matched; SwiGLU192 is close in forward matrix work, with lower backward work. These are not latency measurements or exact full-cost matches.

## Exploratory projection
The projection analysis was specified separately after aggregate risks were viewed. Design matrices contain the constant and all 64 common input coordinates. All have empirical rank 65 under the fixed 1e-12 relative SVD tolerance; retained condition numbers are 277.35, 163.97 and 108.67. Orthogonality makes error energy additive. Projection is fitted on the same validation bank to describe error components; it does not estimate an out-of-sample affine repair. An independent full-rank QR/coefficient-energy computation agrees for all eighteen errors, six teachers and twelve contrasts within 4.11e-15. See [projection-audit.json](projection-audit.json).

The nonaffine share of positive ODE excess is a descriptive ratio. It is not a causal mediation fraction or a variance-explained parameter of training. No confidence interval is manufactured by treating coordinates or positions as independent experiments.

## Preflight and formal audit
The passing preflight checks fifteen cold/trained-module forward and input-Jacobian cases against separate closed formulas, six exact representers, six cached/live label checks and all eighteen exact three-step versus two-plus-resume histories. The largest preflight double-precision discrepancy is 1.78e-15.

The formal audit reconstructs all 367680 common-prefix rows and 735360 label rows bitwise under the original cache batching. Six complete 5000-batch input/label/index streams reconstruct every checkpoint digest. It replays all 72 checkpoint validation predictions, 90 endpoint time panels, eighteen endpoint Jacobians, six teacher Jacobians, and 48 P spectral records. Independent closed local formulas check 36 teacher/student panels; maximum forward and derivative discrepancies are 7.11e-15 and 2.89e-15. All 69 cache files, 58 ready bindings and 77 prior documentation files pass their hashes. See the runtime audit at `/Users/mr.z/ode-workspace-prep/architecture-search/local_ffn_teacher_20261026/audit.json`.

A successful audit proves computational consistency within this scope. It does not prove generalization to images, identify an optimizer cause, establish an expressivity theorem, or nominate a method. No teacher weights are copied into a trained cold student; own-family exact representers are separate procedural controls.

## Exports
- [Every endpoint](comparison.csv): 18 rows.
- [Every fixed-time risk](time-panels.csv): 90 rows.
- [Every saved validation risk](learning-curves.csv): 72 rows.
- [All logged training windows](training-metrics.csv): 198 rows.
- [All P checkpoint summaries](P-trajectories.csv): 48 rows.
- [Actual module costs](costs.csv): 3 rows.
- [All error decompositions](error-components.csv): 18 rows.
- [Both generic contrasts](component-contrasts.csv): 12 rows.
- [All teacher decompositions](teacher-components.csv): 6 rows.
- [Input projection geometry](input-geometry.csv): 3 rows.
