# Measurement and audit scope

The unit is a fixed target/position/student training trajectory. The primary metric is all-validation MSE normalized by target energy. Five time panels and derivative rows are repeated measurements of the same function, not independent training replications. No p-value or training-variability confidence interval is claimed.

Twelve new cells use training seed 0, 5000 updates, batch 512, AdamW 3e-4 with default moments, weight decay 0 and gradient clip 1. Twelve historical records are reused only after constructor, initial-state, optimizer, targets, streams and native-output checks. No held-out test split is used for selection. Full streams comprise six unique target/position streams with 30000 replayed batches; matched arms share a stream.

The independent audit checks 96 native checkpoint forwards, 120 endpoint time panels, 24 endpoint Jacobians, 96 new matrix records with NumPy singular values, all 48 bias vectors and all frozen input/prior-document bindings. Explicit Jacobians and bias adjoints are separately compared with autograd; maximum recorded differences are 2.89e-15 and 5.55e-17. Input-Jacobian error is a squared Frobenius-risk ratio, not RMS error.

CSV exports include all 24 outcomes, 120 time panels, 96 saved validation measurements, 264 training records, four costs, 48 bias summaries, 3072 bias coordinates, 96 matrix summaries and 6144 singular values. Every reference row retains its historical status. The audit receipt lives at `/Users/mr.z/ode-workspace-prep/architecture-search/bias_placement_20261026/audit.json`.

The predeclared hidden-specific criterion requires at least two SwiGLU-target positions with relative MSE at most 1% and at most half both competing SiLU risks. Zero positions pass. This is an operational descriptive decision, not a significance test. Placement alters learning dynamics as well as representable responses; no pure expressivity conclusion follows.
