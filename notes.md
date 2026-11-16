# Current mechanism findings

Equal-parameter/equal-matrix-work RK4 improves hidden accuracy 45.580x/22.245x,
but not native generation. Full sequential-field interventions show the
midpoint bias is exposed and can reduce field error. Both Ring8 weight sets
prefer midpoint, so an own-solver adaptation story is insufficient.

Actual crossed sampling provides the next distinction. Product16 RK4 weights
improve pooled SW2 when sampled with midpoint: confirmation 0.065->0.060 and
selection 0.058->0.054. Both paired adjusted intervals favor the substitution,
although its field MSE slightly increases. Ring8's clear field gain does not
transfer consistently. Four of eight field-based point forecasts and two
adjusted forecasts hold. Free ISRU and SwiGLU remain stronger on Product16.
No general ODE-specific performance advantage is established.

Canonical evidence under docs/research-assessments/2026-11-16-ode-mechanism/:
- rk4-results/: matched training, hidden fidelity, samples and costs.
- field-exposure-results/: full 4,096-input risk accounting and compensation test.
- cross-solver-results/: all four weights/two splits, ordinary controls, paired
  statistics and independent verification. Original eight samples unchanged;
  verification input-binding bug and rejected wrong-split test retained.

Next: target-aware-bias-transport.md. Work around the actual learned field and
measure the derivative of the real target distance; do not revive the rejected
large oracle-field linearization or substitute paired endpoint error for quality.
Only after calibration, replace measured solver differences with the unfitted
ODE defect prediction. No response/architecture/training change yet.

Raw root: /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/.
Warm CPU environment 04f11f75 unchanged. Mac used four one-thread workers;145 GPUs
4-7 occupied and unused, 0-3 prohibited. Final Reports1/2 and image source unchanged.
