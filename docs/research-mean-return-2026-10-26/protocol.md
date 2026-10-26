# Fixed K2 mean-return intervention

Registered 2026-10-26 before computing return-path traces or intervention outcomes. This is a deterministic fixed-checkpoint diagnosis, not a training study. No earlier stopped architecture is reopened.

Let u=ones/sqrt(d), C=I-u u^T, and L_P(z)=z+a(z)Pz+a(z)²/2 P²z. Only the second solve of each original ODEFFN is intervened on. The first solve, SiLU, scalar once-computed input gate, conditioning, residual scales, normalization and readout remain unchanged.

Exact local identity:

```text
C[L_P(z)-L_(CP)(z)] = a(z)²/2 (C P u)(u^T Pz).
```

The primary implementation subtracts this centered term from the native second-solve output. Independent replay replaces each actual second-solve P by CP and uses the original model forward. Their full functions and input Jacobians must agree; local outputs can differ by a uniform shift. All downstream clocks/states are recomputed, with no detached gates.

Use the nine native checkpoints, three datasets, full validation-field banks and all five 512-input panels already bound by the normalized-geometry study: original ODE K2 native-final, SwiGLU192 native-final, SwiGLU128 native-deep. Restore source-v2 actual weights and verify native float32 full-panel predictions/Jacobians against saved artifacts before float64 intervention. Seed remains 0; no fresh data or optimization.

ODE variants, applied to all three second solves together:

- native;
- drop_uniform: delete only the final output mean, a function-preserving negative/procedural control;
- drop_return: subtract only the centered rank-one return path; the sole eligible modification;
- drop_secondorder: remove the entire second-order term of the second solve, a specificity reference, not an eligible model.

Generic variants: native and drop_uniform on complete FFN outputs. Both generic families are retained. The near-compute/loss match is SwiGLU192; SwiGLU128deep is a practical comparator with a different training objective. There is no claim that a generic FFN has a matched K2 pathway.

Every variant receives full-field prediction and five complete input-Jacobian panels. Save per-input outputs, derivatives, and squared-error polarization relative to its frozen native parent. At native ODE panel inputs retain all three blocks' z, duration, centered second-order term and return term; summarize the ratio of summed return energy to summed centered second-order energy. Ratios can exceed one because the return term can cancel other terms. Also report ||CPu|| ||u^T P||, a rank-one operator norm; no spectral/energy value is treated as a full-function stability certificate.

Tests must establish algebra for arbitrary float64 matrices and input-dependent scalar gates, failure of naive K2 gauge invariance, K1 gauge invariance, native/full mean-control identity, equivalence to actual CP generator replacement for full functions/Jacobians, input finite-difference derivatives, and no trace-induced output change. Training parameters and prior files must not change. Independent replay tolerance is prediction rtol1e-9/atol1e-10, Jacobian rtol1e-8/atol1e-9. Float32 native arrays must match bitwise. Stop and investigate any identity failure, without weakening a task-quality rule.

Only global drop_return can nominate more evaluation. Require field MSE at least 10% better than every generic native control on at least one nonlinear task, at most 5% above the best generic value on other tasks, at least 5% better than its ODE parent on that nonlinear task, and full Jacobian error within 10% of the best generic native value on both nonlinear tasks. This identifies at most a frozen inference modification; even a pass requires registered Heun32 SW2 controls, a fresh holdout, and a separately specified seed0 training/compute comparison before an architecture claim.

If no candidate qualifies, stop without return-path coefficients, layer selection, spectral-cap, gate, width, optimizer, norm or derivative-loss sweeps. A harmful ablation does not disprove a retrained constrained family, but it supplies no positive reason to spend a training budget on it. The all-second-order reference cannot become a post-hoc selected winner.

Use one CPU thread on Mac mini, actual process telemetry, terminal status and immutable source/protocol/checkpoint/bank hashes. These small replay computations do not need a GPU. Lenovo is offline at registration; deploy independent work there only if live availability is verified. No active-run restart based on an observation timeout.
