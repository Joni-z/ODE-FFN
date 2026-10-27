# Final native next-batch probes

| Law | Family | Forced loss % | Clip full / removed | Hidden Adam change % | P tangent/normal cosine |
|---|---|---:|---:|---:|---:|
| thin | ode_k2 | 0.00 | 1.0000 / 1.0000 | 0.000 | 0.047 |
| thin | swiglu192 | 0.00 | 1.0000 / 1.0000 | 0.000 | — |
| thin | frame_k2 | 0.00 | 1.0000 / 1.0000 | 0.000 | -0.085 |
| thin | global_k2 | 0.00 | 1.0000 / 1.0000 | 0.000 | 0.034 |
| thin | frame_k1 | 0.00 | 1.0000 / 1.0000 | 0.000 | -0.075 |
| thin | swiglu257 | 0.00 | 1.0000 / 1.0000 | 0.000 | — |
| thin | jet_c | 0.00 | 1.0000 / 1.0000 | 0.000 | 0.033 |
| thin | affine_c | 0.00 | 1.0000 / 1.0000 | 0.000 | 0.037 |
| thin | free_c | 0.00 | 1.0000 / 1.0000 | 0.000 | 0.032 |
| thin | direct_b | 0.00 | 1.0000 / 1.0000 | 0.000 | -0.019 |
| thin | swiglu258 | 0.00 | 1.0000 / 1.0000 | 0.000 | — |
| thick | ode_k2 | 73.12 | 1.0000 / 1.0000 | 0.000 | 0.111 |
| thick | swiglu192 | 72.98 | 1.0000 / 1.0000 | 0.000 | — |
| thick | frame_k2 | 73.02 | 1.0000 / 1.0000 | 0.000 | 0.234 |
| thick | global_k2 | 72.90 | 1.0000 / 1.0000 | 0.000 | 0.089 |
| thick | frame_k1 | 73.10 | 1.0000 / 1.0000 | 0.000 | 0.217 |
| thick | swiglu257 | 72.93 | 1.0000 / 1.0000 | 0.000 | — |
| thick | jet_c | 73.12 | 1.0000 / 1.0000 | 0.000 | 0.185 |
| thick | affine_c | 73.13 | 1.0000 / 1.0000 | 0.000 | 0.166 |
| thick | free_c | 73.11 | 1.0000 / 1.0000 | 0.000 | 0.188 |
| thick | direct_b | 73.02 | 1.0000 / 1.0000 | 0.000 | 0.153 |
| thick | swiglu258 | 72.79 | 1.0000 / 1.0000 | 0.000 | — |

Adam columns are promoted one-step predictions with saved moments, not actual retraining.
