# Static stage: eliminate a default advantage of fixed K2

Six5000-step cells completed on Lenovo and were independently audited locally. All36 snapshots,30 common-state arrays, frozen source hashes and population oracles were checked; validation oracle errors were recomputed from the saved final weights. [Full audit](static_20261024/audit.json), [comparison](static_20261024/comparison.csv).

| Target | Fixed K2 field MSE | Collapsed free-B field MSE | SwiGLU field MSE |
|---|---:|---:|---:|
| Gaussian2D |.003600|.003251|.003453|
| Ring8 |.076519|.028901|.013891|

Fixed K2 and free-B have56706 parameters each, matched initial hidden functions and surrounding weights. Free-B uses55616 forward linear MACs/example versus80192 for fixed K2. Its Ring8 field error is62.2% lower with30.6% fewer MACs. K2's global feasible matrix set differs, so the experiment does not establish a pure optimizer mechanism.

This rejects treating static Taylor parameterization as an established advantage in this setting. It does not establish a new useful architecture: free-B is an ordinary narrow FFN and remains worse than SwiGLU on Ring8. SwiGLU's81792parameters are a different budget; a cheaper-model claim would also need a comparably narrow SwiGLU reference.

The two dimensional cases disagree on some derivative and generation rankings; retain the full table. No test split was used for the screen. Raw results are backed up at `/Users/mr.z/ode-workspace-prep/architecture-search/static_20261024/output`; the original runtime on Lenovo is frozen at `D:\Research\ODE-FFN\runtimes\static_20261024`.
