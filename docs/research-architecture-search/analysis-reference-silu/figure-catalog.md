# Plain SiLU reference figures

Both figures use saved validation data only and were visually inspected. PDF versions preserve the same information as the PNGs.

| Figure | Data | Meaning |
|---|---|---|
| [Endpoints PNG](figures/silu-reference-endpoints.png), [PDF](figures/silu-reference-endpoints.pdf) | [comparison.csv](comparison.csv) | All seven arms and three metrics on all three tasks. Each panel divides by the better SwiGLU96/128 metric; lower is better. The plain SiLU reference is green. Both failed ODE variants and the larger SwiGLU192 remain visible. |
| [Learning PNG](figures/silu-reference-learning.png), [PDF](figures/silu-reference-learning.pdf) | [learning-curves.csv](learning-curves.csv) | Every recorded oracle-MSE point from update0 through5000 on a logarithmic y-axis. New SiLU trajectories are solid green; historical Stage17 controls are dashed. No smoothing or retrospective checkpoint selection. |

Oracle MSE uses the fixed uniform-time validation stream. Jacobian error uses the separate five-time diagnostic anchors. SW2 is the recorded Heun32 validation statistic; sampling paths were not rerun by this audit. Historical and new wall times are not shown as a controlled speed comparison because execution concurrency differs.
