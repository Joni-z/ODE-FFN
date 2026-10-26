# Figure catalog

All four figures were visually inspected after complete analysis. Labels, legends and scope statements are readable; all six registered arms remain in the endpoint and learning comparisons. PDF copies are provided for export.

| Figure | Data and limits |
|---|---|
| [Endpoints](figures/affine-forcing-endpoints.png) · [PDF](figures/affine-forcing-endpoints.pdf) |`comparison.csv`. Each metric is divided by the best of SwiGLU96, SwiGLU128 and GLU128 on that task; lower is better. Each panel has its own scale. Gaussian improvements alone do not qualify. |
| [Learning](figures/affine-forcing-learning.png) · [PDF](figures/affine-forcing-learning.pdf) |`learning-curves.csv`. All 11 logged updates per arm/task, including zero; logarithmic error axes, no smoothing. Near-overlapping ODE curves are retained rather than offset for visibility. |
| [Internal quantities](figures/affine-forcing-internals.png) · [PDF](figures/affine-forcing-internals.pdf) |`forcing-trajectories.csv`. Lines are six-solve means and shading is the range across solves, not statistical uncertainty. Raw P norms are shown for all three ODE arms; bias/correction/local-Jacobian rows describe the conditional arm. Mean and centered correction RMS are separate quantities. |
| [Fixed-weight replacement](figures/affine-forcing-staticized.png) · [PDF](figures/affine-forcing-staticized.pdf) |`affine-to-static.csv`. Every time and task is included. All six conditional bias terms are replaced with static ones; downstream clocks and complete Jacobians are recomputed. This is checkpoint dependence, not retraining, a primary-metric evaluation or a generation test. |

Complete numerical tables and channel trajectories accompany the plots; visible small differences are interpreted using the registered practical margins.
