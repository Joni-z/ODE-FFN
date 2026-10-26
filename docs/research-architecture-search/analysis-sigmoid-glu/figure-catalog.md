# Figure catalog

All three figures were visually inspected after the completed analysis. Labels, legends and empty-bin handling are readable; PDF copies support export. Every panel uses the registered arms and saved results.

| Figure | Inputs and reading |
|---|---|
| [Endpoints](figures/sigmoid-glu-endpoints.png) · [PDF](figures/sigmoid-glu-endpoints.pdf) |`comparison.csv` and `higher-cost-historical-context.csv`. Primary error, Jacobian error and SW2 are divided by the better matched SwiGLU for that metric/task. Lower is better. Gray hatching marks the higher-cost historical256 model, excluded from the matched screen. Each panel has its own vertical scale. |
| [Learning curves](figures/sigmoid-glu-learning.png) · [PDF](figures/sigmoid-glu-learning.pdf) |`learning-curves.csv`: all 11 logged updates per arm/task, including step 0. Logarithmic error axes; curves connect saved updates without smoothing. This is fixed-budget learning, not a convergence claim. |
| [Posterior regions](figures/sigmoid-glu-posterior-regions.png) · [PDF](figures/sigmoid-glu-posterior-regions.pdf) |`posterior-regions.csv`: all Product16 times and all four fixed entropy bins. The first panel shows total field MSE; other panels show conditional coordinate MSE. Empty-bin means have no bar. High conditional error in a rare bin need not contribute much to total error; counts/contributions remain in the CSV. These are velocity errors, not classification errors. |

The historical baseline state replay is deterministic compatibility evidence, and the larger-model context is not an independent new run.
