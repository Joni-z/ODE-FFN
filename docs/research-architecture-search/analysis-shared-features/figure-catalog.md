# Stage17 figure catalog

All figures use the saved Stage17 data. PNG and vector PDF versions contain the same information. They were visually inspected after generation.

| Figure | Source | Reading and scope |
|---|---|---|
| [Endpoints PNG](figures/shared-endpoints.png), [PDF](figures/shared-endpoints.pdf) | [comparison.csv](comparison.csv) | All six arms, three tasks, and three metrics. Each panel divides by the better SwiGLU96/128 value for that metric and task. SwiGLU192 remains visible; untied64's registered screen additionally compares against it. Lower is better. Oracle MSE uses the uniform-time validation stream; the Jacobian metric uses five fixed times. |
| [Learning PNG](figures/shared-learning.png), [PDF](figures/shared-learning.pdf) | [learning-curves.csv](learning-curves.csv) | Every recorded oracle-MSE point from update0 through5000, including all six arms. Logarithmic y-axis; lines connect observations without smoothing or checkpoint selection. |
| [P trajectories PNG](figures/shared-P-trajectories.png), [PDF](figures/shared-P-trajectories.pdf) | [P-trajectories.csv](P-trajectories.csv) | Largest singular value, relative drift from initial P, and clock-logit slope relative to its initial value. Solid lines average the six solves; shading is their full range, not a confidence interval. Drift is available at saved checkpoints every1000 updates; other statistics are recorded every500. Only the three ODE arms have P. |

The generation sample paths were not replayed or newly generated in this audit. SW2 is the recorded validation statistic; all raw values and checkpoint/source hashes remain available.
