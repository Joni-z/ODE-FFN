# Stage19 figure catalog

All figures use recorded Stage19 values and were visually inspected. PNG and PDF versions are equivalent.

| Figure | Source | Interpretation |
|---|---|---|
| [Endpoints PNG](figures/diagonal-endpoints.png), [PDF](figures/diagonal-endpoints.pdf) | [comparison.csv](comparison.csv) | Every arm and task, with each metric divided by the better of the two SwiGLU controls. Lower is better. Oracle MSE and five-time Jacobian error use distinct validation measures. |
| [Learning PNG](figures/diagonal-learning.png), [PDF](figures/diagonal-learning.pdf) | [learning-curves.csv](learning-curves.csv) | Every recorded primary oracle-MSE point from update0 through5000. Log y-axis, no smoothing or retrospective checkpoint selection. Overlapping K2/exp curves are retained rather than offset. |
| [Internals PNG](figures/diagonal-internals.png), [PDF](figures/diagonal-internals.pdf) | [diagonal-trajectories.csv](diagonal-trajectories.csv), [projection-trajectories.csv](projection-trajectories.csv) | Top: solid lines are the latent P singular maximum; dotted lines are the maximum absolute conditional-G eigenvalue on probes. Middle: observed scalar-map derivative minima. Bottom: E/R spectral-norm products. Lines average six solves in the first two rows and three FFN blocks in the last. Shading is their full range, not statistical uncertainty. |

P, conditional G and the full velocity Jacobian are different quantities. The observed scalar derivatives are probe statistics; the formal K2/exp bounds apply to each scalar map and do not imply full-network or sampling stability. Exact channel vectors are preserved in [eigenvalue-channels.csv](eigenvalue-channels.csv), and the complete recorded diagonal entries in [raw-diagonal-trajectory.json](raw-diagonal-trajectory.json).
