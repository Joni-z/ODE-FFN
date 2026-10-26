# Figures and data

1. [Teacher fitting](figures/teacher-fitting.png), also [PDF](figures/teacher-fitting.pdf). The heatmap shows relative teacher MSE in percent for all six endpoints; the color scale is logarithmic and every cell is numerically annotated. The two curve panels retain all four saved validation evaluations for each student. Sources: comparison.csv and learning-curves.csv. Lower error is better. Relative squared error is not relative RMS or FID.

2. [Original P trajectories](figures/P-trajectories.png), also [PDF](figures/P-trajectories.pdf). All six P matrices in each of the two ODE students at the four saved checkpoints. B denotes Transformer block, S denotes its first/second original solve. These are raw matrix spectral norms, not complete-field Jacobian norms, stability certificates, uncertainty intervals or architecture-quality metrics. Source: P-trajectories.csv; all Frobenius norms and symmetric minima are also retained.

Both exported PNG figures were visually inspected. Labels, data annotations and legends are readable, no model/time/checkpoint is selected away, and all measurements come from independently replayed saved outputs and checkpoints. The plotting script exports standalone PDF versions alongside the PNGs.
