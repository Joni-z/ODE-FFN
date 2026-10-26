# Figure catalog

- field-comparison: all seven final primary MSEs on all tasks. The horizontal axis is logarithmic; red/orange denotes ODE and blue denotes generic controls. Model labels identify the readout and width.
- learning-curves: every saved primary validation point, including initial state and fixed5000 endpoint, for all seven arms. No favorable checkpoint is selected. Lines connect sparse logged observations, not continuous measurements.
- p-trajectories: mean of six actual P largest singular values at each of four snapshots; shading is the minimum/maximum across matrices. It is not a confidence interval and cannot certify full-network stability.

All three PNGs were visually inspected, with readable labels/legends. Corresponding vector PDFs are retained. Plots are generated from actual saved measurements with matplotlib. The complete field/Jacobian per-time table is available as per-time.csv, without selecting a favorable time or example.
