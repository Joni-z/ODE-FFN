# Figure catalog

Both figures were visually inspected: labels, legends and signs are readable,
all five times and three datasets are present, and axes identify the distinct
field-MSE and mean squared Frobenius-Jacobian units. Linear scales and task-specific
vertical limits retain the measured magnitudes; no confidence interval or
population/generalization uncertainty is implied.

1. [Native and collapsed errors](figures/clock-collapse-errors.png)
   ([PDF](figures/clock-collapse-errors.pdf)). Upper panels show velocity-field
   MSE; lower panels show full Jacobian error. The important counterexample to
   a universal derivative improvement is visible at Ring8's two earliest times.
   All fifteen field errors worsen when clocks are fixed.
2. [Risk polarization](figures/clock-collapse-polarization.png)
   ([PDF](figures/clock-collapse-polarization.pdf)). Black is the measured risk
   change, red is the signed residual-alignment term and blue is finite-change
   energy. Black equals red plus blue, subject to floating-point error. Negative
   alignment alone does not imply that the full finite intervention helps.

Both use exactly the fifteen per-time rows of `clock-collapse.csv`. Equal-time
means appear in the report and CSV, not as extra observations in the figures.
