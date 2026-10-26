# Figures

- gradient-cosines: all time-gradient pair similarities for the oracle calibration, averaging the two halves. Shows local disagreement in both ODE and ordinary controls. Does not establish a causal bottleneck.
- transfer-slopes: local risk decrease on the opposite half for every checkpoint and both label definitions. Shows transferable temporal sensitivity, including generic controls. No error bars; exact fold values are in aggregate.csv.
- finite-versus-local: local expected decrease against actual finite change, all oracle-calibrated folds and methods. Shows that the declared fixed radius overshoots even in the shared control, blocking a finite improvement or rejection claim.
