# Current Report2 figures

Both current figures come from [make_professor_figures.py](../make_professor_figures.py).
Every plotted scalar is bound to its exact primary CSV row and source hash in
[professor-evidence.json](../data/professor-evidence.json). No model runs or
new sample generation are used.

| Figure | Purpose | Observation and limit |
|---|---|---|
| professor-muon | Product16 original scalarK2: update stable rank and learned P spectral norm over20k; AdamW.001 versusMuon.02 | Broader updates coexist with growing P. Six-matrix means, not global stability bounds. Both series fit the full axis range. |
| professor-diagonal | Latest constant-D/ISRU-D/free-ISRU trajectories under both registered optimizers, with selected Swi192 reference | Optimizer ordering changes with the feature response. SolidMuon/dashedAdamW/dottedSwi; log-y. No best-checkpoint or convergence claim. |

The full latest endpoint table includes all three SwiGLU widths and both
metrics; the selected Swi192 curve does not replace that complete comparison.
There are no replicate error bars. The earlier ode-mainline and CIFAR figures
remain preserved, with their historical source bindings, but are not displayed
in this revision. The visual-review receipt is valid only for its PDF hash.
