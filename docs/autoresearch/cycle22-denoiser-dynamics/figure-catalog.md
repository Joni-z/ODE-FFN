# Figure catalog

Both figures were rendered and visually inspected. Same-stem PNG/PDF versions
contain the same content. Their data are deterministic identities and fixed
probes, not trained-model performance.

| Figure | Data | Interpretation |
|---|---|---|
| [Counterexamples](results/figures/counterexamples.png) | [controls.json](results/controls.json), closed formulas in U5–U8 | Prediction slopes, material derivatives, exact excess risks and the two posterior precision derivatives. Zero covariance-only bars are intentional. The plotted examples distinguish objects and assumptions; they do not validate a new architecture. |
| [All probes](results/figures/all-probes.png) | [point-cases.csv](results/point-cases.csv) | All 8 points at every one of the 5 times for all 3 laws. Gray points show fixed cases with deterministic display offsets; blue lines join their means. Columns use different coordinates/units and are not a quality ranking. |

The [finite-difference table](results/finite-differences.csv) retains both
registered steps and every derivative/PDE residual. The [identity summary](results/identity-summary.csv)
gives all law-level maxima; no failed point or numerical discrepancy is omitted.
