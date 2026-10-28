# Figure catalog

All figures use every declared checkpoint/domain/time. Solid curves denote
per-time medians; dotted curves denote per-time 95th percentiles. Percentiles
are spread over native states, not uncertainty bands.

| Figure | Data and interpretation |
|---|---|
| [Clock variance](results/figures/clock-variance.png) | `variance-pooled.csv`; separately centered clocks, raw-component weighting. Fractions describe this finite cohort. |
| [Complete core gain](results/figures/complete-core-gain.png) | `time-profiles.csv`; full 64-by-64 FFN Jacobian, all three blocks. Native coordinate differences prevent interpreting this as a whole-model stability ranking. |
| [Gate feedback](results/figures/gate-feedback.png) | `time-profiles.csv`; relative derivative correction with pointwise native clocks held only for the reference derivative. No forward ablation. |

The same basenames have PDF exports. All three PNGs were inspected after
curation: labels, legends, source ordering and complete curves are readable.
The fixed-case CSV retains every first-clock/image 0/token 0 case at all five
times; the report's late-time example is not an extreme-state selection.
