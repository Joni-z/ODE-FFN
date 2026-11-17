# Evidence map for Report 3

Source SHA256 values and full-precision numbers are in
[data/evidence.json](data/evidence.json). This map supplies implementation and
cohort detail for the results in the PDF.

| Report claim | Source | Comparison represented |
|---|---|---|
| High-rank Muon updates can coexist with concentrated, poorly conditioned weights | [Weight/update analysis](../../docs/research-assessments/2026-11-15-professor-execution/spectrum-findings.md) and its `endpoints.json` | Product16, six-matrix means at20k; selected AdamW/Muon settings retained |
| Better Ring8 fitting can accompany growing asymmetric error | [Orbit decomposition and frozen C8 intervention](../../docs/research-assessments/2026-11-15-professor-execution/findings.md) | 15k/20k ISRU-D; rotation-balanced orbit MSE differs from the original finite validation probe |
| C8 projection repairs mode imbalance across architectures | Same source, `results/metrics.csv` and the unchanged case figures | Each checkpoint before/after projection on identical banks, eight predictions per call. The replayed Ring8 cohort is separate from the older unavailable Lenovo run |
| ISRU's particular response is not necessary for these results | [Matched scaled tanh](../../docs/research-assessments/2026-11-15-professor-execution/findings.md), [eight-response confirmation](../../docs/research-assessments/2026-11-15-response-screen/current-comparison.md) | Table2 contains the matched K1 ISRU/tanh comparison; the wider screen is summarized without changing the confirmation shortlist |
| Sharing P helps the fixed-work stage comparison | [Sharing](../../docs/research-assessments/2026-11-16-ode-mechanism/sharing-results/analysis-report.md), [mean-preserving coordinates](../../docs/research-assessments/2026-11-16-ode-mechanism/coordinate-control/analysis-report.md) | Table4 uses the coordinate-control common banks. Coordinates partly recover Ring8 quality; Product16 still favors sharing |
| Prescribed coefficients show no consistent sampling advantage | [Completed coefficients](../../docs/research-assessments/2026-11-16-ode-mechanism/completed-comparison.md), [activity and convergence](../../docs/research-assessments/2026-11-16-ode-mechanism/primary-findings.md) | Table5 uses the coefficient/state/stage confirmation cohort; it is not pooled with Table4 |
| RK4 improves internal accuracy without confirmation generation gains | [Matched training](../../docs/research-assessments/2026-11-16-ode-mechanism/rk4-results/analysis-report.md) | Eight P calls per solve; each learned field has its own high-accuracy reference. The46x/22x ratios concern normalized squared error |
| An unfitted analytic correction predicts the complete field change | [Registered equations](../../docs/research-assessments/2026-11-16-ode-mechanism/learned-defect-field-protocol.md), [complete result](../../docs/research-assessments/2026-11-16-ode-mechanism/learned-defect-field-results/analysis-report.md), [hidden bias](../../docs/research-assessments/2026-11-16-ode-mechanism/solver-bias-results/analysis-report.md) | Four models and two4096-input streams; Figure4 includes all eight refined, qualified cells |
| The same-weight midpoint generation benefit survives Heun2048 | [Population resolution study](../../docs/research-assessments/2026-11-16-ode-mechanism/population-resolution-results/analysis-report.md) | Both streams and all eight banks. This completed measured effect differs from the unfinished analytic target-loss forecast |
| Ordinary controls limit practical transfer of the result | [Same-bank comparison](../../docs/research-assessments/2026-11-16-ode-mechanism/ordinary-population-results/analysis-report.md), its `independent.json` and statistics appendix | Table6 shows native midpoint and ordinary models on both streams atHeun2048. All48source rows/32contrasts remain in the source bundle |
| Native SwiGLU leads widened frozen-D K1/K8 on CelebA128 | [Completed image comparison](../../docs/celeba128-wide-20261114/continuation-160/matched-results.md) | FID50k at100/120/140/160, matching parameters and evaluation; refreshed state flow has not been tested here |

The two case images are copied byte-for-byte. The solver and field-forecast
plots are regenerated from saved summaries. No new statistical test or model
computation is performed by the report builder. Original numerical failures
and later refinements remain in the source analyses.

The professor's readability comment is addressed by separating three pages
of explanation from four supporting pages, defining hidden/global time before
using it, and stating the purpose and observation for each main experiment.
This is a draft report; there is no finalization record.

## Interpretation and final-stage decision, 18 November

The external commentary motivates a sharper conclusion, not new numerical
claims. Sharing is the leading structural explanation within the tested family;
the mean-preserving control prevents calling it the unique isolated cause.
The CelebA result establishes a loss for frozen-D blocks, not that freezing the
diagonal caused the loss. Missing gating, gradient saturation and smoothing by
numerical error remain hypotheses. The field-prediction score is the defined
zero-change-relative squared-error score, not centered explained variance.

The proposed direct recurrence uses eight updates of `y += P q(y) / 8`, sharing
P and q across steps. This matches the midpoint model's eight P calls and unit
internal duration. It is also explicit Euler: the experiment tests the value
of midpoint staging beyond simple shared residual iteration, not whether the
same computation can be called an ODE. Full training-cost matching must also
include backward and optimizer work. The primary comparison fixes every model's
existing width and uses the complete 20k midpoint trajectory as its budget.
Continue the ordinary checkpoints with their original recipes until cumulative
training FLOPs match, including their already completed training. Train the new
tied residual model to the same budget. The forward-MAC ratio is not the full
training-cost ratio. Widening an ordinary FFN while fixing the update count
changes the architecture and optimization problem; it is not included in this
comparison. Actual cost accounting and continuation settings belong in the run
protocol before launch. Existing trained models remain references;
the new control and ordinary continuation require a separately registered run
decision. No activation search or immediate image launch is added.

The [earlier recurrence study](../../docs/research-recurrent-depth-2026-10-26/results/analysis-report.md)
used the original scalar K2 blocks and a different 5k training protocol. It
cannot substitute for this direct control of the current state-flow family.
All four supporting pages were rendered again: no stray axis fragments appear.
The actual Figure4 caption error was corrected from vertical to horizontal.
