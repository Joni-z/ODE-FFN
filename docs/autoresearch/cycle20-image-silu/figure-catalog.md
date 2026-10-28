# Figure catalog

All figures were visually inspected after rendering. PNG and PDF versions
contain the same data. Public CSVs retain full precision.

| Figure | Source | Reading |
|---|---|---|
| [Endpoint comparison](results/figures/fixed-comparison.png) | [comparison.csv](results/comparison.csv) | Every new and reused endpoint; log weighted-error axis and ordinary linear FID axis. Points describe fixed banks, not uncertainty. |
| [Learning curves](results/figures/learning-curves.png) | [curves.csv](results/curves.csv) | All77saved cold checkpoint rows. Four native curves dashed, three new ordinary curves solid; frozen interventions have no retraining curve. |
| [Matrix histories](results/figures/matrix-histories.png) | [matrix-history.csv](results/matrix-history.csv) | All six W maps per ordinary model at0/1000/2500/5000; largest singular values only in the plot, full spectra in CSV. These are not P generators or full-network stability bounds. |
| [Generation cases](results/figures/generation-cases.png) | Registered128-step raw banks, first eight fixed starts | Nine endpoints, same ordering and image conversion. Each tile's first row is starts1–4 and second row5–8. No image selection; raw banks stay outside Git. |

Each image has a same-stem PDF in results/figures/. The full45final fixed-time
panels are tabulated in [time-panels.csv](results/time-panels.csv), rather than
selected for the text. Per-image fitting distributions and all18new/reference
contrasts are also retained. The analysis distinguishes measured cost, frozen
interventions, fitting and image feature distance.
