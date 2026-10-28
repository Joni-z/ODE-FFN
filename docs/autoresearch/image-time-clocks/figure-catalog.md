# Figure catalog

| Figure | Fixed source and interpretation |
|---|---|
| [All endpoints](results/figures/all-endpoints.png) | `comparison.csv`; all eleven native/control/new rows, FID and weighted validation MSE. Zero-baseline bars preserve the different scales. |
| [Clock calendar](results/figures/clock-calendar.png) | All six rows of `calibration-pooled.csv` and all 774 rows of `calibration-profile.csv`. Shading is native token/image SD, not model uncertainty. |
| [Time panels](results/figures/time-panels.png) | `time-panels.csv`; the three new/native K1 states, old constant and three ordinary controls. All 55 scores, including remaining context rows, are retained numerically. |
| [Fixed examples](results/figures/fixed-examples.png) | All eleven models on the same first eight starts, indices 0–7. No output is selected by quality; images are coarse 32-pixel examples and are not independent evidence for a FID decomposition. |

Each basename has a PDF export. All four PNGs were inspected for readable
labels, complete row/source ordering and clipping. The example display uses
the registered common pixel conversion; inference raw states are not clipped.
