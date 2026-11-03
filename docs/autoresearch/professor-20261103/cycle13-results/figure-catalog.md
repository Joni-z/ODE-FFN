# Figure catalog

Every figure has PNG/PDF counterparts and was visually inspected. No error
bars: each mark/curve is a fixed registered configuration, not a replicate mean.
A=AdamW, M=Muon; exact rates and both metrics are in endpoints.csv.

| Figure | Purpose and source | Observation and decision | Limits / caption requirements |
|---|---|---|---|
| endpoints | All18validation/test endpoints from native replay | AdamW feature gains do not imply superiority to free ISRU; no continuation | Both metrics lower-is-better; logarithmic axes have truncated baselines; all settings visible |
| learning | Every stored field evaluation, curves.csv | Constant/Muon and feature/AdamW favor different settings | No convergence or selected best-step claim; solid M/dashed A |
| matrix-norm | Update logs, updates.csv | Muon norms grow while nonlinear-feature fitting worsens | Six-matrix means; P versus ordinary W are different parameterizations; no causal claim |
| product-cases | Fixed Adam/ordinary models, axes0/7/15, saved4096samples | Weak constant-D axis0 gap improves with either ISRU architecture | Analytic black density; gap and outside fractions; not full joint recovery |
| ring-cases | Fixed Muon/ordinary models, indices0,4,...4092 | All visible clusters remain; plots alone do not determine score | Gray target, colored model; full4096used for scores; Adam endpoints remain in full metrics |

The first Product16 rendering had overlapping long titles. Layout was corrected
to three-line titles and regenerated without changing any numerical input.
