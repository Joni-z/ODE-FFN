# Cycle12 figure catalog

All four figures use audited frozen-path outputs, six models on each law and 2,048 shared sources per model. PNG is for GitHub reading; the matching PDF retains vector text/lines. There are no training repeats, fitted trend lines or significance markers. Error bars are absent because the plots describe the complete fixed bank; reporting-only conditional source precision is tabulated separately in [conditional-mc.csv](results/conditional-mc.csv). All four PNGs were visually inspected for clipping, legends and agreement with the tables.

## Signed temporal accounting

- Files: [PNG](results/figures/temporal-accounting.png), [PDF](results/figures/temporal-accounting.pdf).
- Purpose: test the idea that late field error dominates final sampling discrepancy.
- Data: [bins.csv](results/bins.csv), tangent scope, four registered eight-step bins.
- Variables: color/text give each bin's signed contribution divided by its model's paired tangent endpoint MSE. Both laws use one diverging color scale. Rows share sources, not independent experimental units.
- Caption requirements: define signed accounting and common-source oracle coupling; distinguish it from SW2 and causal interval ablation.
- Observation: thin frame K2 has 49.69% in the second quarter and 1.01% in the final quarter; original K2 has 32.68% in the last quarter.
- Interpretation: late held-out field residuals alone do not select where a useful architecture must improve path behavior.
- Checks: four shares sum to one per row; negative contributions remain allowed; no best interval is selected.

## Normal-space restriction

- Files: [PNG](results/figures/normal-restriction.png), [PDF](results/figures/normal-restriction.pdf).
- Purpose: separate a shared output restriction from hidden-block explanations.
- Data: [endpoints.csv](results/endpoints.csv) and [steps.csv](results/steps.csv), thick law.
- Variables: left bars split normal paired MSE into T and N−T energies, both divided by 126; right lines give actual path normal RMS, including final endpoints. The oracle line uses the same cap/grid.
- Caption requirements: T is the fitted target-normal/output-null intersection, dimension 63 for all models; the fixed-readout identity does not constrain all normal coordinates identically.
- Observation: T contains 91.53–93.15% of normal paired error; every learned path has less final normal spread than the oracle path.
- Interpretation: a P-only explanation of this shared component is invalid at fixed readout. Relative tangent failures remain to be explained separately.
- Checks: stacked bars recover normal MSE; no model's subspace is substituted for another's; normal RMS is not a covariance eigenvalue or an empirical W2 value.

## Finite defect propagation

- Files: [PNG](results/figures/finite-defect-gain.png), [PDF](results/figures/finite-defect-gain.pdf).
- Purpose: show why local step discrepancies and endpoint influence can differ.
- Data: [steps.csv](results/steps.csv), tangent scope, every step of each model/law.
- Variables: x is external step-start time; y is square root of mean transported-defect energy divided by mean local-defect energy. This is a ratio of RMS quantities, not a mean of source-wise ratios.
- Caption requirements: finite defects are transported by oracle suffixes; the displayed gain is not a Lipschitz constant, hidden-P spectral radius or worst-case stability certificate.
- Observation: input-frame K2 peaks at 16.77 on thin data and 16.52 on thick data; later gains approach one.
- Interpretation: a temporal diagnosis needs propagation as well as local error. The plot does not justify a new spectral penalty.
- Checks: denominators use the registered nonzero-energy rule; all six families appear; no smoothing, peak selection or interval search changes the decision.

## Prespecified source paths

- Files: [PNG](results/figures/fixed-source-paths.png), [PDF](results/figures/fixed-source-paths.pdf).
- Purpose: make actual sampled trajectories and source coupling visible.
- Data: private audited `output/raw/{law}-{family}.npz` and oracle path arrays, projected onto the saved 128×2 basis.
- Variables: 2D tangent coordinates for sources 0, 512, 1024 and 1536 on both laws; learned endpoints are marked; plus signs are the eight target component centroids.
- Caption requirements: indices were registered before outcomes; curves are actual learned/oracle paths, not the diagnostic suffix hybrids. Normal behavior is not visible in this projection.
- Observation: maps can follow different paths from the same source; the aggregate centroid disagreement is measured separately on all sources.
- Interpretation: a paired endpoint metric attaches meaning to source assignments that distribution metrics can ignore. Four paths do not estimate mode coverage or establish a winner.
- Checks: equal coordinate scales/aspect ratio, all families and oracle in legend, no outcome-based replacement of examples.
