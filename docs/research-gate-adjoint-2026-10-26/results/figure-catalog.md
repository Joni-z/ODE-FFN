# Figures and saved data

- gate-adjoint-comparison.png/pdf: six paired cold endpoints and within-family omission/full ratios on Product16. The field-error panel has a log axis; the ratio panel has a linear axis and a reference at1.
- learning-curves.png/pdf: all saved oracle-validation checkpoints for each family's full and omitted functions. All six final values match the independently replayed comparison.csv.
- p-and-clipping.png/pdf: six original ODE P largest singular values summarized by mean and min/max range at each snapshot; the right panel retains every500-update clipping window. The spectral band is matrix heterogeneity, not uncertainty, and does not bound a full-network Jacobian.
- context-controls.csv: all ten previously audited Product16 direct/A/B controls, including the wider direct SwiGLU. These are reused context, not additional independent training.
- symmetric-skew.csv: all five complete512-input derivative panels for each of six native models.
- finite-differences.csv and parameter-finite-differences.csv: independent native partial-Jacobian value calculations at perturbed inputs/parameters, compared to derivatives of the explicit function.

All three saved figures were inspected. The independent analyzer and report-only plotter are preserved separately from the immutable training source; report generation does not rerun models.
