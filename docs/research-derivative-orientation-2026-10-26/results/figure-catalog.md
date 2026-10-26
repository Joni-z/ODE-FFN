# Figure catalog

- field-comparison.png/pdf: all30 final full-validation field MSEs; per-task log axes; orange denotes ODE and blue generic controls.
- crossover-errors.png/pdf: log10 of fixed-weight switched/native MSE for both training orientations and all three families; zero means no penalty. These are not retrained crossovers.
- learning-curves.png/pdf: all30 saved validation trajectories; colors identify families and line styles identify readouts. Every step5000 value agrees with comparison.csv.
- p-trajectories.png/pdf: means and min/max bands of the six ODE P largest singular values at steps0,1000,2500,5000 for direct/A/B. Bands reflect matrix heterogeneity, not statistical confidence or complete-network stability.

Figures were inspected at their saved resolution. The first two originate in the independent analyzer; the last two are produced from saved metrics/spectra without new model evaluation.
