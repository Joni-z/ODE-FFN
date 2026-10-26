# Figures

All figures are exported as PDF and PNG and have no error bars.

- field-comparison: all 42 endpoints, normalized against the best generic field MSE for each task. Notice that the ODE gap is largest on Product16. This rejects a practical quality claim for the screened setting. Values are exact descriptive endpoint ratios.
- learning-curves: fixed checkpoint errors for the native and recurrent primary families. Notice whether supervision changes convergence rather than selecting a favorable checkpoint. Missing between-checkpoint behavior is not interpolated evidence.
- depth-behavior: frozen shared models with extended and fixed internal horizons. Notice that extra internal work does not reliably improve the learned predictor. This separates numerical refinement from longer computation; neither curve is a retrained architecture.
- p-spectra: maximum singular value across unique P matrices at each checkpoint. Notice the actual bounded numerical trajectories. This is not the Jacobian spectrum of the conditioned whole network, and cannot by itself explain the quality gap.
