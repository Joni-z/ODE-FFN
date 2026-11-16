# Units, controls and verification

All256 original selection sources and16 coordinates are retained. These are
deterministic evaluations of fixed analytic maps, not independent model fits.
Coordinates, times and error multipliers are repeated within each source;
no confidence interval, hypothesis test or model ranking is inferred from them.
Sensitivity percentiles describe this finite population of coordinate/source
pairs. The exact marginal identity is algebraic and does not require a sample
significance test. Positive/negative endpoint fractions concern a fixed coupling.

The four multipliers and cyclic skew matrix were fixed in the derivation before
running the control. No source or direction was selected to obtain a failure.
The quantile control uses bisection on strictly positive mixture densities and
preserves all input probabilities without clipping. Source and time derivatives
are checked at decreasing finite-difference sizes; every error is in the bound
summaries. The independent verifier reconstructs CDF conservation, transition
ratios, tangent coefficients, exact endpoint probabilities and all four energy
ratios without importing the control implementation.

All three actual author/check/verifier commands exit0. Raw arrays remain outside
Git; the complete hashes and measured check values are in the three summaries.
No new generation scores, training, optimizer or FFN changes are involved.
