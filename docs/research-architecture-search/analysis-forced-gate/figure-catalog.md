# Stage20 figures

All three figures use complete terminal evidence and were visually inspected. PDF files provide vector exports; PNG files provide previews. No panel displays confidence intervals or seed variation.

1. [Endpoint comparison](figures/forced-gate-endpoints.png) ([PDF](figures/forced-gate-endpoints.pdf)): primary oracle MSE, five-time Jacobian error and saved validation SW2, each divided by the better of the two SwiGLU baselines for that metric and task. Lower is better. The baseline defining1 can differ between rows; Ring8 Jacobian uses SwiGLU96, while Ring8 primary MSE uses SwiGLU128. Near-overlapping bars are resolved numerically in comparison.csv.
2. [Full learning curves](figures/forced-gate-learning.png) ([PDF](figures/forced-gate-learning.pdf)): every recorded oracle-MSE point from0 through5000 updates, with logarithmic y axes and no checkpoint selection. Static/K2/exact and SwiGLU128 nearly overlap; these overlapping curves are retained rather than amplified into an apparent effect.
3. [Gate and projection trajectories](figures/forced-gate-internals.png) ([PDF](figures/forced-gate-internals.pdf)): latent max|lambda| and conditional rates, correction RMS relative to the native gate, and gate/value/output spectral-norm products. Lines are three-block means; shading spans the blocks and is not statistical uncertainty. The full channel table preserves individual eta/lambda entries. Norm products are not whole-network Jacobian bounds.

Endpoint/learning figures are produced by source/analyze_forced_gate.py. The internal figure uses the identical audited CSV inputs and source/render_internal_figure.py, which wraps long axis labels for readability. No numeric values or axis scales were changed by that presentation adjustment.
