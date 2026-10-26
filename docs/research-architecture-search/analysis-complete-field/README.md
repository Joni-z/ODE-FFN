# Complete-field constraint diagnostic

The measured violations are a small part of the ODE parent's Jacobian error, particularly on Product16. These results do not support selecting symmetry/PSD enforcement as the main next mechanism on the strength of the constraint alone.

All values below use the original velocity-Jacobian squared Frobenius error. Denoiser-space distances were divided by `(1-t)^2` before averaging over the five times and 512 anchors at each time. “Structural distance” is the squared distance to the symmetric PSD cone for `I+(1-t)Jv`, expressed in these velocity units.

| Dataset | Model | Original Jacobian error | Structural distance / error | Pointwise projected error | PSD-violating anchors |
|---|---|---:|---:|---:|---:|
| Ring8 | Vector ODE parent | 0.82858 | 8.07% | 0.75811 | 1.05% |
| Ring8 | SwiGLU96 | 0.50875 | 11.47% | 0.44964 | 0.74% |
| Ring8 | SwiGLU128 | 0.62464 | 9.38% | 0.56596 | 0.12% |
| Product16 | Vector ODE parent | 50.00194 | 1.24% | 49.30226 | 25.51% |
| Product16 | SwiGLU96 | 26.75389 | 3.32% | 25.66726 | 36.64% |
| Product16 | SwiGLU128 | 23.95783 | 5.61% | 22.25959 | 39.57% |

For the ODE parent, negative symmetric eigenvalues alone account for 0.118% of Ring8 error and 0.210% of Product16 error; most structural distance comes from the skew part. Product16 has many violating anchors, but their squared contribution is small. Moreover, the stronger SwiGLU model has more violations while fitting the target Jacobian substantially better. Violation counts by themselves are therefore a poor guide to the current performance gap.

The pointwise projection leaves most of the parent's error: 91.49% on Ring8 and 98.60% on Product16. Its projected Jacobian error is still larger than the better unmodified SwiGLU's error on both datasets. This does not bound what a retrained constrained architecture could achieve. It says that correcting only the recorded violations, through this nearest-cone projection, does not remove the main existing discrepancy.

The decomposition is `original error = skew distance + negative-spectrum distance + projected-target error + cone cross term`. The cross term is nonnegative for a symmetric PSD target. All 15360 inequalities pass within the recorded floating-point tolerances; tiny negative slacks are numerical roundoff, retained in the CSVs. All six original aggregate errors reproduce the native Stage17 summaries, and all three arms use identical saved anchors and oracle arrays.

The projected matrices are independent at each saved point. They do not constitute a velocity model, an integrable Jacobian field, a trained improvement or a sampling result. No model or training run was added.

- [Aggregate error components](figures/complete-field-error-components.png)
- [Time profiles](figures/complete-field-time-profiles.png)
- [Aggregate values](aggregate.csv), [all 30 panels](per-panel.csv), [all anchors](per-anchor.csv)
- [Numerical summary](summary.json) and [input/script/protocol provenance](provenance.json)
- [Protocol registered before outcome inspection](../protocol-complete-field-diagnostic.md)
