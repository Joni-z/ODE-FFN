# Where the product16 derivative error occurs

This reuses the exposed20000-update heldout evaluations of the unchanged scalar/vector ODE and two SwiGLU references. It is descriptive error analysis, not fresh confirmation. The program verifies array hashes, common inputs, the rotated oracle Jacobian and exact reconstruction of every total error. Raw measurements and deterministic case selection are in analysis-operator/posterior-regions.json; the program is scripts/diagnose_posterior_regions.py.

For each latent coordinate, q=t*mu*z/c is half the posterior sign log-odds, c=(1-t)^2+t²*.04. The ideal derivative is

```text
lambda(z,t) = (t*.04-(1-t))/c
              + t*(1-t)*mu²/c² * (1-tanh(q)²).
```

The first part contracts within resolved modes; the second can expand a coordinate where the target sign remains ambiguous. Partition coordinates by the exact derivative sign, then separate diagonal error from off-diagonal coupling error in the known latent basis. Orthogonal rotation preserves the squared Frobenius total, so these contributions add exactly to the saved full-field Jacobian error.

| Time | Model | Expanding-coordinate diagonal error | Contracting-coordinate diagonal error | All off-diagonal error |
|---|---|---:|---:|---:|
| .5 | Scalar ODE | 8.537 | 4.193 | 1.622 |
| .5 | Vector ODE | 7.418 | 3.081 | 1.158 |
| .5 | Full SwiGLU | 2.053 | .619 | .819 |
| .7 | Scalar ODE | 27.233 | 41.680 | 3.570 |
| .7 | Vector ODE | 23.520 | 30.309 | 3.003 |
| .7 | Full SwiGLU | 5.855 | 5.646 | 2.349 |
| .9 | Scalar ODE | 1.234 | 27.442 | 1.747 |
| .9 | Vector ODE | 1.231 | 19.419 | .970 |
| .9 | Full SwiGLU | .806 | 3.217 | .514 |

Each number is a contribution to the mean squared Frobenius error per anchor, not a conditional mean within its region. Small SwiGLU is also retained in the JSON. At t=.7, only5.31% of coordinate-anchor entries have positive ideal derivative, but their complete output-row errors account for37.8% of scalar and41.7% of vector Jacobian error. Most remaining error still lies in contracting rows. At t=.9, expanding coordinates are only.037% of entries, and contracting diagonal error dominates. Thus “missed rare expanding boundaries” and “insufficient contraction” both matter at different times. No single contraction cap follows from this decomposition.

## Two reproducible cases

Both cases use t=.7, latent coordinate7, mu=1.31544. The first selects the largest ideal derivative among the fixed512-by16 bank, independent of model residuals. It is deliberately a sharp-curvature case, not a typical sample. The second selects the median absolute posterior log-odds among the same bank.

| Selection | Latent state | Posterior P(positive sign) | Ideal derivative | Scalar | Vector | Small SwiGLU | Full SwiGLU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Largest ideal derivative, anchor390 | −.01216 | .44909 | 27.456 | 10.779 | 11.175 | 13.889 | 18.986 |
| Median certainty, anchor232 | −1.00734 | 4.46e-8 | −2.482 | −2.479 | −2.567 | −2.897 | −2.828 |

All four models underestimate the sharp positive derivative in the first case. In the second, the posterior has essentially selected one sign and the derivative is negative. These examples illustrate the target's sign-dependent geometry; they do not substitute for the aggregate errors. In particular, a better derivative estimate at the selected sharp point need not give a better velocity value there: the ideal velocity is−.33644, while scalar/vector/full SwiGLU predict+.03702/+.45800/+.15020. Velocity fitting and derivative fitting remain distinct observations.

The covariance-shaped candidate was registered and launched before this old-bank partition analysis. No threshold or training change was selected from these case values. Any useful future candidate still needs a new independent evaluation bank before acceptance.
