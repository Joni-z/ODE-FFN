# Measurement and verification scope

The row pair is the independent Monte Carlo unit. Its16 coordinates, five times, seven models and two conditional redraw evaluations remain paired. The target population factorizes in the recorded latent coordinates; the cached training bank has a different conditional target. All conclusions here concern population evaluation of fixed learned functions.

Let eA,eH be conditionally independent errors given q. For an out-of-fold fitted b(q),

```text
C = E[eA*eH] = E[m(q)^2]
R = E[eA*eH-(eA-b)*(eH-b)]
  = E[2*b*m-b^2].
```

The fitted prediction is independent of its evaluation fold. The affine and full reductions are compared as nested diagnostic fits, retaining negative values if present. Their incremental difference need not be positive on every finite panel. No oracle correction is applied during training or sampling. Full-risk and conditional-risk reductions coincide by the finite paired identity, while the other-coordinate component remains unchanged.

The256 bootstrap samples independently draw2048 rows with replacement inside each original fold. Both fits are recomputed for every sample, and all coordinates/times/models share row weights. The reported percentile intervals describe Monte Carlo uncertainty of this fixed-model fitting diagnostic. They are not variation over model training. No resample count, time, basis, coordinate or threshold is adjusted after outcomes.

Numerical safeguards include full-rank float64 SVD fits with rcond1e-12 and all160 recorded full-basis fold condition numbers. A constructed independent discrete population tests a representable conditional error plus off-coordinate leakage, a conditional residual orthogonal to the basis, the paired-risk identity, out-of-fold isolation and weighted QR agreement. All saved analytic oracle and T/C/D arrays replay before fitting.

The separate verifier reconstructs original fields/errors and independently associated basis arithmetic. It uses QR solves and direct corrected-error products, rather than the main SVD solver and expanded risk-reduction formula. It reproduces all35 model/time cases,1120 full-basis coefficient sets and256 refitted bootstrap outcomes. Maximum differences are4.44e-16 for basis values,2.50e-12 for fitted predictions,6.15e-13 for coefficients and1.67e-16 for bootstrap risks. The registered decision agrees exactly.

Main execution completed once in18.8seconds with maximum observed RSS143.5MiB. Independent QR replay took24.2seconds with observed RSS about146MiB. Both used one local CPU/BLAS thread. No GPU was allocated. The main session28825 and independent session91789 exited0. Lenovo's Tailscale peer remained offline on the current host inspection.

The first independent verifier failed before any QR fits because an inappropriate bitwise comparison rejected differently associated floating-point products differing by2.22e-16. The failed checker and failure receipt are preserved. Only the verifier's basis comparison was corrected to a stated1e-15 absolute/relative tolerance; the main source, basis, results, resamples and decision were unchanged and never restarted. A preceding old-artifact inspection also initially imported unavailable psutil through system Python; removing that unused import allowed the read-only validation. This was inspection tooling, not a training or diagnostic failure.

Figures were revised after visual inspection to avoid title overlap and to show binned out-of-fold fits only in occupied bins. There is no extrapolated curve into empty bins. Fixed16bins over[-4,4] and all numeric risks remain unchanged. Both corrected figures were visually inspected before sealing.

Runtime root:
`/Users/mr.z/ode-workspace-prep/architecture-search/mode_response_20261026`

Original inputs:
`/Users/mr.z/ode-workspace-prep/architecture-search/coordinate_dependence_20261026/output-v2`

The runtime contains the frozen source/protocol,43 input bindings, per-time cross-fitted predictions and coefficients, every bootstrap weight/result, condition numbers, telemetry, independent receipt and plot receipt. The source hash is9bb34a68fdd424f6bde35f0c3b1d335cf3fb1755a547acb023403812ab822b46. Prior reports, model code, checkpoints, training inputs and submitted Report1 are unchanged. The useful-ODELayer goal remains active and unachieved.
