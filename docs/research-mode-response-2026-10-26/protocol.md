# Conditional response shape versus amplitude

Registered before fitting this diagnostic, 26 October 2026. The previous goal turn made progress through a primary-paper synthesis. Current inspection verifies that local-adjoint Phase A is terminal and negative, all prior sources/results remain intact, and Lenovo is offline. The overall useful-ODELayer objective is active and unachieved.

## Question and fixed population

The existing paired-redraw diagnosis places almost all of the original direct ODE's Product16 excess in its conditional own-coordinate response. A raw-affine account already failed. Does that nonlinear error mostly follow the known posterior-mean response with incorrect coefficients, or does it require a different response shape?

Use exactly the seven final models and five times (.1,.3,.5,.7,.9) from the completed coordinate-dependence study. Reuse all4096 independent row pairs and their native/hybrid latent-coordinate predictions. No new network evaluation, training, checkpoint selection or target-bank substitution. The original direct ODE is compared primarily with direct SwiGLU128/192/384; all three gradient-readout arms remain visible.

At time t, coordinate q has two Gaussian mixture components with means +/-t*mu and within-component variance d=(1-t)^2+t^2*sigma^2. Here sigma^2=.04 and mu is the recorded coordinate location. Gaussian conditioning gives

```text
v*(q,t) = alpha*q + beta*tanh(kappa*q)
alpha = (t*sigma^2-(1-t))/d
beta = (1-t)*mu/d
kappa = t*mu/d.
```

This is a privileged diagnostic basis. The model is never given mu, the rotation, the oracle nonlinearity or a fitted correction during training or sampling.

## Two nested fits, with held-out rows

Let eA,eH be errors at independent redraws of all other coordinates, retaining the same q. Their conditional mean is m(q), and g=(eA+eH)/2 is an unbiased conditional-response observation. Split row pairs0:2048 and2048:4096. Fit g on one half, evaluate on the other, then swap. Fit each time/coordinate independently, with all seven models sharing the same design matrix:

- affine basis: [1,q/SD(q)]
- full basis: [1,q/SD(q),tanh(kappa*q)].

Use float64 SVD least squares, rcond1e-12. Record condition numbers/ranks and every coefficient. A rank failure is an analysis failure, not permission to change the basis. For each out-of-fold correction b, record

```text
R(b) = eA*eH - (eA-b)*(eH-b)
     = 2*b*g - b^2.
```

Conditional on its training fold, E[R]=E[2*b*m-b^2], the decrease in conditional-response risk. The same finite R is the decrease in the average of the two squared errors. Dependence on other coordinates is unchanged by subtracting b(q). This is not an achieved deployable predictor or an oracle-free architecture gain.

Average outputs and times while keeping each original row pair intact. Retain signed reductions, including negatives. Compare the full-basis and affine reductions separately; the incremental nonlinear reduction is R_full-R_affine.

## Primary convention and uncertainty

For each direct generic reference define positive original-ODE excess DeltaT from the paired-redraw total risk. Compute the differences in full-basis reduction and incremental-nonlinear reduction, each divided by DeltaT. Do not replace total excess with a favorable denominator.

Use256 paired bootstrap resamples, generator seed0, independently resampling rows within each original fold. Refit both bases in every bootstrap, preserving fold membership and all coordinates/times/models of each sampled row. Report95% percentile intervals conditional on these fixed learned functions.

- A majority coefficient-response account survives only when the lower interval bounds of BOTH shares are at least .50 for ALL three references.
- It is rejected as a majority explanation if every full-basis share has upper bound below .50.
- Otherwise report mixed/inconclusive. No extra basis functions, coordinates, times, resamples, thresholds or tuned correction are selected after outcomes.

This convention localizes error and does not identify its optimization/capacity cause. It neither advances a trained architecture nor revives local-adjoint, potential-readout, recurrence or stopped gate variants.

## Case studies, verification and resources

Preselect coordinates0,7,15 and times.3,.5,.7 for display. Show conditional error estimates in16 fixed bins of q/SD over[-4,4], with counts and fitted response curves; retain tails in all numeric risks. Show all seven aggregate results and all five times.

Before fitting real errors, test the exact paired-risk identity, a fully representable conditional error with independent off-coordinate leakage, an orthogonal conditional shape residual, and out-of-fold isolation. Reproduce the analytic oracle arrays and all old T/C/D arrays. Save all input/source/protocol hashes. One local CPU process, bounded memory, single BLAS thread, process/time/memory telemetry. An independent QR implementation must reproduce all original coefficients, reductions,256 refitted bootstrap outcomes and the final decision from saved inputs. Inspect figures before sealing the report. No new training is selected by either outcome.
