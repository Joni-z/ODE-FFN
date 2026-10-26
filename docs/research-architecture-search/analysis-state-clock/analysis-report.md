# Stage21: reevaluating the original clock inside a solve

Midpoint does not pass the registered screen. Recomputing the original vector clock at an intermediate state produces a real, learned correction, but it does not improve the quality/compute tradeoff. It is worse than Euler on both nonlinear tasks, worse than frozen K2 on Product16, and substantially behind SwiGLU at comparable computation. This setting stops without a further solver, temperature, width or head variation.

## The question and its control

The original layer freezes its input-dependent clock during a conditional linear solve. With `f(z)=D(z)Pz` and `first=f(h)`, the tested maps are:

```text
Euler:     h + first
Frozen K2: h + first + .5*D(h)*P*first
Midpoint:  h + f(h + .5*first).
```

The midpoint's nonlinear Taylor coefficient includes the clock derivative term `.5*diag(Ph)*Js(h)*f(h)`. Frozen K2 correctly omits it because its auxiliary field keeps D(h) fixed. These are different auxiliary dynamics, not a corrected bug in the professor's conditional Taylor layer. Two solves, the intervening SiLU, the original P/controller and all clock constants remain unchanged. No external FM solver, loss, label, output-head or spectral intervention is combined.

All ODE arms start with the same tensors and constant clocks. Frozen K2 and midpoint therefore have identical initial functions and P/input derivatives. Their nonzero-head controller responses differ: midpoint accesses controller features at the intermediate state. Their formal zero output head initially hides that difference. The comparison is not a theorem that one optimizer geometry is better.

## Fixed endpoints

Primary validation oracle MSE after 5000 updates, lower is better:

| Arm | Gaussian2D | Ring8 | Product16 |
|---|---:|---:|---:|
| Euler | 0.003771549 | 0.019929636 | 0.266155201 |
| Frozen K2 | 0.004403100 | 0.023842773 | 0.301849081 |
| Midpoint | 0.004331943 | 0.023265256 | 0.321165100 |
| SwiGLU96 | 0.004188221 | 0.018257763 | 0.109160768 |
| SwiGLU128 | 0.003927254 | 0.015296290 | 0.092325770 |
| SwiGLU256 | 0.004468461 | 0.016592856 | 0.058669595 |

On **Ring8**, midpoint improves frozen K2 by only 2.42%, short of the required 5%, while it is 16.74% worse than Euler and 52.10% worse than SwiGLU128. Its Jacobian error is .903356 versus frozen K2's .993957, but the better SwiGLU Jacobian error is .468551. Its SW2 is .248764 versus .226360 for the better SwiGLU. A small improvement over the weaker frozen control does not establish useful progress.

On **Product16**, midpoint is 6.40% worse than frozen K2 and 20.67% worse than Euler. Its primary error is 5.474 times SwiGLU256, the close dense-computation reference, and 3.479 times SwiGLU128, the near-parameter reference. Its Jacobian error is 57.3012 versus 16.0550 for SwiGLU256; SW2 is .136523 versus the better baseline's .095722. Gaussian primary error also regresses 10.30% against the better SwiGLU. The complete curves show a persistent nonlinear gap rather than an endpoint chosen against a favorable earlier checkpoint.

Only midpoint was eligible. The registered rule required a nonlinear primary gain of at least 10% over all three SwiGLUs and 5% over both ODE controls, with other-task and Jacobian/SW2 guards. [Independent arithmetic from the raw terminal results](independent-review.json) agrees with the [main screen](screen-decision.json): no qualifying task and no nomination.

## Was the feedback active?

Yes. In the midpoint arm, final initial-to-midpoint clock-change RMS averages .01983 across the six solves on Ring8 and .04003 on Product16. The finite correction's RMS relative to each solve's input averages 3.13% and 4.81%; the separate clock-Taylor diagnostic is 2.56% and 4.16%. These numbers describe measured internal probes and are not full-network error bounds. They rule out a dormant-controller explanation of this particular negative result.

The frozen quadratic term in the same midpoint models is larger, at 25.44% and 23.94% of input RMS. This does not justify increasing the new term: changing its amplitude would be an unregistered intervention. The raw P spectra also do not isolate the failure. Product16's final mean largest singular value is 1.6856 for midpoint versus 1.6749 for frozen K2 and 1.6744 for Euler, despite their different errors. No full-field stability claim follows from those norms.

Euler and frozen K2 also have saved midpoint-clock and correction probes. Those are counterfactual measurements; those models do not execute the extra controller stage during their native forward pass. The Taylor diagnostic is not explicitly added during any model's forward pass.

## Cost, initialization history and disposition

Product16 midpoint uses 83,664 parameters and 131,136 dense MACs. SwiGLU256 uses more parameters, 133,068, but slightly fewer dense MACs, 130,752. SwiGLU128 has almost the same parameter count as midpoint, 83,598, with only 81,792 MACs. Thus neither the near-compute nor near-parameter comparison supplies a practical advantage. Recorded training-loop times are 47.93s for midpoint, 34.98s for frozen K2, 29.75s for Euler and 23.42s for SwiGLU256; these include diagnostics and concurrent CPU execution, not isolated inference timing.

The recent Stage17 vector parent already used constant-paired clocks. The new Euler arm has exactly the same initial tensors, buffers and Torch/data RNG, so constant initialization is not a new difference from that historical control. Final states are numerically close but not bit-identical; the tiny endpoint differences are not an additional independent result.

The local nonlinear-ODE derivation and controller-response difference remain valid. This fixed implementation does not turn them into improved FM approximation. The evidence closes the proposed midpoint setting; it neither proves a global limitation of state-dependent ODEs nor selects another architecture. [The audit appendix](stats-appendix.md) and [figure catalog](figure-catalog.md) retain every control, curve and scope limit.
