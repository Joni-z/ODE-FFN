# Stage 20: a forced-response correction of SwiGLU

The registered family fails the practical screen. The new arms stay close to SwiGLU, unlike the earlier rebuilt two-solve FFNs; the forced-response correction adds no useful improvement over the same-width SwiGLU. Neither adaptive K2 nor the exact response is nominated. This was one fixed five-arm comparison; no rescue setting is selected.

## What this tested

The ordinary gate is b=SiLU(a), with an independent value feature v. We kept both projections and the output matrix, and replaced b by the unit-time response of y'=c*y+b, y(0)=0, where c=tanh(eta)*sigmoid(a). The exact gate is b*phi1(c), with phi1(c)=expm1(c)/c; K2 uses b*(1+c/2). Eta starts at zero, so both begin with exactly the ordinary SwiGLU function and common-parameter gradients. Their complete initial parameter Jacobians also match each other. Static c=.5*tanh(eta) is an active redundant parameterization whose gain can be absorbed into the output matrix.

This tests whether that particular learned gate response helps. The auxiliary solve is not the FM generation trajectory, and the all-orders response changes the gate curve and range. Its comparison with K2 does not isolate external ODE solver accuracy.

## Results

Primary validation oracle MSE, lower is better:

| Arm | Gaussian 2D | Ring 8 | Product 16 |
|---|---:|---:|---:|
| SwiGLU 96 | 0.004188221 | 0.018257763 | 0.109160768 |
| SwiGLU 128 | 0.003927254 | 0.015296290 | 0.092325770 |
| Static response | 0.003927286 | 0.015296200 | 0.092396394 |
| Adaptive K2 | 0.003924789 | 0.015401099 | 0.092542308 |
| Adaptive exact | 0.003924641 | 0.015408046 | 0.092552074 |

Ring 8 provides the clearest control comparison. K2 is 0.685% worse than SwiGLU 128 and 0.686% worse than static; exact is 0.731% worse than both. Both beat the narrower SwiGLU 96 by roughly 15.6%, but the unchanged SwiGLU 128 already achieves that advantage. Their Jacobian errors are slightly better than SwiGLU 128, but remain about 20.2% above the narrower baseline, which has the better Jacobian metric. Neither the primary improvement requirement nor the Jacobian guard passes.

On Product 16, K2 and exact are respectively 0.235% and 0.245% worse than SwiGLU 128. The static control is 0.0765% worse. Exact is 0.0106% worse than K2; on Ring 8 it is 0.0451% worse. These observations offer no support for a higher-terms benefit. The small Gaussian improvement, below 0.07%, does not satisfy the required nonlinear-task improvement.

The registered screen required at least 10% lower primary MSE than both SwiGLUs and 5% lower than static on a nonlinear task, with limits on other primary and Jacobian/SW2 metrics. [The independent calculation](independent-screen-check.json) and [the main analyzer](screen-decision.json) agree. The lack of primary improvement already rejects both arms independently of the Jacobian guard.

## Did the model use the new parameter?

Yes. All eta vectors start at zero and become nonzero. At the final Ring 8 checkpoint, the exact gate's correction RMS is 2.416% of the original gate RMS, averaged across the three blocks; on Product 16 it is 3.020%. The final block maximum reaches 2.873% and 3.346%, respectively. Thus the negative result is not explained by a dormant eta parameter or a zero derivative at initialization.

The recorded conditional rates remain modest. Exact reaches maximum logged |lambda|=.204733 on Ring 8 and.255754 on Product 16. The corresponding maximum conditional |c| is.202377 and.252790. Since phi1(c)=1+c/2+O(c^2), the all-orders and K2 gates are close in this regime. At a common fixed argument |c|<=a, the difference in relative gain is bounded by exp(a)*a^2/6:0.00836 for the Ring 8 range and 0.01371 for Product 16. This scalar bound helps interpret the close responses; it does not bound the error difference between separately trained networks.

The exact gate's logged relative gain spans approximately 0.986–1.108 on Ring 8 and 0.957–1.138 on Product 16. The maximum logged |lambda| across the whole family is.2631, leaving the tanh parameter derivative at least.9308 at recorded states. There is no logged saturation at the imposed rate bound. Gate/value/output norm products also stay close to SwiGLU 128: final block means 4.329 versus 4.354 on Ring 8, and 3.637 versus 3.652 on Product 16. Those products are descriptive projection statistics, not bounds on the multiplicative network Jacobian. Negative gate derivatives are expected for SiLU and are not treated as failed identity-flow monotonicity.

## Cost and conclusion

For Product 16 the new arms have 83,853 parameters versus 83,598 for SwiGLU 128, with the same 81,792 dense MACs. Pointwise response work is additional. Recorded training-loop times are 25.36s for K2,34.97s for exact,38.35s for static and 19.57s for SwiGLU 128; concurrent execution and in-loop diagnostics prevent interpreting these as isolated inference benchmarks.

This fixed training regime learns a small correction without a useful quality gain. It does not prove that all learned gates or all forced-ODE architectures fail, nor that the new tangent is globally redundant. It does remove the present motivation to continue this particular response family with an unregistered temperature, bound, width or order change. Further design needs a different, evidence-backed representational hypothesis; these results do not justify another tuning sweep.

[Complete curves and plots](figure-catalog.md), [numerical and audit details](stats-appendix.md), and [all endpoint values](comparison.csv) accompany this readout. The experiment uses exposed validation tasks; no fresh holdout or architecture confirmation is claimed.
