# A sigmoid gate does not improve the current flow-matching baseline

The fixed sigmoid-GLU control fails its registered screen. At the same parameters and dense MACs as SwiGLU128, it increases primary oracle MSE by **77.48% on Ring8** and **71.20% on Product16**. Its Gaussian error is 5.35% lower, but a Gaussian-only improvement was explicitly insufficient. This was a known generic baseline comparison, not a proposed ODE architecture. The fixed substitution stops here.

## What the comparison isolates

Both networks compute two independent affine projections `a,value` and apply the same output projection. The single change is `SiLU(a)*value` versus `sigmoid(a)*value`. Nominal width128 gives 85 actual gated channels. All biases, backbone, time conditioning, cached-bank pair objective, 5,000 updates, optimizer, sampling and common non-FFN initialization remain fixed. SwiGLU96 supplies the established lower-budget comparison.

Native GLU and SwiGLU128 initial tensors, runtime buffers and random draws match exactly. Their hidden functions differ, as do output-head gradients even though their formal initial velocity outputs are both zero. This therefore tests the whole activation substitution; it does not isolate shape from feature amplitude or optimization geometry. In Product16, each full model has 83,598 parameters and 81,792 dense MACs. Their pointwise activation work differs.

The motivation was deliberately narrower than mode classification. Before training, the [saved-region diagnostic](../posterior-velocity-regions-readout.md) had already shown that most of the ODE gap occurs where component identity is confident. At a fixed time, the Product16 population oracle is an affine response plus signed tanh offsets; an ordinary raw-input GLU represents that map constructively. The unit construction is neither the full normalized, time-conditioned network nor its cached-bank training law. It supplied a testable basis hypothesis, not a performance theorem or analytic initialization.

## Endpoints and learning

| Task | SwiGLU96 oracle MSE | SwiGLU128 oracle MSE | Sigmoid GLU128 oracle MSE |
|---|---:|---:|---:|
| Gaussian2D | .00418822 | .00392725 | .00371720 |
| Ring8 | .01825776 | .01529629 | .02714816 |
| Product16 | .10916077 | .09232577 | .15806040 |

The secondary measures reinforce the negative result. Relative to the better established baseline for each metric, GLU's Ring8 Jacobian error is 2.149 times larger and sampling SW2 is 18.62% worse. Product16 Jacobian error is 47.43% worse and SW2 is 14.54% worse. Both exceed the registered 10% guards. [All endpoints and costs](comparison.csv) and the [independently checked decision](screen-decision.json) retain both budgets.

The [learning curves](figures/sigmoid-glu-learning.png) show a substantial nonlinear fitting deficit before the final update, not merely an isolated final evaluation. Ring8 GLU improves sharply after 3,500 updates, while Product16 GLU remains above both SwiGLUs at every logged nonzero update. Continued improvement does not establish convergence or an irreducible capacity limit; the conclusion is failure at the fixed practical budget. Longer-training or activation-parameter rescues were not selected.

Existing SwiGLU256 results remain visible as [higher-cost historical context](higher-cost-historical-context.csv), with gray hatching in the endpoint figure. Its Product16 oracle MSE is .05866959 at 130,752 dense MACs versus 81,792 for the matched128 arms. It is neither a matched-budget arm nor an independent new repetition.

## The confident-coordinate case also fails

All previously fixed posterior-entropy bins and times are retained in the [regional table](posterior-regions.csv) and [figure](figures/sigmoid-glu-posterior-regions.png). At Product16 t=.7, total field MSE is .448701 for GLU versus .260577 for SwiGLU128. In the lowest-entropy bin, conditional MSE is .350451 versus .169648; this region contributes 87.74% of that same-time excess. Across the five saved panels, confident coordinates contribute 75.28% of GLU's excess, while the highest-entropy bin contributes 1.93%. The signed-offset rationale therefore does not translate into a gain in the very region it was meant to address.

At t=.5 the GLU deficit is distributed across all four bins; their excess-error shares are 40.11%, 26.64%, 28.92% and 4.34%. At t=.1 GLU has a small aggregate advantage, but it does not persist at later times. Regional errors locate prediction mismatch; these networks do not emit posterior probabilities, and the calculation cannot separate slope, offset, time dependence or cross-coordinate causes.

All nine cells are terminal. Native audits cover 54 checkpoints and 45 diagnostic arrays; nine endpoint replays have zero prediction/Jacobian discrepancy. Both SwiGLU budgets reproduce all 36 historical saved states exactly, and 36 cross-arm data-generator comparisons pass. These compatibility reruns do not add independent repetitions. The [audit appendix](stats-appendix.md) records source, initialization, monitoring and scope. No generic lead, ODE mechanism or completed architecture objective is established.
