# Adding a bias does not close the original ODELayer gap

Neither static bias nor conditional affine forcing passes the registered screen. Both remain close to the homogeneous parent, slightly worsen its nonlinear field errors, and retain a large deficit against SwiGLU. Conditional forcing improves over static bias by only **0.32% on Ring8** and **0.040% on Product16**, far below the required additional 5% contrast. No bias-completion lead or conditional-bias mechanism is established; this fixed setting stops.

## The question and controlled difference

The original value projection `P h` has no bias. This study tested that missing component directly while keeping the two original K1 maps, middle SiLU and existing vector controllers. Each solve was one of:

```text
parent:       h + D(h) P h
static bias:  h + D(h) P h + c b
conditional:  h + D(h) P h + D(h) b
c = shift + scale/2
```

Both new arms start at `b=0` and the existing constant clock `D=cI`. Their initial functions and all parameter responses match, including under a nonzero receiving head. Common tensors, buffers and random draws also match the parent; the parent has no new bias columns. The contrast between the two bias arms is `(D-cI)b`, whose input derivative is `diag(b) J_s`. These identities motivate the control without predicting that it will learn better.

Formal controllers have hidden width 32, with model width 64 and depth 3. Each bias arm adds 384 parameters and no dense MACs: Product16 parent 83,664 parameters versus 84,048 with bias, all 81,984 MACs. SwiGLU128 and GLU128 each use 83,598 parameters and 81,792 MACs. The budgets are close. Data, objective, time sampling, 5,000 updates, initialization references and Heun32 remain fixed.

## Outcomes

| Model | Gaussian oracle MSE | Ring8 oracle MSE | Product16 oracle MSE |
|---|---:|---:|---:|
| ODE parent | .00377155 | .01992964 | .26615519 |
| Static bias | .00376464 | .02011860 | .26917801 |
| Conditional forcing | .00376476 | .02005406 | .26907141 |
| SwiGLU96 | .00418822 | .01825776 | .10916077 |
| SwiGLU128 | .00392725 | .01529629 | .09232577 |
| Sigmoid GLU128 | .00371720 | .02714816 | .15806040 |

Conditional forcing is 31.10% worse than the best generic Ring8 field result and 2.914 times worse on Product16. Its Jacobian error is 1.645 and 2.108 times the best generic result, respectively. Ring8 SW2 remains within the 10% guard, but Product16 SW2 is 23.43% worse. Static bias similarly fails. Small Gaussian gains against the parent cannot satisfy a nonlinear improvement rule. [The full table](comparison.csv) and [independently recomputed screen](screen-decision.json) retain every comparator and cost.

The [learning curves](figures/affine-forcing-learning.png) nearly overlap for the three ODE arms. This fixed-budget failure does not establish an architecture-wide capacity limit.

## The learned bias is active, but its conditional contribution is small

At the final Product16 checkpoint, conditional-arm bias norms average .30885 across the six solves. The total bias RMS averages 0.659% of the receiving-input RMS. Its conditional correction `(D-cI)b` averages 0.207%; the mean and centered parts average 0.171% and 0.107%, respectively. On Ring8, conditional correction and centered variation are 0.143% and 0.080%. These are averages of separately normalized internal statistics, not a whole-network norm decomposition.

The local conditional-bias Jacobian norm is nonzero: its six-solve mean is .006758 on Product16. Raw P maximum singular values average 1.6744 for the parent, 1.6976 for static bias and 1.6940 for conditional forcing. [Complete trajectories](forcing-trajectories.csv), [checkpoint matrices](P-forcing-checkpoints.csv) and [coordinate records](forcing-coordinate-trajectories.csv) preserve the variation behind those averages. Active bias does not imply benefit; raw P cannot certify the full field.

## Fixed-weight removal separates dependence from training benefit

At each trained conditional checkpoint, replace every `D(h)b` by `c b` while keeping all tensors fixed and recomputing downstream gates. Across the five native panels, Product16 field MSE changes .25904571→.25910647, a 0.023% increase; Jacobian error changes 50.51308→50.52177. Ring8 field MSE instead improves .01663957→.01632610 while Jacobian error slightly worsens. Gaussian field error increases 2.02%. [All fifteen panels](affine-to-static.csv) remain visible.

An independent implementation keeps all modes conditional and subtracts the live correction through forward hooks; it reproduces the altered fields and complete input Jacobians within float64 roundoff. This shows limited output dependence under these saved checkpoints. It is neither retraining nor a generation evaluation, and removing `(D-cI)b` also removes a possible mean shift. The separately trained static arm is the appropriate training control.

All 18 cells, 108 checkpoints, 90 native arrays and 18 endpoint replays pass audit. All 54 historical generic saved states and 90 data-RNG comparisons match. The [appendix](stats-appendix.md) records exact sources, monitoring and independent intervention checks. The narrow conclusion is that adding this bias does not solve the current gap; no forcing-strength, clock, width, head or higher-order rescue is selected.
