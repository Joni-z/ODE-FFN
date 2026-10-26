# Clock coordinates affect fitting, but this scale still loses to SwiGLU

The function-preserving clock-coordinate change improves Ring8 oracle MSE by11.1% and product16 by33.6% relative to the unchanged vectorK1 parent. Its initial functions, full feasible function class, parameter count and linear MACs are unchanged. This establishes a fixed-budget parameterization effect, rather than an improvement from added expressive power.

Neither coordinate arm passes the registered baseline-sensitive continuation rule. The paired arm still has13.8% higher Ring8 oracle error and2.16 times the product16 error of small SwiGLU. This scale is stopped; no temperature sweep, longer-budget rescue or K2 extension is selected. [Frozen protocol](protocol-clock-coordinates.md).

| Arm | Gaussian oracle MSE | Ring8 oracle MSE | Product16 oracle MSE | Ring8 Jacobian error | Product16 Jacobian error |
|---|---:|---:|---:|---:|---:|
| Original vectorK1 | .00377666 | .01943755 | .29524142 | .91532 | 53.73663 |
| Paired coordinates | .00389334 | .01728022 | .19610946 | .59062 | 39.09282 |
| Unpaired tau1.25 | .00390413 | .01597974 | .20337498 | .55346 | 40.84621 |
| Small SwiGLU | .00430438 | .01518420 | .09083060 | .39370 | 22.59627 |
| Full SwiGLU | .00392798 | .01622070 | .10823417 | .57103 | 25.27259 |

The unpaired temperature control is better than the paired arm on Ring8 but slightly worse on product16. It changes the initial duration function, so its behavior is not attributed solely to the paired coordinate effect. Neither new arm beats the smaller, cheaper practical reference on either nonlinear task.

## What the mathematical control establishes

Dividing the last controller affine weights/bias and tau by8 leaves g/tau and every duration unchanged. An inverse maps all learned functions back to the old coordinates. Native CPU tests and an independent4060 nonzero-head forward check are bit-identical at initialization; model RNG, parameters and linear MACs match. At paired parameter values, last-controller parameter gradients scale8 and other gradients are unchanged. A20-step double-precision Adam test with active clipping verifies the lr/epsilon/clip-metric transformation needed to recover exactly equivalent optimizer trajectories. Formal experiments deliberately retain the ordinary optimizer settings for all arms.

Thus the result identifies parameter coordinates and their optimizer response as relevant. It does not prove that the coordinate change reduces conditional noise or that eight-times larger parameter response yields eight-times faster convergence. The actual zero output head makes all backbone responses zero at the first step; the effect can begin after the common head has moved. The derivation and qualifications are in clock-coordinate-mathematics.md. No new function class, flow stability certificate or useful final architecture is claimed.

## P and the clocks took different paths

The plots include complete P-norm and invariant-controller-norm trajectories, rather than only final quality. The following values average over the six internal layers at5000 updates; these layers are not independent experimental replications.

| Task | Arm | Mean P spectral norm | Mean W_last/tau Frobenius norm | Mean sample variation of durations |
|---|---|---:|---:|---:|
| Ring8 | Original | 1.8606 | .6523 | .08359 |
| Ring8 | Paired | 1.7141 | 2.4164 | .15982 |
| Ring8 | Unpaired | 1.7631 | 4.5186 | .17351 |
| Product16 | Original | 1.6948 | .7448 | .09252 |
| Product16 | Paired | 1.7743 | 2.7132 | .11174 |
| Product16 | Unpaired | 1.7985 | 4.7898 | .13081 |

The paired arm begins with the same invariant controller norms and sample variation as the parent. Its learned invariant norms subsequently grow faster, consistent with the coordinate-response argument. P's norm moves in opposite relative directions on the two tasks: the improved Ring8 arm has a smaller average norm, while the improved product16 arm has a larger one. The gain cannot be summarized as uniformly shrinking P. Neither this average nor the clock variation is the full velocity Jacobian or a stability guarantee. The fixed probes include variation in state and time.

## Verification and next question

Native audits pass90 checkpoints and75 common-field arrays, including the exact initialized weight scaling, the unpaired control's unchanged tensors, source/data/configuration hashes and matching data-generator states at every snapshot across all five arms. Default parent and SwiGLU endpoints reproduce the previous bank-mode results. Raw initial/1000/final collection and independent Mac endpoint replay are tracked in collection.json. Historical submitted Report1 remains untouched.

The next step is a read-only study of the actual input features and downstream responses seen by P. It must distinguish structural null input directions, positive-spectrum conditioning and loss-relevant responses before any new whitening or low-rank design is selected. It is specified in next-input-geometry-question.md; no additional architecture is launched by that question. The full useful-architecture goal remains unmet.

Selected raw evidence is now collected, SHA256-verified and extracted. Independent Mac replay reloads all15 final checkpoints and reproduces all75 saved validation velocity/Jacobian arrays; maximum forward difference6.81e-15 and derivative2.76e-14. Recorded runtime buffers are restored. Sampling trajectories and a new independent holdout are outside this replay scope.
