# The hidden covariance constraint did not help this screen

All18 Stage9 cells completed under the registered5000-update protocol. Neither nonlinear new arm passes the baseline-sensitive continuation rule. The family stops here: no temperature/head sweep, gamma clamp, cap or K2 extension is selected. See [the frozen protocol](protocol-covariance.md) and [the operator derivation](covariance-operator-mathematics.md).

| Field or reference | Gaussian oracle MSE | Ring8 oracle MSE | Product16 oracle MSE | Ring8 Jacobian error | Product16 Jacobian error |
|---|---:|---:|---:|---:|---:|
| Original vectorK1 | .00377666 | .01943755 | .29524142 | .91532 | 53.73663 |
| Tied covariance | .00508143 | .02606104 | .32127269 | 1.11877 | 54.64110 |
| Untied decoder | .00486313 | .01858477 | .29773239 | .71009 | 51.21013 |
| Static tangent | .00533175 | .05479328 | .30593871 | 2.21450 | 54.09005 |
| Small SwiGLU | .00430438 | .01518420 | .09083060 | .39370 | 22.59627 |
| Full SwiGLU | .00392798 | .01622070 | .10823417 | .57103 | 25.27259 |

The untied arm improves Ring8 relative to the original vector model by4.39%, but still has22.4% higher oracle error than small SwiGLU. The tied arm is40.2% worse than the exactly initialized untied control on Ring8. It also loses on product16 and the Gaussian control. The tested covariance tying therefore has no supported practical advantage. Nonlinearity improves Ring8 versus the static tangent, but the practical references show why that alone does not establish a useful ODE architecture.

At dim2, tied and tangent have69576 parameters, close to small SwiGLU69570, but80192 linear MACs versus the small reference67904. Untied has81864 parameters/80192 MACs versus full SwiGLU81792/80000. Captured training wall times for the new arms are61–68s per cell versus27–38s for the two references. These include the scheduled diagnostics and concurrent execution; they are not an isolated latency benchmark. Quality already fails, and no speed advantage is established by the smaller tied parameter count.

## The mathematical property did hold

For the tied update, the derivation gives a sufficient one-step decrease of its internal E=-Phi whenever max(-gamma,0)*s_max<2. On the recorded final probes, all36 tied/tangent layer records have a positive sufficient margin (minimum.785), and the largest measured E change is−.653. Thus the intended internal energy property is present while the FM quality remains worse. This separates a verified mathematical property from the desired empirical benefit.

E is not the FM loss, is not shared between the two solves or across blocks, and need not be bounded below. For example, U=0,gamma=1,D=.6I makes F(h)=1.6h: E decreases while repeated updates expand without bound. The actual gate derivative, normalization and output map also prevent transferring a hidden-field theorem to the complete velocity. The negative readout does not disprove the independent-Gaussian posterior covariance identity; it fails to validate this particular hidden-coordinate implementation of it.

## Evidence and next identification question

Native Windows27 tests and the RTX4060 CUDA forward/backward check passed before launch. Frozen-source audits pass108 checkpoints and90 validation diagnostic arrays, including initial U/P pairing, all surrounding tensors and the tied/untied decoder equality. The six-paper assumption memo distinguishes independent Gaussian coupling, OT coupling, hidden-field geometry and known associative-memory algebra. Curated audit/metadata files are under evidence/covariance_*/; raw collection/replay state is in collection.json. Historical Report1 is untouched.

Before another architectural constraint, identify how much of the remaining difference is attainable fitting versus noisy-target optimization and finite-bank generalization. Conditional-gradient variance can be derived from the same posterior covariance. An oracle-supervised training comparison must first distinguish the analytic population teacher from the exact empirical-bank teacher; they need not define the same objective. This is the next diagnostic question, not a proposed rescue of the stopped covariance family. The full architecture goal remains unachieved.

Selected raw evidence is now SHA256-verified and extracted locally. Mac replay independently reloads all18 final checkpoints and reproduces all90 common validation field/Jacobian arrays; max forward discrepancy6.67e-15 and derivative4.98e-14. The replay uses frozen source and recorded runtime buffers and does not regenerate sampling trajectories.
