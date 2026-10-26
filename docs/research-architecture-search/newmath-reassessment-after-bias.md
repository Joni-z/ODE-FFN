# After the bias test: what the original block is learning

**No new architecture is selected.** The bias test is negative, and the original block still loses to the practical SwiGLU reference on both nonlinear tasks. The useful new distinction is between its constant-clock width64 core and the computation spent adapting that core. The existing checkpoints do **not** support calling the trained model a small perturbation of a constant-clock network. They establish some optimization sensitivity, but do not identify a representation lower bound or the remaining cause of the gap.

All numbers below use already completed native results. The only proposed further measurement is one finite, read-only clock-collapse diagnostic, described at the end. It is not a new training arm or permission to reopen stopped settings.

## Exact normal form, with the second clock evaluated at the right state

For the homogeneous vector K1 parent, let `c=.6`, `phi=SiLU`, and

```text
A=I+cP1,                 E1(h)=(D1(h)-cI)P1
B=I+cP2,                 E2(y)=(D2(y)-cI)P2
y=phi((A+E1(h))h).

H(h) = (B+E2(y)) phi((A+E1(h))h).
H0(h) = B phi(Ah).
```

This is exact, with no small-clock approximation. D2 is computed from the actual post-SiLU state y; it cannot be replaced by D2(h) or by its value at `phi(Ah)` without changing the expression.

At constant clocks, P1 and P2 are unrestricted square matrices, so `P1=(A-I)/c` and `P2=(B-I)/c` realize every pair A,B. H0 is precisely a **bias-free width64 two-linear SiLU FFN**, inside the unchanged normalized, time-conditioned residual backbone. It is not an exponential flow at K1. The identity in each solve changes coordinates and initialization, not the set of possible static A,B. An identity-centered P initialization and an ordinary FFN initialization still need not train alike.

The adaptive part is coupled to the same matrices: `E1=(D1-cI)(A-I)/c`, and similarly for E2. The core weights and modulation receivers are therefore not independently adjustable. This is a real algebraic restriction, but it supplies neither a wide-network approximation lower bound nor proof that removing the coupling would help. The zero-order, additive-generator and feature-sharing comparisons already tested related changes without meeting the practical screen.

### The exact budget allocation

Per FFN, width64 and controller hidden32 give:

| Computation | Dense MACs | Parameters |
|---|---:|---:|
| Two square P products | 8,192 | 8,192 |
| Two controllers, each64→32→64 | 8,192 | 8,384 |
| Original vector K1 total | 16,384 | 16,576 |
| Ordinary SiLU128 | 16,384 | 16,576 |
| SwiGLU nominal128, actual85 | 16,320 | 16,554 |

Half the original FFN's dense computation is in controllers. Those controllers contain nonlinear features, so they are not inert overhead; their features modulate existing64 receiving coordinates rather than supplying128 ordinary SiLU channels. There is no general theorem that this is a better or worse use of the budget. The adaptive model is not simply a width64 ordinary MLP, and no function-class inclusion between the full adaptive family and these generic controls is established.

The existing comparison illustrates the limitation of a width-only explanation:

| Completed reference | Ring8 primary MSE | Product16 primary MSE |
|---|---:|---:|
| Original vector K1, Stage23 | .019930 | .266155 |
| Ordinary SiLU128, Stage19 | .033714 | .153892 |
| SwiGLU128, verified across stages | .015296 | .092326 |

SiLU128 has exactly the parent's parameter/MAC budget. It improves Product16 substantially but worsens Ring8. These are native-family comparisons with different hidden initial functions, not a pure width intervention. SwiGLU does better on both. The earlier expanded diagonal-flow family did not repair the SiLU reference and remains stopped.

## A finite bound, and why it does not license a static approximation

For one input h, suppose `||E1(h)||2 <= e1` and `||E2(y)||2 <= e2`. SiLU satisfies `||phi(z)||2 <= ||z||2` and has the conservative global Lipschitz bound L=1.25. One proof uses `phi'(u)=sigmoid(u)+u*sigmoid'(u)` and `|u*sigmoid'(u)|<=1/4`: `cosh²(|u|/2)>=1+u²/4>=|u|`.

Writing `y0=phi(Ah)` gives the exact difference and a finite bound:

```text
H-H0 = B*(y-y0) + E2(y)*y

||H(h)-H0(h)||2
 <= [1.25*||B||2*e1 + e2*(||A||2+e1)]*||h||2.

ei <= max_channels|si-c| * ||Pi||2
||A||2 <= 1+c||P1||2;   ||B||2 <= 1+c||P2||2.
```

The maximum must cover the inputs on which the bound is asserted. Sample maxima provide a sample bound, not a global guarantee. The second maximum concerns native y, which suffices for this single-block comparison; it does not cover new hidden states after simultaneously replacing other blocks. Small raw clock deviations alone are insufficient if P or the receiving maps amplify them.

Stage23's final parent logs use the same256 saved probes, mixing input states and times. Means below average six solves; they are not six experimental repetitions.

| Task | Mean P spectral norm | Mean clock | Range of layer mean clocks | Mean within-channel sample std | Range of `max|s-c|*||P||2` across solves |
|---|---:|---:|---:|---:|---:|
| Gaussian2D | 2.0063 | .6383 | .6231–.6476 | .03197 | .5222–.7423 |
| Ring8 | 1.8161 | .6938 | .6293–.7732 | .09102 | .6432–.7677 |
| Product16 | 1.6744 | .7858 | .6718–.9414 | .08981 | .5412–.8391 |

The full Product16 probe clock range reaches .2025–1.0, and Ring8 reaches approximately .2000–.99995. The conservative block error coefficients above range3.00–3.96 on Product16 and3.55–4.21 on Ring8. These loose **upper bounds do not prove large errors**, but they fail to certify a small perturbation. Conversely, nonzero clock variance does not identify its useful contribution to the output: the needed joint products `(D-cI)Ph`, their directions and downstream propagation are absent from marginal duration/P summaries. For example, a varying clock on a row with `(Ph)_i=0` on the data contributes nothing there.

A small forward perturbation would not by itself imply a small derivative perturbation. The local Jacobian is

```text
J_F = I+cP + (D-cI)P + diag(Ph)J_s.
```

The last term needs a separate bound. There is even a counterexample within a two-SiLU-feature controller. In one dimension take P=1, `k=eps^-2`, and

```text
g_eps(h)=eps*[phi(k*(h-1)+1)-phi(k*(h-1)-1)-1]
s_eps(h)=.2+.8*sigmoid(g_eps(h)/10).
```

The controller has exactly the permitted affine–SiLU–affine form. Its g is uniformly O(eps), hence `s-c=O(eps)`; the forward correction is O(eps) on any fixed bounded input interval. But at h=1, g=0 and

```text
s_eps'(1)=.02*[phi'(1)-phi'(-1)]/eps,
```

which diverges. Thus even uniform closeness of clock **values** does not bound `diag(Ph)J_s` without control of the controller derivatives. No local bound transfers automatically through the two solves, LayerNorm and the complete velocity head.

## What the accumulated negative results identify

The Stage23 static/conditional biases give Ring8 MSE .020119/.020054 and Product16 .269178/.269071, both behind their parent and generic controls. Replacing learned Db by c*b in the frozen affine checkpoint changes Product16 five-time field MSE from .259046 to .259106, and Ring8 from .016640 to .016326. These panel metrics are distinct from primary random-time validation. The small biased-branch intervention does not measure the original `DPh` contribution or justify treating the full clocks as inactive.

The function-preserving scale8 clock-coordinate experiment is stronger evidence about **one part of the optimization issue**: its function class and initial function are unchanged, yet fixed-budget errors improved by11.1% on Ring8 and33.6% on Product16 under the unchanged optimizer. So representation alone cannot explain every observed difference. That setting still failed SwiGLU and is closed; this result is not permission for a temperature, learning-rate or longer-budget rescue.

The converse does not follow. Repeated failures of different parameterizations do not prove the original function class cannot fit the target. The Stage16 tangent experiment allowed independent pointwise vector-gate responses and found zero local residual floors, but shared-controller fits on16 calibration paths overfit the other16 Product16 paths for both ODE and SwiGLU. It was neither a global approximation test nor a reliable class-specific optimization diagnosis. Frozen-head fitting, conditional-noise removal and regional error breakdowns likewise do not optimize the complete original function class to a certified minimum.

A useful logical separation is:

- **Established:** parameter coordinates and optimizer response can materially change fitting within the same ODE family; the existing generic frontier remains better.
- **Not established:** that a width64 static core is the actual bottleneck, that gates are unused, that controller saturation is the dominant cause, or that the remaining gap is irreducible by the original function class.
- **Closed settings:** scale8, feature sharing, zero-order placement, nonlinear drift, diagonal activations, state-clock integration and bias completion. None is reopened by this normal form.

## At most one further diagnostic: finite clock collapse

If a further read-only measurement is selected, ask only: **does the trained state-dependent modulation improve the complete learned function relative to its own learned constant-clock core?** This differs from the earlier anchor-frozen derivative diagnostic, which preserved every forward prediction and removed selected input derivatives.

For each final homogeneous parent checkpoint, capture each layer's native per-channel mean clock `sbar_i` on the existing256 probe inputs, using no target labels. Fix these six vectors once. Compare the native model with one globally defined counterfactual replacing every clock by its corresponding sbar. This collapses the two solves to constant matrices `I+diag(sbar_i)Pi` and removes sample/time-dependent clock variation while retaining its learned mean. Do not use per-time fitted means, oracle fitting, a selected interpolation coefficient, refitting or retraining.

Replay both complete fields and their actual input Jacobians on every existing five-time validation panel, restoring runtime buffers and recomputing downstream states. The native replay must reproduce the saved arrays. For the finite output change `delta_v=v_collapsed-v_native` and residual `r=v_native-v_oracle`, report the exact decomposition

```text
MSE_collapsed-MSE_native
 = 2*mean(r*delta_v) + mean(delta_v^2),
```

and its Frobenius-Jacobian counterpart. This separates a large perturbation from one aligned with error reduction. No linearization, unconstrained oracle gate fitting or tiny-bank kernel regression is involved. The probe and validation sets are already exposed; neither is called a fresh holdout.

Worse collapsed error would demonstrate useful finite adaptation **at that checkpoint**, not prove that a retrained width64 model cannot compensate or that the controller cost is optimal. Similar or better collapsed error would weaken the claim that the learned state variation helps this frozen function, without proving an expressive limitation or selecting a cheaper architecture. A collapse to c=.6 instead would confound state dependence with a potentially large learned mean shift and is not this proposed test. No outcome alone meets the architecture goal.

## Evidence paths

The Stage23 calculations use [native comparison](analysis-affine-forcing/comparison.csv), [clock trajectories](analysis-affine-forcing/forcing-trajectories.csv), [P checkpoint summaries](analysis-affine-forcing/P-forcing-checkpoints.csv) and [the completed bias intervention](analysis-affine-forcing/affine-to-static.csv). The static-core budget and generic numbers are independently retained in the [Stage19 report](analysis-diagonal-flow/analysis-report.md). Optimization and local-response scope follow the [clock-coordinate readout](clock-coordinates-readout.md) and [Stage16 report](analysis-controller-tangent/analysis-report.md).

SHA256 of the four Stage23 input CSVs, respectively:

```text
comparison:            fc4d1bdbfcfe0b71d1c4eae6c8910110113c24ded33cc19d3e4e2cd8069f1682
clock trajectories:    859d2210a5b40a8f2e28c9c2ad7cb1b01dd33eae67675dc8bda34ec41e7c181a
P checkpoint summaries:2f9cd668fb7e44905f3e3767cf418f184d39d2b4820ba7194abfdf4d2a892c71
bias intervention:     2a445d264dc201338d631bdbfffc3518fc8486064cee95e50a417bdd0904df1e
```

No new checkpoint evaluation or training was performed for this memo; its numeric calculations summarize the existing audited CSVs. The finite clock-collapse diagnostic is proposed only and has not been executed.
