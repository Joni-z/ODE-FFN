# Professor's proposals: current evidence chain

The target is an architecture improvement, not simply an ODE interpretation.
All current evidence comes from time-conditioned synthetic velocity MLPs.
JiT transfer still requires an adequate ordinary image baseline.

| Link | Evidence or deduction | Remaining test |
|---|---|---|
| Muon reshapes matrix updates | Recorded DeltaP stable ranks rise from about3.08 to32.74 on Product16 at the selected larger AdamW/Muon rates. Full spectra are retained. | Separate reshaping from magnitude and approximate-NS effects. |
| Better update geometry need not control P | Mean P sigma grows3.37 to19.17; shape and scale move together. | Monitor actual A(h), not just raw P. |
| Magnitude alone is insufficient to explain fitting | Same-state norm-matched momentum yields MSE0.04123 versus Muon0.02385 at the larger Product16 rate; sampling ordering differs. | Retain ordinary optimizer controls; the effect is not ODE-specific. |
| Simply shrinking P is harmful in tested runs | Decay lowers local Jacobians but raises error about2.7–3x. | A smaller matrix is not a success criterion. |
| Input-dependent diagonal modulation changes the operator | A(h)=P D(h); dA[v]=P diag(J_d v). Both values and input derivatives matter. | Is useful variation limited by the nonlinear controller or by its output transform? |
| Placement alone is not sufficient | PD/DP/sandwich fail to beat strong controls; PD/DP ranking changes with optimizer. | Cycle1 holds PD fixed and changes one part of D construction. |
| Better candidate fitting is not an FFN advantage | Lowest-field Product16 ordinary control is SwiGLU192, MSE0.01567 and SW2 0.07921. | Require ordinary controls and report sampling tradeoffs. |
| Tied K2 is not supported as the present mechanism | Scalar K1 improves all8 paired field settings; rawK1 Ring8 is a small field lead with worse sampling. | Cycle4 distinguishes the deterministic diagonal features at K1. |
| Coarse sampling is not supported as the frozen discrepancy's cause | Heun32/64/128 retains all six-checkpoint rankings. | Keep primary Heun32 and inspect field structure without sampler-based selection. |

New descriptive evidence: the [row/column decomposition](spectrum/findings.md) finds large row-magnitude growth with much smaller changes in normalized directional coherence. This strengthens the scale question without proving its cause.

The [local-step derivation](muon-metric.md) explains why ideal Muon constrains
raw matrix updates rather than P itself or PD automatically. The
[signed-diagonal contrast](signed-diagonal.md) shows that conditional singular
values alone do not determine the feature map. The [two ODE interpretations](nonlinear-flow.md)
separate entrance-frozen linear dynamics from a state-dependent nonlinear flow.
These are mathematical distinctions; they do not establish a new useful model.

## Series convergence versus perturbation amplification

The original recurrence has factorial denominators. For any finite matrix A,
Q_K(A)=sum_{j=0}^K A^j/j! converges to exp(A) as K grows; no condition
rho(A)<1 is required. This is different from a geometric/Neumann series.
The trained layer uses a finite polynomial, which is defined for every A.
What needs assessment is approximation accuracy and sensitivity at its learned
unit-horizon operator, not convergence of the exponential series itself.

For an arbitrary matrix perturbation E, without assuming A and E commute,

```
DQ_K(A)[E] = sum_{j=1}^K (1/j!) sum_{r=0}^{j-1} A^r E A^(j-1-r).
||DQ_K(A)[E]||op <= Q_(K-1)(||A||op) * ||E||op.
```

Each term in the inner sum has norm at most ||A||^(j-1)||E|| and there are j
terms, giving the bound directly. Integrating this derivative along A+tE gives
the finite-perturbation bound

```
||Q_K(A+E)-Q_K(A)||op <= ||E||op * Q_(K-1)(||A||op+||E||op).
```

For K2 the derivative is E+(AE+EA)/2, with bound(1+||A||)||E||. With d held
fixed, A=PD and E=DeltaP D. If both factors change, their exact operator
increment is DeltaP D+P DeltaD+DeltaP DeltaD. An input perturbation also changes
D(h), requiring the earlier input-Jacobian terms. Full-network sensitivity adds
the other learned layers and can contain cancellations.

The truncation error has the scalar majorant
`||exp(A)-Q_K(A)||op <= sum_{j>K} ||A||op^j/j!`.
Large norms can make a small K a poor approximation without making the series
diverge. These are upper bounds, not measured amplification factors or evidence
that more accurate exponential approximation improves the task. The decay and
K1/K2 experiments already warn against treating either a smaller P or a closer
Taylor approximation as a sufficient learning objective.

## Why A(h) must be measured

For unconstrained diagonal D and an invertible constant diagonal C,

```
P' = P C^(-1),     D'(h) = C D(h)
P' D'(h) = P D(h).
```

The complete entrance-frozen polynomial is unchanged, even though the singular
values of P can change. An unconstrained affine-output controller can implement
this rescaling through its final weights and bias: with d=.6+.02g, choose
W2'=C W2 and b2'=C b2+30(C-I)1. Fixed sigmoid bounds generally break this exact
family symmetry; do not claim it for every bounded controller.

Consequently, interpreting P spectrum as the learned dynamics is incomplete.
Cycle1 logs conditional A(h) spectra and full local layer Jacobians. Even those
are local: derivative terms involving J_d and the surrounding nonlinear network
are not determined by the eigenvalues of A alone.

## What the first cycle can distinguish

The bounded affine controller uses essentially the same parameter budget as the
nonlinear MLP (32 fewer parameters per solve). Matching its value and Jacobian
at the zero hidden anchor avoids an arbitrary constant initialization shift;
it does not make the initial functions identical at all hidden inputs.

The affine controller can also grow to rank64 while the original logit Jacobian is capped at32. It reallocates controller capacity; a factorized-linear follow-up is needed to separate rank from nonlinear features. This confound was recorded before reading candidate endpoints.

The affine output keeps the identical MLP tensors but removes sigmoid saturation
and the(.2,1) constraint. It changes those two coupled properties together, so a
positive outcome would need a follow-up that separates them. The earlier failed
K1/Adam range screen remains relevant negative evidence; an interaction with K2
or Muon must be demonstrated, not assumed.

Direct D(h)=diag(h), if selected after cycle1, is a distinct function family:

```
F(h)=h+P(h*h)+.5 P[h * P(h*h)].
```

It supplies quadratic/cubic features without a learned gate. It also changes gate
parameter count and input scaling, so it needs a no-learned-gate bounded counterpart
and strong ordinary controls. This is a prospective branch, not a current result.

## Recent primary literature, screened3November2026

- [Muown, May2026](https://arxiv.org/html/2605.10797v1): the authors attribute observed Muon spectral-norm drift in their language-model experiments mainly to row magnitudes rather than row coherence. This motivates distinguishing scale and directional structure; it does not establish the same mechanism for our P or license a new optimizer bundle.
- [Musec, September2026](https://arxiv.org/html/2609.11655v2): replaces flattening by clipping of the momentum spectrum, with convergence assumptions and LLM stability experiments. This offers a later direction-versus-magnitude test; no such method is implemented in cycle1.
- [Spectral Scaling Laws of Muon, June2026](https://arxiv.org/abs/2606.04058): abstract screening highlights layer-dependent momentum spectra and incomplete approximate NS orthogonalization. We must measure the actual update, not assert every singular value equals1. Full paper review is pending before adopting its scaling claims.
- [GLU Variants,2020](https://arxiv.org/abs/2002.05202): gated products and changing gate nonlinearities are established architectures. This is background attribution, not presented as a frontier recommendation or a novelty claim for PD.

These papers motivate questions, not expected wins or a theorem of FM-specific
architectural advantage. Our algebra above is a direct derivation; our controlled
training must determine whether it is useful.

## A bound connecting both professor suggestions

Let d(h) be the vector diagonal, A=P diag(d), H=||h||2, p=||P||2,
a=||A||2 and b=||J_d(h)||2. Differentiating the exact entrance-frozen
K2 computation gives the local bound

```
||J_F(h)||2 <= 1+a+a^2/2 + H*p*b*(1+a).
```

For any direction v, dA[v]=P diag(J_d v), hence
||dA[v]||2<=p*b*||v||2. Apply this to
v+A v+(dA[v])h+.5[(dA[v])A h+A(dA[v])h+A^2v].
The inequality follows by triangle/submultiplicative bounds. It is loose and local;
it does not predict training error or certify the surrounding network.

This separates four quantities: P scale, the actual conditional A scale,
controller input sensitivity, and the hidden input magnitude. Muon and different
D constructions can affect different pieces. Lowering just p is neither necessary
nor sufficient for quality, as our decay intervention already illustrates.

For the bounded-positive parent, every d lies in(.2,1), so
`.2*||P||2 <= ||P D||2 <= ||P||2` at every entrance state.
The lower bound follows from P=(PD)D^(-1). A decreasing gate can compensate only
within this finite range; the exact arbitrary scale symmetry described above is
available to unconstrained affine-output D, not to this fixed bounded family.

## Why the optimizer and D parameterization cannot be treated independently

The same-function diagonal rescaling also changes optimization coordinates.
Holding gate parameters fixed while differentiating P, let P'=P C^(-1),
D'=C D. Since dP=dP' C, the loss gradient transforms as G'=G C^T.
For a positive uniform C=cI, the ideal canonical partial-polar update (zero on null singular subspaces) obeys polar(cG)=polar(G).
Thus using the same Muon LR gives the same P-coordinate displacement in the
rescaled coordinates, but its contribution to the conditional operator changes:

```
original DeltaA_P = -eta polar(G) D
rescaled DeltaA'_P = -eta polar(G) cD = c DeltaA_P.
```

The two starting functions are identical, yet the same nominal LR produces a
different effective-operator step. For a nonuniform diagonal C, even the update
singular directions can change. The practical finite-NS, epsilon, clipping and
momentum version must be measured rather than assumed exactly scale-invariant.

This is a frozen-gate partial-update calculation; the full training step also
updates controller/backbone parameters. SGD/Adam can also depend on coordinates,
so it is not a proof of a unique Muon defect. It explains why Muon and the choice
of D must be crossed experimentally, and why matching raw parameter-update norms
alone does not establish matching effective dynamics or overall training speed.

Capacity and speed claims also require the [explicit budget controls](budget-controls.md). Direct input gates reduce parameters but still reuse P twice; same parameter count does not imply same dense MACs or latency.

## A concrete limit of P-only constraints

Even perfectly benign P spectra do not certify the input-dependent map. In one
dimension choose P=-1 and

```
d(h)=.2+.8 sigmoid(10(h-1)).
```

This has the professor's positive gate range and can be represented by the original
SiLU controller: SiLU(h)-SiLU(-h)=h, so two hidden units give logit100(h-1)
with tau10. At h=1, d=.6 and d'=2. P has spectral norm1, condition number1 and
a strictly negative eigenvalue. Yet

```
F2(h)=h*(1-d(h)+d(h)^2/2)
F2'(1)=.58+(-1+.6)*2=-.22.
```

The scalar entrance-frozen map folds locally despite the benign P. Replacing
the polynomial by the entrance-frozen exact exponential does not remove the
issue: H(h)=h*exp(-d(h)) has H'(1)=-exp(-.6)<0. Every fixed-duration linear
solve is invertible, but selecting duration from the initial input is a different
composite map. The example is not a claim that our trained checkpoints fold at
these values or that folding is harmful to field prediction.

This gives a precise reason to study D construction and J_d alongside P spectra,
rather than relying only on eigenvalue caps or more accurate exponentials.
The external FM sampler integrates the learned vector field; a hidden block is
part of the predictor, not the complete transport map. Invertibility of each hidden
map therefore is not automatically the right objective-specific design target.

## Why eigenvalues alone are insufficient even with a constant gate

Consider the real two-dimensional matrix

```
P = [[-1, M], [0, -1]],    s=.6.
```

Both eigenvalues stay at -1 for every M. However,

```
Q2(sP) = [[.58, .24 M], [0, .58]]
exp(sP) = exp(-.6) * [[1, .6 M], [0, 1]].
```

Their operator norms grow without bound with M. This is nonnormal amplification,
not an error repaired by increasing Taylor order. The example can be embedded
in the64-dimensional P. It does not assert that our trained matrices equal this
Jordan block. It explains why we record singular values and nonnormality in
addition to eigenvalues, and why a strictly negative eigenvalue constraint is
not an operator-norm guarantee for a general learned real P.

A singular-norm bound does give ||P^k||2<=||P||2^k, but still does not bound the
input-gate derivative term in the full layer Jacobian without additional assumptions.
Our experiments then supply the other necessary caution: reducing the measured
local norm need not improve field fitting or sampled distributions.

The [native P-only coordinate case](spectrum/coordinate-step.json) checks the uniform rescaling with the trained momentum and actual five-BF16-NS implementation. The effective DeltaA ratio is2.9983 at the same LR and0.9994 after inverse LR compensation, with the stated frozen-auxiliary/no-clipping scope. This is experimental support for the coordinate coupling, not a full-training invariance guarantee.

## Order and feature interpretation after cycle3

The registered K1 contrast now completed: the original scalar improves validation
field error in all8 pairings. Direct h improves in all3 Ring8 pairings, but not
consistently on Product16. The tied K2 term is not supported as the source of
the current benefit. This conclusion is specific to these models/20k budgets.

For raw D, K1 is exactly F(h)=h+P(h*h), J=I+2Pdiag(h). The branch uses quadratic
features while the identity path retains the odd component. In particular,
F(h)-F(-h)=2h for one such block. The composed FFN with SiLU need not have that
symmetry. K2 adds a tied cubic expression, not a free cubic feature map.

A minimal next D family is d(h)=h+c, giving

```
F_c(h)=h+P(h*h)+c*P*h,
J_Fc(h)=I+P*diag(2*h+c).
```

The fixed c=1 contrast introduces a tied linear feature without new parameters.
Constant d=1 removes input dependence: F(h)=(I+P)h. Bounded d=.2+.8sigmoid(h/10)
replaces the unbounded quadratic feature by h*d(h); its Jacobian is
I+Pdiag(d+h*d'), d'=.08sigmoid(h/10)(1-sigmoid(h/10)). Together these distinguish
the role of input dependence, range and linear/quadratic mixture. They change
initial functions and effective scale, so outcomes cannot attribute a gain to
one abstract property independently of those changes.

The [frozen solver diagnostic](solver-results/findings.md) retains ordering at
Heun32/64/128. Thus coarse integration is not supported as the explanation for
the observed Product16 raw-K2 sampling deficit. Finite field errors and local
Jacobians still do not identify a transport cause or certify the global bound.

## Why constant versus shifted D is a useful minimal contrast

Let L(h)=h+Ph and S(h)=h+Ph+P(h*h), with identical P. At h=0,
L(0)=S(0)=0 and J_L(0)=J_S(0)=I+P. Their local difference is precisely quadratic:

```
S(h)-L(h)=P(h*h),
D^2 S(0)[u,v]=2P(u*v),     D^2 L(0)[u,v]=0.
```

Thus these two cycle4 arms match the initial value and first derivative at zero
while adding a structured curvature tensor with no new parameters. They do not
match at all hidden states, or in their complete network predictions. The
surrounding time-modulated LayerNorm also shifts the actual state away from zero.

For bounded d=.2+.8sigmoid(h/10), the coordinate feature has the local expansion

```
h*d(h)=.6h+.02h^2-h^4/60000+O(h^6).
```

This arm is mostly a linear feature near zero, unlike the raw quadratic branch.
The first solve therefore contrasts different local feature mixtures, not just
different eigenvalues of a fixed operator. The second solve sees the output of
the existing SiLU and need not share the first solve's input distribution.
We should inspect both layers' recorded state/term statistics before claiming
that one local expansion describes the complete trained FFN.

The constant-D full FFN is exactly `(I+P2) SiLU((I+P1)h)`: a bias-free ordinary
width64 two-linear-layer FFN under W_i=I+P_i. With no selected-matrix weight decay,
the affine translation does not change its parameter gradients or the optimizer
update rule; it does change the initialization compared with a standard randomly
initialized W. Therefore a constant-D success would not establish a benefit from
input-dependent dynamics. This control must be interpreted as an ordinary FFN
within the inherited initialization, not marketed as a new ODE method.

## Longer-budget outcome

[All8 matched80k trajectories](cycle6-results/findings.md) are complete. Raw K1 improves field MSE over matched Swi64 by9.4%/2.9% on Ring8/Product16, but has worse sampling on both. It does not beat the wider ordinary frontier. The earlier short-budget Ring8 lead over Swi192 disappears. Neither80k nor the original20k is asserted to be converged. Next isolate singular-value shape from contraction on frozen checkpoints before adopting another P constraint.

## A controlled spectral-shape question

[Cycle7](cycle7.md) preserves each P's Frobenius norm while moving its singular
values toward a common value. This changes shape without equating smaller
weight energy with improvement. Its paired scalar contraction matches the
resulting leading singular value, exposing a different amplitude explanation.

The shape path has a useful local identity. For nonzero P=U diag(sigma) V^T,
put q=||sigma||2, alpha=q/sqrt(r), and S=sum(sigma). Then at k=0,

```
P_shape'(0) = alpha U V^T - (alpha S/q^2) P
<P, P_shape'(0)>F = 0.
```

Differentiate the normalized singular-value blend to obtain this expression;
the zero Frobenius inner product means no first-order change in energy. It
does not mean no first-order change in the network function or loss. A learned
anisotropy can encode useful directions, and PD changes even though D is frozen.
This is a checkpoint sensitivity question, separate from training a constrained
model. No spectral penalty or modified optimizer follows automatically.

[Cycle7 completed54frozen cases](cycle7-results/findings.md): flat spectra at fixed Frobenius energy do not generally improve a learned function. The lone small Ring8/raw20k benefit reverses at80k. This separates a failed post-hoc repair from the still-untested possibility of adaptation during constrained training. Neither result is implied by Muon flattening its updates.
