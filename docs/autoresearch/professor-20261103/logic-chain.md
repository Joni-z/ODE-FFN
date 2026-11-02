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
| Better candidate fitting is not an FFN advantage | Strongest Product16 ordinary control remains MSE0.01567/SW2 0.07921. | Require ordinary controls and report sampling tradeoffs. |

New descriptive evidence: the [row/column decomposition](spectrum/findings.md) finds large row-magnitude growth with much smaller changes in normalized directional coherence. This strengthens the scale question without proving its cause.

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
