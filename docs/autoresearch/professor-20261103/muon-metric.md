# What ideal Muon constrains, and why PD changes the geometry

These are local matrix-step calculations. They are not a proposed new optimizer,
convergence theorem, or proof of a trained FFN advantage. The unchanged official
implementation uses momentum and five approximate BF16 Newton–Schulz steps;
the exact-SVD formulas below describe an idealization, not its exact output.

## The local spectral-norm step

For a gradient or momentum matrix M, minimize the linearized objective
`<M,DeltaP>` subject to `||DeltaP||op<=eta`. If M=U diag(s) V^T, one minimizer is

```
DeltaP=-eta U_r V_r^T,
minimum=-eta sum(s)=-eta ||M||nuclear.
```

Here U_r,V_r span the nonzero singular subspaces. To see this, transform the
inner product to the singular basis: each diagonal entry of U^T DeltaP V has
magnitude at most eta, so the sum is at least -eta sum(s); the stated update
attains it. Nullspace choices may be nonunique. Momentum means M need not be
the current loss gradient, and finite NS does not make all nonzero singular
values exactly equal.

This constrains the update's norm in raw parameter coordinates. It does not
project P onto orthogonal matrices or keep P's singular values near one.
Even an ideal per-step bound only gives
`||P_T||op<=||P_0||op+sum_t ||DeltaP_t||op`; it does not prevent accumulation.
Our observed broad update spectra and growing row magnitudes are compatible
with this distinction. The harmful decay controls show that limiting growth
is not automatically the missing ingredient.

## Holding one diagonal fixed

At a fixed input with invertible diagonal D, A=PD. If instead the desired local
constraint were `||DeltaA||op<=eta`, then DeltaP=DeltaA D^(-1) and

```
<G,DeltaP>=<G D^(-T),DeltaA>.
DeltaA=-eta polar(G D^(-T)),
DeltaP=DeltaA D^(-1).
```

This differs from applying polar to G and multiplying its result by D. A flat
raw-P update spectrum need not be flat after input-dependent column scaling.
Under equivalent coordinates P'=P C^(-1),D'=CD with invertible diagonal C,
G'=G C^T gives G' D'^(-T)=G D^(-T). The ideal fixed-D effective step is invariant
under that coordinate change. Ordinary raw-coordinate Muon need not be, as the
trained native uniform-rescaling case already demonstrates.

This formula is not a drop-in rule for our network: D varies with each input,
raw input diagonals can be singular or arbitrarily close to singular, learned D
also changes in the optimizer step, and whole-field sensitivities include
downstream layers. Naively inverting D would introduce precisely those problems.

## A simpler diagnostic metric, not a method claim

With D frozen as a function and averaged over a chosen hidden-state distribution,

```
E ||DeltaP D(h)||F^2 = tr(DeltaP E[D(h)^2] DeltaP^T).
```

This measures operator changes. The actual K1 increment is P q(h),q=d(h)*h,
so an output-change metric instead uses E[q(h)q(h)^T], generally not diagonal.
The two are different objects. A full predictor metric adds the downstream
Jacobian as well. These distinctions explain why useful diagnostics must state
whether they measure raw weights, conditional operators, hidden maps or the
complete learned field. Choosing one of these metrics as an optimizer requires
new controlled evidence and a novelty review; no such change is made here.
