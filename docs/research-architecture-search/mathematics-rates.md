# Mathematical candidate: frozen channel-dependent rates

This is a candidate mechanism, not an established advantage. The professor's scalar s(h) gives `A(h)=s(h)P`; all fixed-rate effective maps are polynomials of P, so they commute and share invariant eigenspaces. The complete network need not have commuting Jacobians. Nonlinearities, normalization and depth already supply further adaptation, so the restriction is not a proved network bottleneck.

Replace only the scalar rate by a vector s(h), set D=diag(s(h)), and keep the original recurrence:

```text
term0=h
termk=D P term(k-1)/k
F_K(h)=T_K(DP)h.
```

D is computed once and frozen across the inner recurrence. This is the Taylor map of a frozen linear generator DP, not a high-order solver for the autonomous nonlinear field D(y)P y. External generation time remains distinct. A uniform vector rate exactly recovers the scalar layer.

## Why the actual matrix product matters

For K2,

```text
F(h)=h+DPh+0.5 DPDPh.
```

It is generally incorrect to replace the last term by `0.5 D^2 P^2 h`:

```text
(DP)^2-D^2P^2 = D[P,D]P.
```

Thus this change allows relative directional responses and generally removes the commuting-family restriction. It introduces no independent second matrix and keeps shared powers, but it is a richer input-dependent generator family. It could improve fitting or simply be unnecessary flexibility.

## Gate Jacobian contribution

Let Js be the d-by-d derivative of the vector gate and q=Ph. For K1:

```text
J_F=I+DP+diag(q) Js.
```

For K2:

```text
J_F=T2(DP)+[diag(q)+0.5 diag(PDq)+0.5 DP diag(q)] Js.
```

For a two-layer gate MLP with hidden width r, rank(Js)<=r. A scalar gate contributes rank<=1 per layer, whereas a width32 vector gate can contribute rank<=32. This is a statement about one layer's differential, not proof that the output function class needs those ranks. In particular our full network has six scalar gates and a 2D output in the original tasks. Higher-dimensional tests and K1 controls are needed to assess this mechanism.

The target Jacobian for our Gaussian source and independent linear path is a scalar contraction plus scaled posterior covariance. It can require several directional responses. That motivates testing directional adaptation; it does not require hidden P or DP to equal that data-space Jacobian, nor prove this design uniquely appropriate for flow matching.

## Parameter and compute controls

At d=64:

- Scalar gate hidden64: 64*64+64+64+1=4225 trainable parameters.
- Vector gate hidden32: 64*32+32+32*64+64=4192 parameters.

The difference is33 parameters per layer. Both use the same number of P applications for a given K; the vector gate uses64 fewer forward linear MACs per layer. Gate width changes to keep the budget matched and must be disclosed. Initial surrounding weights and P are paired by preserving the scalar-reference initialization draws, while actual gate networks differ. No zero-gate or special spectral initialization is added.

K1 vector gating is a crucial rival: it could explain all gains through generic coordinate gating without P-squared. SwiGLU is the practical rival, already providing coordinate-wise input gates. If K2 does not beat K1 on quality/compute, do not attribute a result to higher-order flow structure.

Input-dependent discretization has substantial prior art: [Mamba Sections3.2 and3.5](https://arxiv.org/html/2312.00752v2). Gate granularity also matters in other settings, e.g. [Gated Attention](https://arxiv.org/html/2505.06708v1). These are reasons to include rival explanations, not novelty claims or proof of this FFN's effectiveness.
