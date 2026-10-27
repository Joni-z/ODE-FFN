# An exact gated baseline, rather than a weaker ODE substitute

This note derives an equivalent representation of ordinary SwiGLU. It is a baseline identity, not a new activation, normalizing-flow principle or demonstrated FM improvement.

## Exact conditional flow

With column-vector convention, write the existing biased FFN as

```text
a(h)=Wg*h+bg; g(h)=SiLU(a(h)); v(h)=V*h+bv
F(h)=U[diag(g(h))*v(h)]+bo.
```

For a fixed conditioning h, take y=(u,v) and the lifted linear ODE

```text
u'=diag(g(h))*v,    v'=0,    (u(0),v(0))=(0,v(h)).
```

Its generator P(h) has only the upper-right diagonal block diag(g). Therefore P²=0 and every term in the exponential beyond degree one vanishes. For any real duration s, the unique global solution is `(u0+s*g*v0,v0)`. At s=1 and u0=0, readout `U*u+bo` is exactly F(h), including every bias. There is no truncation error. Normal training need only evaluate the original two projections and pointwise product; it need not allocate a dense P or solve numerically.

The conditional semigroup and inverse follow by addition of durations with the same h/g: flow_s composed with flow_t equals flow_(s+t), and inverse flow_s is flow_(-s). Holding that conditioning fixed is essential. This is not a claim that applying the complete FFN twice equals a longer solve, since the next input changes the encoding and the gate.

## Complete derivative and optimization identity

The complete FFN Jacobian includes both encoder paths:

```text
JF = U[diag(v)*diag(SiLU'(a))*Wg + diag(g)*V].
```

Freezing g in this derivative would be wrong. The input is fixed along each auxiliary integration interval, but gradients through the conditioning must remain live. This differs from adding a frozen-gate feedback correction to a stopped generator. There is no added feedback here.

Because the exact closed form is the same differentiable function of every original parameter, input and bias, its parameter gradients and deterministic optimizer updates are identical for the same optimizer state and minibatch. Induction gives identical training trajectories. The new component deliberately preserves state-dict keys, parameter order and forward arithmetic so this stronger numerical property can be verified, not just assumed from symbolic equality.

LayerNorm, time FiLM, residual scaling, surrounding blocks and final readout remain unchanged. Substitution of an equal FFN function preserves their complete prediction map and derivatives; no physical property of those outer maps is inferred from the latent shear.

## Zero eigenvalues do not bound amplification

Every eigenvalue of P(h) is zero. Its nonzero singular values are |g_i(h)|, so `||P||2=max_i|g_i|`; the remaining lifted singular values are zero. Unless g=0, P is nonnormal. It has learned conditional strength despite its fixed eigenvalues.

After a coordinate permutation, I+sP is a direct sum of matrices `[[1,s*g_i],[0,1]]`. Solving each two-by-two singular-value problem gives

```text
||exp(sP)||2 = [sqrt(4+s²*M²)+|s|*M]/2,   M=max_i|g_i|.
```

The symmetric generator part has largest eigenvalue M/2. Thus zero spectral radius coexists with arbitrarily large finite-time norm amplification as M grows. The flow has determinant one in lifted (u,v) coordinates at fixed h, but not necessarily unit determinant, invertibility, monotonicity or contraction in h after its nonlinear encoder/conditional generator/projected readout. This example directly limits what a constraint on P eigenvalues alone can establish.

## Prior work and scope

The baseline multiplication and two-thirds hidden-width convention come from Shazeer's [GLU Variants Improve Transformer](https://arxiv.org/html/2002.05202v1), Section2. Additive triangular coupling maps, easy inversion and unit determinant are established in Dinh et al.'s [NICE](https://arxiv.org/html/1410.8516v6), Sections3.1–3.2. These are foundational attributions, not frontier reading recommendations. Our particular lifted linear shear is a simple special case of that general algebra; its existence is not evidence of novelty.

The recent [U-Former ODE preprint](https://arxiv.org/html/2602.11738v1), Section3.2.1, uses a SwiGLU-based vector field in a time-series CDE resampler. That is a relevant warning against treating gated differential fields as new by themselves. Its learned CDE evolution is not this exact conditional shear identity, and its forecasting results do not establish an FM benefit here. No accepted-venue claim is inferred from its anonymized submission link.

The [deep-linear gradient-flow survey](https://arxiv.org/html/2511.10362v1), Sections2–3, also uses nilpotent block-shift matrices, but studies continuous optimization of weights in deep linear networks. Those convergence results do not transfer to this nonlinear FFN inference representation. The keyword match supplies no training theorem for our model.

The practical purpose is to establish an ODE-readable strong baseline whose useful computation is retained exactly, and to make eigenvalue/nonnormality reasoning concrete. Perfect equivalence cannot be reported as a quality gain or an ODE-specific inductive bias. Additional generator restrictions or feedback require a separate method and evidence; this cycle authorizes none.
