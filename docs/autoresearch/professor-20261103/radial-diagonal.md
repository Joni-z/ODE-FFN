# A bounded-growth input diagonal: the next functional question

This is a new D(h) contrast, not a rescue of the rejected P-shape constraint.
Return to unchanged vanilla Muon. The question is whether retaining raw quadratic
feature directions while changing their radial growth gives a useful tradeoff.
It is exploratory; the present response measurements do not prove large hidden
amplitudes cause the quality gap.

## One change and one static-amplitude control

At width w define

```
r(h) = sqrt(1 + ||h||^2 / w)
d_radial(h) = h / r(h)
q_radial(h) = d_radial(h) * h = h^2 / r(h).
```

Raw q=h^2 and radial q have the same direction and parity at every common
nonzero h. The original even quadratic feature is retained; this does not
restart the failed odd absolute-diagonal idea. At zero,
q_radial=h^2+O(||h||^4), so the quadratic Taylor coefficient agrees with raw.
At large radius its growth is at most linear instead of quadratic.

At the same P and h, A_radial=P diag(h)/r=A_raw/r. Every singular value is
scaled by the same positive number. Nonzero operators keep the same stable
rank and condition number; this is not singular-value flattening. Nevertheless,
the complete input derivative changes through a rank-one feedback term:

```
J_q_radial = 2 diag(h)/r - (h^2) h^T/(w r^3).
```

This separates response magnitude and input feedback from spectral shape at a
common hidden state. Trained models need not visit the same states or learn
the same P, so the identity cannot predict their final spectral ordering.

The static control uses d(h)=h/sqrt(2), agreeing with radial d on the unit-RMS
surface. It distinguishes an input-dependent radial change from a simple
amplitude reduction at that anchor. It does not match entire initial functions
or later feature distributions. A constant feature rescaling also changes the
effective P coordinates under a fixed optimizer/rate; it is not a pure causal
amplitude control. All common tensors/RNG draws stay paired and defects are
recorded, rather than adding another initialization adjustment.

For K1, the static and raw families have exactly the same realizable functions:
F_static(P,h)=F_raw(P/sqrt(2),h), separately in each solve. They start from the
same stored P here, so they do not start from the same hidden functions. The
static contrast measures this fixed initialization/optimization-coordinate
change; it does not add representational capacity. The earlier
[coordinate-step derivation](logic-chain.md) explains why an unchanged Muon
rate need not give the same function-space step after such scaling. No extra
rate or initialization rescue is included in this screen.

## Complete K1 feature derivative, not just raw-P eigenvalues

Put u=h/r. Then ||u||<=sqrt(w),

```
J_d = (I-u u^T/w) / r
J_q = diag(u) [2I-u u^T/w].
```

The bracket in J_q has eigenvalues between1and2, so

```
||J_q||op <= 2 sqrt(w)
Lip(h + P q_radial(h)) <= 1 + 2 sqrt(w) ||P||op.
||P diag(d_radial(h))||op <= sqrt(w) ||P||op.
```

These are global real-arithmetic module bounds at a fixed P. Raw J_q=2diag(h)
is unbounded for any nonzero column of P. For the unchanged entrance-frozen K2
polynomial, the earlier derivative bound gives the looser global bound
1+2b+1.5b^2 with b=sqrt(w)||P||. The selected training test is K1 only: earlier
evidence does not justify restarting K2 or state-feedback integration.

A finite Lipschitz bound does not imply contraction or invertibility of the
unit Euler-like map. This proposal does not inherit exact-ODE flow guarantees.

The network already contains LayerNorm and finite learned modulation, which
bound the actual incoming domain of each branch at fixed parameters. This is
therefore not a repair of an ill-posed whole model, and its bound is not an
FM-specific convergence or generalization theorem. P can still grow in training.
The relevant empirical question is radial response allocation at fixed cost,
not whether a bound alone certifies good generation.

## Prior coverage and literature limits

The earlier [radial/tangent removal diagnostic](../../research-normalized-geometry-2026-10-26/results/analysis-report.md) changed a trained **whole FFN
branch** and worsened all native fields. The [legacy ODEOnlyFFN](../../research-2026-10-21-reassessment/evidence.md) normalized the
**projected delta**, cancelling a positive scalar clock in its idealized K1
case. Neither is this signed input diagonal before P. [Direct bounded sigmoid](cycle4-results/findings.md)
D was already tested, but changes the raw quadratic feature's sign/direction;
the proposed radial map preserves that pointwise direction. None of these
distinctions establishes novelty or predicts a win.

[Activation-Adjacent Layer Normalization (October2026)](https://arxiv.org/html/2610.21527v1)
studies how inserting LN/RMSNorm changes approximation classes. Sections5.2and9
are a useful guard against calling normalization a free numerical fix: their
results concern shallow compact-set architectures and exclude fixed stabilizers,
deep residual compositions and finite precision. They do not cover or endorse
this softened quadratic feature. Our Jacobian statements above are direct
derivations, and training must establish any practical value.

No new activation between solves, loss, optimizer, spectral penalty, sampler
or width is combined with the D change. Keep strong ordinary SwiGLU controls.
