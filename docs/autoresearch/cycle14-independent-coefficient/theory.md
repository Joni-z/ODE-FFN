# Independent second coefficients: mechanism and ordinary equivalence

## One internal nonautonomous equation

Consider `z'(u)=(P+uC)z(u)`, z(0)=x. Internal u is distinct from the external FM time t. Holding P/C fixed within this solve gives

```text
z'(0)=Px
z''(0)=(P²+C)x
z'''(0)=(P³+PC+2CP)x.
```

The proposed layer evaluates the second Taylor jet at the original input-conditioned duration s(x):

```text
F_C(x)=x+sP*x+.5*s²*(P²+C)*x.
```

The omitted leading term is `s³*(P³+PC+2CP)x/6`. Its small-s order can be checked against independent RK4. No claim of a small error at learned durations in (.2,1) follows. Finite P/C define a continuous linear nonautonomous exact flow, but the truncated map need not be invertible or stable.

Let B=P²+C and A_s=I+sP+.5s²B. Original conditional matrices commute across durations because they are polynomials in one P. Here

```text
[A_s,A_r]=.5*s*r*(r-s)*[P,C].
```

The new matrix can therefore change the conditional operator family. This is a fixed-generator-family statement at independent durations, not a strict capacity separation of the complete input-conditioned network. P, the controller, embeddings and later blocks can compensate during retraining.

## Complete derivatives

At an input perturbation dx with ds=grad(s)^T dx,

```text
dF_C=A_s*dx + (P+sB)*x*ds
partial_P F_C[H]=sH*x+.5*s²*(PH+HP)*x
partial_C F_C[H]=.5*s²*H*x.
```

All controller derivatives remain live. The affine control adds `.3*s*C*x`; the free control adds `.5*s*r*C*x`, whose complete clock term includes both dr and ds. Bounded clocks and spectra of frozen P alone do not bound these complete Jacobians. No spectral regularizer or stability claim is introduced.

## The necessary ordinary control

For each P, the coordinate change `B=P²+C` is a bijection on matrices. Learning independent P/B in

```text
G(x)=x+sP*x+.5*s²*B*x
```

has exactly the same real-arithmetic function class. It takes two matrix-vector products, versus three for the inherited C implementation. In B coordinates, `partial_P G[H]=sH*x`, with B held fixed; the P² gradient route of the C coordinates is absent. Adam, clipping and weight decay are coordinate dependent, so no training-path equivalence is asserted. This direct quadratic map must be measured, not hidden behind an ODE label.

The affine branch tests a new learned direction with a different clock coefficient, keeping the old P² term. The free branch contains the constrained quadratic coefficient at r=s but introduces a trainable second gate. These controls test coefficient binding/parameterization, not whether all possible ordinary networks are weaker. A pure Taylor derivation is not a novelty result.

## Initialization and its asymmetry

C=0 preserves the original layer's arithmetic, value and inherited derivatives exactly. Adding a zero matrix consumes no RNG. The free head copies the original final controller affine into separate parameters, hence r=s initially; its new-head gradients vanish at C=0. Its common C tangent initially matches the quadratic arm. The affine C tangent differs unless s=.6, and is not claimed to match.

The direct-B initial tensor is the native rounded product P@P. Its real-arithmetic function equals original K2, but a single B product and sequential P products need not be bitwise equal in float32. Its inherited P derivatives intentionally differ. Verify/report the small float32 value difference and float64 equivalence; never claim identical complete initial kernels for this control. Complete zero-head predictions may mask hidden differences.

## Why this is not a renamed previous screen

The older external-time adapter changes P as a function of FM time before forming its square. The present generator varies within one hidden solve and contributes an independent second coefficient. The old effective-map study fixes duration and compares two static matrices around SiLU. Cycle11 rotates a whole polynomial using an input frame; no frame is used here. The state-clock study re-evaluates a nonlinear controller at a midpoint; this retains the frozen original duration. These are differences in the tested functional object, not permission to reopen any old budget or initializer search.

## Primary literature boundary

[Neural ODE Transformers, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/82e3509393670d9d478c359b4ce6f950-Abstract-Conference.html), [Section4 full text](https://arxiv.org/html/2503.01329v1), learns attention/FFN weights as functions of continuous layer index. It establishes prior art for nonautonomous transformer parameterizations, with language-model experiments. It does not derive this second-coefficient comparison or predict an FM gain.

[AREX, October2026](https://arxiv.org/html/2610.03483v1) uses an analytically motivated affine part of the external sampler. Its evidence concerns sampling a pretrained FM field. Our hidden P/C are learned coordinates without that data-moment identification; no AREX speed, fidelity or exact-propagation claim transfers. The present experiment is a bounded architecture test, not a new numerical integration result.
