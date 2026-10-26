# Forced-response SwiGLU gate: prior art and claim boundary

Review date: 2026-10-24. This is a bounded mathematical and primary-source review, not a training result.

**Decision:** a controlled test is admissible, but its defensible question is whether a particular higher-order activation shape improves an otherwise unchanged SwiGLU. The literature does not supply an FM-specific mechanism. Neither the forced linear solve nor learnable Swish-like activation is new. No verified exact prior-art match for the complete proposed gate was found; this limited search cannot establish novelty.

The [completed diagonal study](analysis-diagonal-flow/analysis-report.md) remains negative. On Product16, adaptive diagonal K2/exp gave oracle MSE 0.163066/0.163193, versus SiLU 0.153892 and SwiGLU128 0.092326. Keeping SwiGLU's separate gate and value projections is therefore a sensible control against the earlier architectural change. It is not evidence that adding a forced solve will help.

## What the proposed gate actually is

For one hidden channel, let `a=Wg*x+bg`, `v=Wv*x+bv`, `q=sigmoid(a)`, `b=a*q`, and `lambda=tanh(eta)`, initially zero. Preserve the usual output projection U:

```text
FFN(x) = U [g_lambda(a) * v] + bU

g_lambda(a) = b * phi1(lambda*q)
           = a * expm1(lambda*q) / lambda,  lambda != 0
g_0(a)     = a*q

phi1(z) = integral_0^1 exp(t*z) dt,  phi1(0)=1.
```

This is the exact endpoint of `y'=lambda*q*y+b`, `y(0)=0`, at auxiliary time 1, with both q and b frozen from the input. It is also exactly an ordinary learned scalar activation inside SwiGLU. The solve has no carried state, evolving controller, or identified relationship to the physical time in flow matching.

Our algebra gives

```text
g_lambda(a) = a*q + (lambda/2)*a*q^2
                   + (lambda^2/6)*a*q^3 + ...

K1:   a*q
K2:   a*q + (lambda/2)*a*q^2
exact: all terms above.
```

Thus K2 already tests a generic learned mixture of two activation functions. K2 and exact have the same complete initial parameter Jacobian when all projections and biases are paired; their distinction starts beyond the local tangent. Static `q=1/2` in the generator gives `g=b*phi1(lambda/2)`, which can be absorbed exactly into U's columns. That control changes parameterization without enlarging the SwiGLU function class.

For adaptive q and nonzero lambda, the multiplier varies with a, so it cannot be removed by a single constant output-column rescaling while those features are fixed. This is a narrow statement, not a finite-width separation theorem against all reparameterized SwiGLUs. Since `abs(lambda*q)<1`, its positive gain lies between `1-exp(-1)` and `exp(1)-1`, approximately 0.632 and 1.718. This does not bound the full network Jacobian or sampling error.

## Directly relevant prior art

**Forced linear response is standard algebra.** Mamba's Section 2, Eq. (4), writes the zero-order-hold input response as `(Delta*A)^(-1)*(exp(Delta*A)-I)*Delta*B`, with the removable-zero extension represented by phi1. Our scalar endpoint is precisely this familiar affine-response operation with zero initial state. Mamba carries state across a sequence; the proposed activation resets it at each invocation. Its memory mechanism, efficiency results and sequence benchmarks do not transfer. The full primary text was inspected. [Gu and Dao, Mamba, arXiv v2 (2024), Eq. (4)](https://arxiv.org/html/2312.00752v2#S2).

**The exponential deformation already has an activation antecedent.** Godfrey and Gashler's Eq. (2) defines `G(alpha,x)=(exp(alpha*x)-1)/alpha+alpha`. Our gate is exactly `a*(G(lambda,q)-lambda)`, continuously extended at zero. Their final soft-exponential activation, Eq. (4), instead uses a logarithmic branch for negative alpha; our negative-lambda gate retains the exponential branch. This is an exact algebraic predecessor, not an assertion that the whole SwiGLU construction is identical. The full PDF was inspected; its age is justified here solely by the exact expression. [A continuum among logarithmic, linear, and exponential functions (2016 preprint), Eqs. (2)–(4)](https://arxiv.org/pdf/1602.01321).

**A close generic Swish modification already exists.** RoSwish defines `(a+alpha)*sigmoid(beta*a)-alpha/2`, with learned channel parameters. Its sigmoid direction is directly relevant to the local analysis below. The centering term matters: with beta fixed to 1, its alpha derivative is `q-1/2`, not q. Multiplication by v consequently adds a linear-value correction `-v/2`; it is not identical to an uncentered offset-SiLU control. The paper is in *Neural Networks* 192 (2025), article 107892. The publisher abstract/formula and the authors' full implementation were checked; the complete journal article was not accessible. No performance claim is imported. [Paper DOI](https://doi.org/10.1016/j.neunet.2025.107892); [authors' implementation, `RoSwish`](https://github.com/weirdoZHANG/RoSwish/blob/main/deep_network_18_26_32.py#L125).

**Current SwiGLU work already treats this as activation design.** The May 2026 preprint *More Expressive Feedforward Layers: Part I* studies learned linear combinations and input-dependent mixtures of activations while preserving the projections of SwiGLU-type FFNs, especially Eqs. (4)–(7). Our one-parameter sigmoid-power series is not their specified dictionary or router. Their finite-width separation theorem therefore does not prove a separation for this candidate. The direct implication is the appropriate comparison class: learnable activation modifications, not only alternative ODE architectures. Full primary HTML and method equations were inspected; this review makes no venue claim. [Wang et al., Sections 3.2–4.2](https://arxiv.org/html/2605.26647v1#S3.SS2).

## The new initial direction has a simple non-ODE explanation

The following identity is our derivation, rather than a claim taken from those papers. Write `b(a)=a*q`. Then

```text
b'(a) = q + a*q*(1-q)
a*q^2 = b(a) + q - b'(a)

d g / d eta at eta=0 = (1/2)*a*q^2.
```

For each channel, baseline output-column scaling supplies the `b*v` direction, and a gate-bias perturbation supplies `b'*v`. The current SwiGLU implementation has gate biases. After allowing those existing tangent directions, the new direction is equivalent to adding `q*v`, the familiar GLU feature. A generic offset gate `(a+c)*sigmoid(a)`, initialized with c=0, supplies exactly that direction. Conversely, the displayed identity expresses its added direction using the forced-gate tangent plus baseline directions.

This establishes equality of the enlarged **tangent spans** for those paired biased models. It does not establish equal NTK matrices, equal parameter norms, equal optimizer trajectories, or equivalence at finite parameter changes. It also does not mean the added direction is necessarily independent on our data. RoSwish is a relevant nearby prior art, with the centering distinction above.

No extra offset arm is required for the narrowly scoped decision currently proposed: K2 already includes exactly the forced gate's new initial parameter direction, and exact versus K2 asks whether the higher terms help beyond it. An offset comparison would become relevant to a broader claim that this ODE construction is uniquely effective relative to generic activation adaptation. The present review does not recommend expanding the registered test.

## What a bounded comparison could establish

Preserve the same gate/value/output projections, initialization, width, objective and training protocol. Interpret the planned controls in this order:

| Observation | Supported interpretation |
|---|---|
| Static response improves | A different parameterization may help; no larger function class. |
| K2 improves over baseline/static | The constrained `a*q^2` activation correction may help. This does not identify an ODE-specific mechanism. |
| Exact improves over K2 as well as practical controls | Evidence for useful higher-order gate shape under this protocol, beyond their identical initial parameter Jacobians. |
| Exact ties K2 or loses practical controls | No supported reason to retain the exponential response for this task. |

Even the third outcome would not prove that physical flow geometry, semigroup consistency, or a velocity-Jacobian constraint caused the gain. None of the reviewed sources relates these particular hidden sigmoid powers to the FM population target. Exact integration of an invented frozen auxiliary system is a valid derivation, but it is not that missing connection. Architectural acceptance and a novelty claim would still need separate evidence.

Search scope: primary arXiv/publisher texts and authors' code; queries included `activation expm1`, `activation phi_1 exponential`, `forced linear ODE sigmoid activation`, `exponential Swish`, `SwiGLU learnable activation 2025 2026`, `exponentiated sigmoid`, `soft exponential Godfrey Gashler`, and `RoSwish`. Exact-formula searches did not identify a verified full-gate match. Unrelated exponential activations and generic performance analogies were excluded. No source, configuration, or training job was changed for this review.
