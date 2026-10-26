# Coupled SwiGLU generator: prior-art and mechanism review

2026-10-25. Read-only assessment; no implementation or training selected.

**Decision: do not launch this as the next experiment on present evidence.** The cross-channel term is genuinely different from Stage20's coordinatewise correction. Nevertheless, its K2 form is exactly a tied, frozen-gate FFN re-evaluation, with substantial extra computation. Neither the formula nor the reviewed literature establishes why that computation addresses our measured failure. Renaming it a new generator would not override the [Stage20 protocol](protocol-forced-gate.md): “Stop failed settings without width, slope, rate-bound, temperature, coupling or additional-order rescue.”

## Exact operation, including biases

Use the actual SwiGLU parameterization, with input width d and gate width r:

```text
g(h) = SiLU(Wg*h+bg),  D(h)=diag(g(h))
F(h) = U D(h)(Vh+bv)+bo
G(h) = U D(h) V
r(h) = U D(h) bv+bo.
```

Thus `F(h)=G(h)h+r(h)`. Calling G canonical means canonical for this chosen parameterization and frozen-gate convention; it is not the complete input Jacobian of F.

With h fixed, the correct biased forced system and endpoint are

```text
y' = tau*G(h)*y + F(h),  y(0)=0
y(1) = phi1(tau*G(h))*F(h)
     = F(h) + (tau/2)*G(h)*F(h) + O(tau²).
```

Only without biases can this be shortened to `G*phi1(tau*G)*h`. Equivalently use the homogeneous-coordinate generator `A=[[G,r],[0,0]]` and input `(h,1)`. Its first response is `(F(h),0)`; the next is `(G F(h),0)`. Reapplying a biased affine map to the response would incorrectly add r again.

In the bias-free case,

```text
G²h = sum_ij u_i*g_i*(v_i^T u_j)*g_j*(v_j^T h).
```

The `g_i*g_j` terms with i≠j mix gates through VU. Stage20 did not contain these within-block feedback terms. However, existing multilayer SwiGLUs already compose gated features; this algebra is not a fixed-budget representational separation. All correction terms remain in U's output span. No new projection is learned, but additional applications are executed.

## K2 is exactly a simple recurrence

Define the frozen-gate affine map `F_h(z)=U D(h)(Vz+bv)+bo`. Then

```text
K2(h) = F_h(h + (tau/2)*F(h)).
```

This identity is exact, including biases. The proposed K2 therefore adds one residual displacement and reevaluates the same affine feature transformation while reusing the original gates. Its ODE interpretation does not distinguish it from this elementary tied recurrence.

Recomputing the gates instead gives `F(h+(tau/2)*F(h))`. At tau=0 its derivative is `.5*J_F(h)*F(h)`, where

```text
J_F = G + U diag(Vh+bv) diag(SiLU'(Wg*h+bg)) Wg.
```

The proposal specifically omits the second, gate-derivative term. A useful hypothesis would have to explain why freezing these gates improves the computation. The higher-order matrix response alone does not provide that explanation, nor does it approximate the nonlinear ODE `y'=F(y)`.

## Closest useful primary evidence

**Geiping et al., Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (2025).** Section3 defines `e=P(x)`, `s_i=R(e,s_(i-1))`, `p=C(s_r)`. The recurrent core contains attention and gated-SiLU MLPs with normalization; activations are recomputed from the new state. This is direct prior art for obtaining effective depth through parameter reuse, not for our particular frozen G or factorial coefficients. Its language-model reasoning results do not show that FM regression needs this recurrence. The applicable lesson is to separate parameter count from executed depth and computation. Primary full Section3 inspected. [Paper](https://arxiv.org/html/2502.05171v2#S3); [NeurIPS2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Paper-Conference.pdf).

**Lin et al., Allocating Recurrent Compute in Looped Language Models (August2026 preprint).** Eqs.(3)–(5) compare no-loop, full-block looping and mixer-only looping; the last applies its dense FFN once. Each repeated mixer recomputes input-dependent features. This provides a useful counterpoint: benefits from recurrence do not establish that repeating dense FFN computation is worthwhile. Section3.4 separately measures whether successive interventions change the final readout in distinct, non-negligible and useful ways. Its Gated DeltaNet sequence setting and matched-token experiments do not establish the corresponding result for our tokenwise FM model. Full methods inspected; no accepted-venue claim. [Paper, Section3](https://arxiv.org/html/2608.18230v1#S3).

Searches for matrix-polynomial FFNs and the label “Tied Flow/TiedFlow” did not identify a verified exact prior-art match to this full construction. That absence does not establish novelty. Matrix-polynomial attention and elementary activation polynomials concern different operators and supply no justification here.

## Cost, controls, and the stopping decision

A direct factored baseline uses approximately `3dr` dense MACs; K2 uses `5dr` because G is applied once more through V and U. Prefactoring VU changes amortized costs but requires accounting for forming/updating the r-by-r product. Equal parameters are not equal compute. SiLU gates and U/V are unbounded, so a bounded tau also does not reproduce Stage20's bounded scalar response guarantee.

Any independently justified future comparison would need ordinary SwiGLU at matched inference cost, a longer-trained same-width baseline at matched training cost, and a tied re-evaluation that recomputes gates. A frozen recurrence with the same coefficient is identical to K2, so it cannot serve as evidence for an ODE-specific advantage. These are interpretation requirements, not a selected experiment grid.

[Stage20](analysis-forced-gate/analysis-report.md) found active but unhelpful scalar corrections. It did not identify missing cross-gate feedback as the cause. The [complete-field diagnostic](analysis-complete-field/README.md) also supplies no such mechanism: correcting measured structural violations leaves most parent error. G is a hidden conditional operator, not the complete FM Jacobian; neither its powers nor its exponential inherit the posterior-covariance theorem.

The narrow untested hypothesis is that reusing learned gate features for an extra cross-channel feedback computation helps more than ordinary extra computation. Current evidence does not support selecting it. Under the registered stop rule, adding coupling now would be a rescue; independent, task-specific evidence would be needed before treating it as a separate research direction.
