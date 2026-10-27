# Next review: nonlinear feature responses along a flow

This review originated as an unselected hypothesis after the stopped cycle04. The bounded [cycle05 screen](cycle05-feature-flow/analysis.md) has now completed: numerical accuracy passed, quality failed, and the setting is stopped. It tested a different nonlinear computation rather than rescue cycle04's partition, coefficient, rank or feedback settings. It is not an effectiveness or novelty claim.

## Proposed operator

Keep a square-zero gate block P_C but evolve preactivations a instead of mixing already activated features:

```text
a'=P_C*a, a(0)=Wg*h+bg
u'=diag(v)*SiLU(a), u(0)=0, v=V*h+bv held fixed.
```

P_C²=0 gives the exact linear-state path `a(s)=a0+s*P_C*a0`. The intended endpoint is

```text
U[ v elementwise integral_0^1 SiLU(a0+s*P_C*a0) ds ] + bo.
```

Activation inside the integral generates responses at different affine gate directions. Cycle04 only linearly mixed the existing SiLU(a0) dictionary. A single transformed preactivation is insufficient: `SiLU((I+P_C/2)*a0)` is ordinary SwiGLU with reparameterized Wg and bg.

## A controlled comparison to verify

A fixed two-node Gauss-Legendre rule uses `.5±1/(2*sqrt(3))`. At C=0 it preserves ordinary SwiGLU and matches the single-midpoint control's first C derivative. Around the segment centre c with displacement b, the integrated activation's leading extra term is `(b²/24)*SiLU''(c)`. Two-node quadrature matches it; midpoint does not. Componentwise quadrature error is bounded by `|b|^4/4320` times the maximum fourth activation derivative on the segment. This bound requires verification and should not be replaced by a claim of exact nonlinear integration.

## Nonlinear caveat

The pointwise joint Jacobian can be cube-zero while the nonlinear flow's Taylor series does not terminate. Its third u derivative contains `diag(v)*SiLU''(a) elementwise (P_C*a)^2`, generally nonzero even though P_C²=0. Derivatives of the evolving activation supply terms missed by powers of a frozen Jacobian. The complete FFN gradient must include every activation trajectory and the value path.

The [cycle05 derivation](cycle05-feature-flow/design.md) records the checked integral/AD identities, full derivative, midpoint absorption, independent RK4 fixture, primary literature and earlier stop scopes. Its protocol fixes the nonzero-head/stream/resume/cost preflight and a quadrature-accuracy gate. Only two arms across the existing five laws are selected; no extra quadrature-point rescue is permitted.
