# Next review: nonlinear feature responses along a flow

This is an unselected hypothesis after the stopped cycle04, not runnable source, a training authorization or an effectiveness/novelty claim. It connects the observed advantage of extra ordinary nonlinear gates to a different computation, rather than rescue cycle04's partition, coefficient, rank or feedback settings.

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

Before selection, verify the integral/AD identities, nonzero-head tangent pairing, actual cost, runtime buffers, batch streams and resume. An ordinary single-midpoint control would separate new feature integration from redundant parameterization; completed Swi143 remains the budget reference. Predeclare a quadrature-accuracy diagnostic so extra points cannot become a post-outcome rescue. Review the earlier nonlinear-drift/state-clock/anchored-corrector stops and primary integrated-feature literature first. No new training is selected here.
