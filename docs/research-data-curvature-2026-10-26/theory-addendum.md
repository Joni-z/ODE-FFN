# Post-result corollary: the reduced identity includes the data prefix

Derived after the registered analysis, 26 October 2026. This is an algebraic corollary, checked separately; it does not change the registered protocol, frozen source or primary outcomes.

Keep the notation and anchor-dependent frames from theory.md. Since J1=Q1+(C1h)g1^T and J2=Q2+b2g2^T, direct expansion gives

```text
J_F - J0 = b1 g1^T + b2 g2^T D Q1
b1 = J2 D(C1h); b2=C2y; J0=Q2 D Q1
```

Thus the local Jacobian difference from the anchor-frozen map has rank at most two. For the same output projector W, W(J_F-J0)=0. This is a local comparison to a different anchor-frozen map at each point, not a claim that the complete FFN is linear or has rank two.

The general chain rule then gives

```text
W (H_(F∘Z) - H_(F0∘Z))[V,V]
  = W (H_F - H_F0)[J_Z V,J_Z V]
    + W (J_F-J0) H_Z[V,V]
  = 0
```

The first term vanishes by the registered reduced-space identity; the second vanishes by the Jacobian factorization above. Therefore the equality extends to the complete data-composed Hessian, including the nonlinear prefix. It does not require neglecting prefix curvature.

All projectors and frozen clock values are held fixed at the anchor during differentiation. Differentiating an input-dependent projector would add further terms and is not asserted here. The input nullspace still has dimension at least fourteen, and the output complement at least sixty-two. Current modules can change these frames when their weights change, so neither the identity nor a projected target mismatch proves an architecture-wide approximation lower bound.

The supplementary check uses all nine ODE modules and eighty anchors, native local Jacobians, the previously independently audited complete Hessian arrays, and the analytic frozen maps already checked natively. It also evaluates all eighteen students against their targets in the six current ODE-student frames, now using complete Hessians. These are post-result supplementary quantities, distinct from the registered local reduced-curvature metric.
