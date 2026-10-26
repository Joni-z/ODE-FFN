# Next question: characterize the local nonlinear response before changing the operator

This is a post-result mathematical direction, not a registered training experiment or an architectural nomination.

The local study establishes a fitting gap on common inputs, with 81.8%–90.8% of excess over both generic controls outside the empirical affine component. Own-family local fitting is accurate. This makes the function family of the two-solve composition a more specific object of study than whole-network coordination or raw spectral amplification. It does not distinguish expressivity from finite-budget optimization.

One exact starting point is the local second-order structure of a professor solve. Write L(h)=Q(a(h))h, with Q(a)=I+aP+(a²/2)P². Since L(0)=0, its origin derivatives satisfy

```text
A = Q(a(0)); C = P + a(0)P²; g = grad a(0)
DL(0)[u] = A u
D²L(0)[u,v] = (C u)(g·v) + (C v)(g·u)
```

The second-order contribution of this single solve is carried through one shared gate direction at the origin, despite the controller's hidden width. Composing two solves around SiLU adds the coordinatewise activation curvature and two gate-response routes. That algebra suggests examining which nonlinear interactions the operator produces, rather than simply integrating its linear part more accurately.

For the complete composition F=L2∘SiLU∘L1, write A=Q1(a1(0)), B=Q2(a2(0)), C_i=P_i+a_i(0)P_i² and g_i=grad a_i(0). The chain rule gives

```text
DF(0) = (1/2) B A
D²F(0)[u,v] = (1/2) B[(Au) * (Av)]
              + (1/2) B[(C1u)(g1·v) + (C1v)(g1·u)]
              + (1/4) [(C2Au)(g2·Av) + (C2Av)(g2·Au)]
```

Here `*` is coordinatewise multiplication. The first term is the middle SiLU curvature; the other terms are the two scalar-clock routes. A separate native derivative check covers the three frozen ODE teacher modules and all six final ODE students, with four fixed direction pairs each. All 36 checks pass; maximum Jacobian and directional-Hessian differences are 7.77e-16 and 3.33e-16. See [origin-identity-check.json](origin-identity-check.json). This verifies the identity at the stated point, not the simplification's relevance at actual inputs.

The relevance boundary is essential: the actual post-normalization inputs have substantial norm, and at a general input there are additional terms involving the gate Hessian and the input itself. An origin identity is not an on-distribution approximation lower bound. Before selecting a design, derive the full composition and test the validity of any reduced response description at the actual cached inputs, using explicit derivative identities and ordinary generic controls. If the simplification is inaccurate there, stop that explanation.

Any proposed extension must also survive existing negative evidence: vector clocks, shared controller features, affine forcing, state augmentation, nonlinear midpoint feedback, forced SwiGLU responses and generic recurrence have already been examined in their registered settings. A local rank observation does not authorize relabeling one of them as a new method. No bias, gate-count, spectral, activation or optimizer rescue is selected here.

The literature distinction remains useful: an ODE architecture can use a general nonlinear field, while our original operator deliberately restricts that field. A later candidate needs a task-relevant reason for its restriction and matched evidence that it earns its computation. The active project goal remains unachieved until that candidate exists and passes ordinary FM, transfer and cost checks.
