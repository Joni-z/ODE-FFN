# Exact second-order structure at an arbitrary input

All identities here are derived from the unchanged professor layer. They are local differential statements, not approximation lower bounds or performance guarantees. All clocks are scalar, as in the original K2 model.

## One solve
Let L(h)=Q(a(h))h with

```text
Q(a)=I+aP+(a²/2)P²
C(a)=P+aP²
g=grad a(h)
```

The product and chain rules give

```text
J_L = Q + (Ch)g^T
H_L[u,v] = (Cu)(g·v) + (Cv)(g·u)
           + (P²h)(g·u)(g·v) + (Ch) H_a[u,v]
```

The final two terms are zero at h=0. At a general point the gate Hessian can have many input directions, despite a scalar output; dropping it is an additional approximation, not a rank theorem.

## Two solves and the data prefix
Set p=L1(h), y=SiLU(p), D=diag(SiLU'(p)), and let J1,J2 be the complete solve Jacobians at h,y. For arbitrary u,v, let r_u=J1u and r_v=J1v. Then

```text
H_F[u,v] = J2[SiLU''(p) * r_u * r_v]
           + J2 D H_L1[u,v]
           + H_L2[D r_u,D r_v]
```

These are the middle, first-clock, and second-clock curvature origins. First derivatives of both clocks remain inside the middle route. If both clock values are held fixed, J0=Q2 D Q1 and H0[u,v]=Q2[SiLU''(p)*(Q1u)*(Q1v)].

For the common data prefix h=Z(x,t), with fixed t,

```text
H_(F∘Z)[u,v] = H_F[J_Zu,J_Zv] + J_F H_Z[u,v]
```

This separates local module curvature from curvature introduced by the feature map. The frozen-clock comparison uses H0[J_Zu,J_Zv]+J0 H_Z[u,v], so it does not retain a hidden full-clock embedding contribution.

## An exact reduced-space identity
Use U=J_Z, a 64-by-16 matrix. At one anchor define C1=P1+a1P1², C2=P2+a2P2², g1=grad a1(h), g2=grad a2(y). Define the two physical-input covectors

```text
c1 = g1^T U
c2 = g2^T D Q1 U
```

Let V be an orthonormal basis of their common nullspace in R^16, which has dimension at least 14. For columns u,v of UV, the first clock variations vanish, so J1u=Q1u. The second clock's first variations also vanish because g2^T D Q1u=0. Hence the single-solve Hessians reduce to

```text
H_L1[u,v] = (C1h) H_a1[u,v]
H_L2[DQ1u,DQ1v] = (C2y) H_a2[DQ1u,DQ1v]
```

In output space define b1=J2 D(C1h), b2=C2y, and let W be the orthogonal projector onto the complement of their span. Its rank is at least 62. The middle route differs from Q2 by a vector along b2. Applying W therefore yields the exact identity

```text
W H_F[UV,UV] = W Q2[SiLU''(p) * (Q1UV) * (Q1UV)]
```

The right side denotes the bilinear tensor for all pairs of retained columns. This does not assume constant clocks, small inputs or small gate Hessians. It removes at most two directions on each side, using the learned ODE's own geometry.

Normalize nonzero constraint rows and output columns before computing their SVD, to avoid losing a direction merely because its amplitude is small. Use relative rank tolerance 1e-10; vectors with norm at most 1e-15 are zero for this numerical construction. Record dimensions and residuals rather than asserting exact numerical ranks in advance.

## What this does not prove
The projection depends on the current module, and another set of weights can change both frames and the remaining SiLU features. A generic target having curvature error in this frame is not proof that no ODE module can fit it. Restricting input directions to J_Z avoids arbitrary ambient perturbations, but local derivatives still do not determine finite-distance function error or training outcomes. The identity concerns the local Hessian term; prefix curvature must be added separately for the complete data-composed function. No architecture-wide rank lower bound, novelty claim, or proposed new operator follows solely from these formulas.
