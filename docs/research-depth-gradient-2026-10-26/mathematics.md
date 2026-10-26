# Tied-gradient loss and the minimal depth modes

Write a three-copy predictor loss as L(theta1,theta2,theta3,eta), with eta all common parameters. Restricting theta1=theta2=theta3=theta gives exactly the shared predictor. The chain rule gives `g_shared=g1+g2+g3`, while the eta gradients agree. This holds for the complete potential-gradient readout too; its higher input derivatives remain part of each loss gradient.

The duplicated parameter space has norm `sum_i ||delta_theta_i||²`. The orthogonal projection of `(g1,g2,g3)` onto tied directions is `(gbar,gbar,gbar)`, where gbar is their average. Its squared norm is `||g1+g2+g3||²/3`. Therefore the retained fraction is

```text
E0/(E0+E1+E2)
 = ||g1+g2+g3||² / (3*sum_i ||g_i||²),
```

between0 and1. Identical gradients retain all energy; orthogonal gradients retain one third; cancellation can remove it entirely. The unnormalized sum-norm ratio has range0–3 and must not be mistaken for a retained fraction.

The constant, linear and quadratic depth vectors in the protocol form an orthonormal basis of R³. Applying them along the depth axis yields Parseval's identity. An affine depth dependence retains the constant and linear terms, with exactly E2 left out. This is a tangent-space statement at matched parameter values, not a global capacity theorem. Adding an independent depth-slope parameter enlarges the local parameter tangent space, but its learning trajectory, initialization masking and optimizer metric still need testing.

The banks-A/B direction comparison uses the same duplicated Euclidean norm for both subspaces. If dA=-Proj(gA)/||Proj(gA)||, the first-order bank-B loss derivative is `dot(gB,dA)`. Its normalized descent alignment is `dot(gB,Proj(gA))/(||gB||*||Proj(gA)||)`. A positive value indicates local descent transfer, not finite-step or Adam improvement. Zero projected gradients yield no defined unit direction.

For P, the linear-depth tangent corresponds to a nonautonomous internal generator `P(s)=P0+u1(s)*P1`, with P1 initially zero. It preserves the same frozen P at all three visits initially and adds a specific depth-varying direction. The existing shared whole-block Jacobians already vary with hidden state, so the proposal must not be described as the first source of noncommutativity. No slope, spectral constraint or training change is implemented by this derivation.

## Descriptive magnitude/direction factorization

Added after the diagnostic, not used to change selection. For nonzero gradients let `r_i=||g_i||`, `S=sum r_i` and `T=sum r_i²`. Multiplying the two factors gives the original fraction exactly:

```text
A=S²/(3T), D=||sum g_i||²/S², R=A*D.
```

Cauchy–Schwarz and positivity give1/3<=A<=1; the triangle inequality gives0<=D<=1. Identical directions but unequal magnitudes can therefore lower R without any opposing pair. Equal magnitudes do not guarantee alignment. At T=0, all ratios remain undefined. These factors are not additive causal contributions and are not invariant to general parameter reparameterization. Every existing state/group/bank is retained in the descriptive table; no new optimizer direction, parameter group, mode or training arm is selected from it.
