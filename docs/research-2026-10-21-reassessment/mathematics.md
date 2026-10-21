# Mathematical reassessment of the original ODELayer

These are derivations from the implemented model and standard probability/ODE identities. They are not claims of new theorems. Column vectors are used. `h` is a hidden feature; `x` is a data-space state; `t` is generation time; `s` is a hidden-layer duration. They must not be identified with one another.

The [verification script](verify_math.py) checks the principal identities in float64, including the repository's Gaussian-mixture oracle. [Results](math-checks.json) agree to at worst 2.85e-14 on the tested inputs. Numerical checks supplement the following arguments; they do not establish global claims.

## 1. The implemented operation and what its ODE interpretation means

For one layer, let `s(h)=0.2+0.8 sigmoid(g(h)/10)`. The recurrence gives, by induction,

```text
term_k = s(h)^k P^k h / k!
F_K(h) = T_K(s(h)P) h
T_K(A) = I + A + ... + A^K/K!
```

For each initial h, freeze s(h). Then this is a truncated solution of `dy/du=P y`, evaluated at `u=s(h)`. The exponential series exists for every finite matrix P. A spectral constraint is not needed for convergence of that infinite series.

This is different from solving the autonomous nonlinear ODE `dy/du=f(y)=s(y)P y`. For a small inner interval epsilon, the latter has expansion

```text
y(epsilon) = h + epsilon*s*P h
             + epsilon^2/2 * [s^2 P^2 h + s*(grad(s)^T P h)*P h]
             + O(epsilon^3).
```

Proof: `Df = sP + (Ph)grad(s)^T`; multiply `Df` by `f=sPh`. The original K2 operation has the `s^2 P^2 h/2` term but not the derivative term. It is a valid approximation to its frozen-duration linear system, not a second-order solver for this different nonlinear system. The epsilon expansion is local; evaluating its displayed polynomial at epsilon=1 does not assert a small remainder.

The data-space FM ODE has another second-order term:

```text
x(t+dt) = x + dt*v(x,t)
          + dt^2/2 * [partial_t v(x,t) + J_v(x,t)*v(x,t)] + O(dt^3).
```

Neither term in brackets is generally equal to `P^2 h`. Thus the presence of a second-order matrix polynomial in an FFN is not evidence of second-order accuracy for the generative flow.

## 2. Separate function class from parameterization

For a constant s>0 and any fixed s0>0, put `P0=(s/s0)P`. Then `T_K(sP)=T_K(s0 P0)`. Learned scalar duration and fixed duration have the same function class when P is unrestricted. A norm constraint or a parameter penalty can break that equivalence of feasible/penalized models. Optimizer trajectories need not be equivalent even without a penalty.

For constant durations, the complete two-layer FFN is exactly

```text
B2 * SiLU(B1*h),     Bi=T_K(si Pi).
```

This is a width-preserving, bias-free ordinary FFN with polynomial matrix parameterization. At K1 every real B is possible: `P=(B-I)/s`. At K2 this needs qualification:

```text
B = I + A + A^2/2 = [(A+I)^2+I]/2,    A=sP.
```

For symmetric A, every eigenvalue of B is at least 1/2. For general real A that lower bound does not apply. For example `A=[[-1,-1],[1,-1]]` gives B=0 exactly. A matrix exponential is always invertible; a Taylor layer need not be. Restricting P to symmetric, negative definite or skew-symmetric matrices would therefore change the function class, not merely remove numerical noise.

There is also a concrete optimization effect. Write `B(A)=I+A+A^2/2`. In direction H,

```text
L_A(H) = DB(A)[H] = H + (A H + H A)/2
L_A*(G) = G + (A^T G + G A^T)/2.
```

The adjoint follows from the Frobenius inner product. Under Euclidean gradient flow directly on A, an objective `loss(B)` induces

```text
dB/dtau = -L_A L_A* [grad_B loss].
```

This is a parameter-dependent positive-semidefinite preconditioning operator. It is not necessarily a beneficial one: at A=-I, L_A=0. If A is symmetric with eigenvalues a_i, L_A scales matrix direction E_ij in that eigenbasis by `1+(a_i+a_j)/2`, which can vanish. With P as parameter, the derivative has the corresponding s and s^2 factors. This argument does not describe AdamW exactly.

Consequently, a fixed-duration ODE advantage, if established, could arise from optimization geometry rather than a more expressive ODE model. Existing experiments have not established such an advantage.

## 3. The exact structural restriction imposed by one scalar gate

At K2 the effective matrix is

```text
B(h) = I + s(h) P + s(h)^2 P^2/2.
```

The coefficients are linked: the quadratic coefficient is the square of the linear coefficient divided by two. The matrices P and P^2 are linked too. Across inputs, B moves on a one-parameter curve in `span{I,P,P^2}`. For any two scalar durations a,b, `T_K(aP)` and `T_K(bP)` commute, since they are polynomials in the same P. If P is diagonalizable, they share its eigenvectors, although their gains differ. This is more specific than arbitrary input-dependent matrix mixing.

It does NOT mean the entire network's Jacobians commute, nor that hidden feature directions cannot change through SiLU, normalization or other layers.

Differentiating gives

```text
J_F = A + b u^T
A = T_K(sP),    b = P T_(K-1)(sP) h,    u = grad(s).
```

The gate correction has rank at most one and norm `||b|| ||u||`. Its symmetric part has at most two nonzero eigenvalues, `(b^T u +/- ||b|| ||u||)/2` (the noncollinear case; the degenerate case follows continuously). Its contribution to directional growth depends on alignment, not just its norm. In particular, `sym(b u^T)` need not be positive semidefinite.

For the two-layer block, evaluated at its native intermediate states, let `D=diag(SiLU'(F1(h)))` and `Ji=Ai+bi ui^T`. Then

```text
J_block = J2 D J1
J_block - A2 D A1 = A2 D b1 u1^T + b2 u2^T D J1.
```

The difference has rank at most two. Freezing gates here means holding their *values at these states* fixed for differentiation; it is not retraining fixed gates or changing forward outputs. In a 2D output model a rank bound of two is not evidence of inadequate output expressivity.

Even exact exponentiation would not make the gated map automatically invertible. For `F(h)=exp(s(h)P)h`, the determinant lemma gives

```text
det(J_F) = exp(s*tr(P)) * [1 + grad(s)^T P h].
```

The second factor may vanish. Thus fixed-time linear-flow properties cannot be transferred wholesale to input-dependent duration, let alone to the complete FFN containing SiLU.

## 4. What the population flow-matching objective actually requests

Assume `x0~N(0,I)` independent of target x1 and `x_t=(1-t)x0+t x1`. Let `Y=x1-x0`. Squared regression gives

```text
v*(x,t) = E[Y | x_t=x]
E||v-Y||^2 = E||v-v*||^2 + E||Y-v*||^2.
```

The last term is label variance, independent of the predictor. It explains why pair MSE is not the same quantity as oracle field MSE. There is no hidden-exponential-accuracy term in the loss. This conditional/marginal equivalence is foundational [Flow Matching](https://arxiv.org/abs/2210.02747), not a contribution of this project.

For `0<t<1`, define the posterior mean `D(x,t)=E[x1|x_t=x]`. Then

```text
v*(x,t) = [D(x,t)-x]/(1-t).
```

The conditional likelihood of x given x1=y is Gaussian, with mean ty and covariance `(1-t)^2 I`. Differentiating its normalized posterior gives

```text
grad_x log p(y|x) = t/(1-t)^2 * (y-D)
J_D = t/(1-t)^2 * Cov(x1 | x_t=x)
J_v* = -I/(1-t) + t/(1-t)^3 * Cov(x1 | x_t=x).
```

Interchanging differentiation and expectation requires integrability; our finite Gaussian mixtures satisfy it. This is a posterior-covariance identity, related to Tweedie's formula and explicitly developed in [Guidance of FM, Section 3.4/Appendix A.12](https://arxiv.org/html/2502.02150v3) and [Stéphanovitch, Proposition 9](https://arxiv.org/html/2604.06065v1). It is not a new theorem.

Consequences under this particular independent Gaussian path:

- The optimal data-space Jacobian is symmetric. For `0<t<1`, its eigenvalues are at least `-1/(1-t)`; its positive eigenvalues need not be small.
- Uncertainty about the clean endpoint is exactly what supplies the expansive contribution.
- This does not apply unchanged to arbitrary couplings, non-Gaussian sources or paths. It does not require a hidden P to be symmetric. Composition, coordinate changes and output projections separate hidden matrices from the data-space Jacobian.
- Symmetric instantaneous Jacobians do not imply a symmetric finite-time transport Jacobian: matrices encountered at different times/states need not commute.

## 5. Our existing targets specify which directions must shrink or expand

For target `N(m,Sigma)`, Gaussian conditioning gives

```text
C_t = (1-t)^2 I + t^2 Sigma
v*(x,t) = m + [t Sigma-(1-t)I] C_t^-1 (x-tm).
```

In a covariance eigen-direction of variance lambda, the velocity derivative is

```text
a_lambda(t) = [t*lambda-(1-t)] / [(1-t)^2+t^2*lambda].
```

It changes sign at `t=1/(1+lambda)`. Our target has variances 0.09 and 4. At t=0.5, the exact derivatives are **-1.669725 and +1.2**: one direction contracts while the other expands. An everywhere contractive *data-space velocity field* cannot match this target field. This is not a proof that a constrained hidden layer prevents the full network from matching it.

For Ring8, all component variances equal sigma^2=0.04. Let

```text
c(t) = (1-t)^2+t^2*sigma^2
w_k(x,t) proportional to exp(-||x-t*mu_k||^2/[2c])
m_w = sum_k w_k mu_k
C_mu = sum_k w_k (mu_k-m_w)(mu_k-m_w)^T
a(t) = [t*sigma^2-(1-t)]/c(t).
```

Expanding each Gaussian component's conditional velocity and averaging gives

```text
v*(x,t) = a(t)*x + (1-t)/c(t)*m_w(x,t)
J_v*(x,t) = a(t)*I + t*(1-t)/c(t)^2*C_mu(x,t).
```

Proof of the second line: `grad w_k = (t/c)w_k(mu_k-m_w)`, hence `J_m_w=(t/c)C_mu`. This also agrees with the general posterior-covariance formula, since

```text
Cov(x1|x_t=x) = sigma^2*(1-t)^2/c * I + (1-t)^4/c^2 * C_mu.
```

Exact cases at t=0.5 from the current radius-3, eight-component dataset:

| State x | Eigenvalues of J_v* | Meaning |
|---|---|---|
| center `(0,0)` | 14.795858, 14.795858 | Strong outward separation locally |
| intermediate mode center `(1.5,0)` | -1.499636, 0.441038 | Anisotropic; not yet fully contracting |
| midpoint of neighboring intermediate centers | -1.719076, 3.335650 | Shrink in one direction, separate in the other |

At t=0.8, a mode center contracts in both directions (about -2.561), whereas the adjacent midpoint has eigenvalues about -2.561 and 46.443. Those midpoint/center examples can have low probability, especially late in time. They establish local geometry, not their contribution to average loss or generated sample mass. Any learned-model comparison must weight or stratify by actual path probability as well.

For a posterior concentrated on two components, `C_mu` is approximately `w(1-w)(mu1-mu2)(mu1-mu2)^T`, a rank-one directional correction. This gives a concrete *hypothesis* for scalar gating: it might cheaply allocate local directional sensitivity. The original hidden correction `b u^T` is neither automatically symmetric nor aligned with this data-space direction. The correspondence must be measured through the full network; similarity of algebraic rank is not a proof of suitability.

## 6. Why smaller internal P or smaller average field error does not settle generation

Consider exact trajectories `dX/dt=v*(X,t)` and `dY/dt=v_theta(Y,t)` from the same initial state. Suppose the learned field has an integrable one-sided Lipschitz bound ell(t) on the region connecting the trajectories:

```text
<y-x, v_theta(y,t)-v_theta(x,t)> <= ell(t)||y-x||^2.
```

Set e=Y-X and delta(t)=`v_theta(X,t)-v*(X,t)`. Then

```text
d||e||/dt <= ell(t)||e|| + ||delta(t)||
||e(1)|| <= integral_0^1 exp(integral_t^1 ell(u)du) * ||delta(t)|| dt.
```

For smooth fields, `ell(t)` may be bounded using the largest eigenvalue of `sym(J_v_theta)`, rather than its full singular norm. Under finite second moments, Minkowski gives the analogous L2 bound and hence an upper bound on endpoint W2 under this coupling. This is conditional on the stated regularity, not a global guarantee for an arbitrary neural network.

This shows what a stability-oriented design would have to relate: **field error, its time/state location, and full-field amplification**. It does not establish a monotone relation between an internal P norm and empirical SW2. The external numerical solver adds a separate discretization error.

First-order endpoint perturbation further has `e(1) approximately integral Phi(1,t)delta(t)dt`, where `Phi` solves the variational equation along a reference trajectory. Direction and cancellation matter. An average oracle MSE alone discards them. This is an explanation of why rankings can differ, not evidence that this particular mechanism caused our observed ranking.

## 7. The link back to JiT must respect its prediction parameterization

Our toy model directly outputs v. Current JiT outputs a clean-data estimate D_theta, then the code computes, for `c_eps(t)=max(1-t,t_eps)`,

```text
v_theta(x,t) = [D_theta(x,t)-x]/c_eps(t)
J_v_theta = [J_D_theta-I]/c_eps(t)
loss = E ||D_theta(x_t,t)-x1||^2 / c_eps(t)^2.
```

At each fixed t, positive weighting preserves the population conditional-mean minimizer D*. But finite-capacity shared-parameter optimization and time sampling differ from direct-v regression. Near the clamped terminal interval this implemented velocity is not exactly the unclamped independent-linear-path velocity even at D*. The code also uses a logistic-normal time distribution, class conditioning, attention and RMSNorm/AdaLN. These matter before extrapolating a toy result.

Therefore a credible JiT argument must connect the original block to conditional denoising or full-field geometry, not equate each hidden FFN with a physical step in generation time. See [JiT, Algorithm 1 and its bottleneck study](https://arxiv.org/html/2511.13720v2).
