# From the FM objective to a covariance-shaped ODE field

Status: the implemented candidate and controls completed Stage9 and failed its continuation rule; see covariance-readout.md. No useful architecture has yet been established. This derives a testable change to P, rather than assuming that the numerical ODE interpretation guarantees good prediction.

## 1. What the present objective actually asks for

Let Y be a target sample and Z an independent standard Gaussian. Our path is X=aZ+tY, a=1-t. For0<t<1, conditioning on X=x eliminates Z:

```text
v*(x,t) = E[Y-Z | X=x] = (m(x,t)-x)/a,
m(x,t) = E[Y | X=x].
```

The posterior law is proportional to exp(-||x-tY||²/(2a²)) p1(dY). Differentiating its normalized expectation gives, coordinate by coordinate,

```text
d m_i / d x_j = (t/a²) [E(Y_i Y_j | x)-m_i m_j].
J_v* = -I/a + (t/a³) Cov(Y | X=x).
```

Thus J_v* is symmetric and bounded below by -I/a in the positive-semidefinite order. It need not be positive, contractive, or uniformly bounded as t approaches an endpoint. The within-component covariance of a smooth target can also cancel apparent singular terms; the displayed formula is used only in its stated open-time interval.

This calculation uses independent isotropic Gaussian source noise. It does not use optimal transport. It also holds when p1 is the finite empirical target measure, because the same derivatives apply to a finite weighted sum. The implementation samples fresh Gaussian noise at each update, so caching target samples does not by itself invalidate this identity. A generic non-Gaussian source or target-dependent affine noise schedule need not share it.

For another direct check, define

```text
Z_t(x) = E_Y exp((t*x^T*Y - t²*||Y||²/2)/a²).
Psi_t(x) = (a/t) log Z_t(x) - ||x||²/(2a).
```

Then grad Psi_t=v*. The covariance is the Hessian of a log-partition function. The non-isotropic part represents posterior uncertainty about the target, not arbitrary damping. This is why simply making all singular values small or equal is not an objective-derived constraint.

## 2. The hidden-space qualification

The complete backbone includes learned embeddings, LayerNorm, time modulation, nonlinear compositions, residuals and an output projection. A hidden P is not J_v*. Making P symmetric, or making each internal field a potential gradient, does not make the complete velocity conservative. Earlier derivative residuals were mainly symmetric anyway; eliminating curl alone could leave most error untouched.

The proposed test transfers one algebraic pattern into the internal field. Whether this is useful in hidden coordinates is the hypothesis. It is not a consequence of the data-space theorem.

## 3. An explicit field replacing P*h

Use H groups of M learned row vectors, collected in U_h, and a fixed positive temperature T. Let q_h=softmax(U_h h/T) and q0=1/M. Define

```text
f(h) = gamma*h + sum_h U_h^T (q_h-q0),
Phi(h) = gamma*||h||²/2
         + sum_h [T logsumexp(U_h h/T) - mean(U_h h)].
```

Gamma is an unconstrained learned scalar, so the field permits an isotropic negative part. The centering makes f(0)=0, as in the original bias-free P*h. Direct differentiation yields

```text
f = grad Phi,
J_f(h) = gamma*I + (1/T) sum_h U_h^T C(q_h) U_h,
C(q) = diag(q)-q*q^T.
```

C(q) is a categorical covariance, hence positive semidefinite. Its row absolute sums are2q_i(1-q_i)<=1/2, so its operator norm is at most1/2. Stacking all U_h into U therefore gives

```text
gamma <= eigenvalue(J_f(h)) <= gamma + ||U||_2²/(2T).
```

This is an analytical bound in terms of learned weights, not a training cap. The covariance contribution has rank at most H(M-1), which is28 for H=4,M=8. It is not a full-network rank bound. The basis vectors are learned but fixed within a layer; probabilities can change the local mixture and its principal directions within their span. This does not create an unrestricted input-dependent basis.

The origin linearization is

```text
P0 = gamma*I + sum_h U_h^T (I-11^T/M) U_h/(M*T),
f(h) = P0*h + O(||h||²).
```

Replacing softmax-q0 by its exact origin tangent therefore recovers a structured version of the professor's linear field. This is the direct connection to the original P, rather than labeling a general MLP an ODE.

## 4. Keep the original first-order solve and its composition

Retain the original positive vector controller, independent at each solve:

```text
D0 = diag(.2+.8*sigmoid(g(h)/10)),
F(h) = h+D0*f(h),
FFN(h) = F2(SiLU(F1(h))).
```

The internal autonomous field for one solve is y'=D0*f(y), with D0 computed once at the input and held fixed during the solve. At K1, its update is the displayed F. With finite learned weights f is globally Lipschitz; the fixed-D0 ODE has a well-defined finite-time flow. K1 is an approximation and inherits no automatic invertibility certificate.

At a frozen field anchor, D0*J_f is similar to the symmetric matrix sqrt(D0)*J_f*sqrt(D0), so its eigenvalues are real. Neither their signs nor the stability of the Euler update are guaranteed. When differentiating the learned layer with respect to its input, the actual formula is

```text
J_F = I + D(h)*J_f(h) + diag(f(h))*J_s(h).
```

The last term and multiplication by D generally destroy symmetry. Tests explicitly check this counterexample. Do not describe this as a conservative complete velocity model or a guaranteed stable integrator. No K2 training, new loss, spectral cap or range alteration is included.

### A separate energy statement for each tied update

The input-dependent gate still allows a useful pointwise statement. Write Phi=gamma*||h||²/2+A(h), where A is convex, and define E=-Phi. For u=D(h)*f(h), the Bregman identity gives

```text
E(h+u)-E(h) = -f(h)^T*u - (gamma/2)*||u||² - B_A(h+u,h),
B_A >= 0.
```

Let L=max(-gamma,0) and s_max be the largest entry of D at this input. Since ||D*f||²<=s_max*f^T*D*f,

```text
E(F(h))-E(h) <= -(1-L*s_max/2)*f(h)^T*D(h)*f(h).
```

Thus L*s_max<2 is a sufficient strict descent condition away from stationary points. It does not involve a cap on U's singular values. No derivative of D is needed for this one-step inequality: it evaluates the gate once at h. The proof also holds for the tied linear control, and does not hold in general for the untied decoder. This is an internal energy statement at frozen learned weights, not monotonic training loss. Different layers and the middle SiLU do not share a common E. E can be unbounded below, so descent does not by itself guarantee bounded iterates, attractor preservation or good samples.

The condition has a direct failure example: U=0,gamma=-4,D=.8I gives F(h)=-2.2h and increases E=2||h||². Gamma remains unconstrained in the screen; log the actual energy changes and sufficient-condition margin rather than adding a new clipping rule. Both the exact Bregman identity and this counterexample are tested. The log-sum-exp retrieval algebra is known from associative-memory models; it is not claimed as a new general-purpose field construction. [Recent numerical-analysis comparison](https://arxiv.org/html/2608.21304v1).

## 5. Controls that distinguish the hypothesis

1. **Tied nonlinear field:** the construction above.
2. **Untied nonlinear field:** replace the decoder U^T by an independent V, initialized exactly to U^T. Every initial hidden function and surrounding tensor matches; after training J_f=gamma*I+V*C(q)*U/T need not be symmetric. A win shared by both variants does not establish a covariance/gradient benefit.
3. **Linearized tied field:** replace softmax-q0 by (logits-mean(logits))/M. The initial value/Jacobian at zero match the nonlinear field, while the internal field remains P0*h at every input. This separates nonlinear covariance variation from a static factorization.
4. **Original vectorK1 and both SwiGLU budgets:** retain the actual parent and practical references. A gain over the parent alone is insufficient.

The budget fixes H=4,M=8,T=1 before outcomes. Thirty-two encoder rows are copied from the corresponding reference P initialization, with no extra random draws. All original controllers and surrounding tensors remain paired. At dim2 the tied/linearized networks have69576 parameters versus69570 for small SwiGLU; untied has81864 versus81792 for full SwiGLU. All three new arms use80192 linear MACs, close to full SwiGLU80000. Softmax and actual runtime must be measured separately; parameter savings alone are not a compute advantage.

If the tied nonlinear arm does not beat its controls and practical references, this proposed transfer of posterior geometry is unsupported. If untied wins instead, report that result as evidence against the tying constraint. If linearized matches it, nonlinear covariance adaptation has not earned its complexity. Full completion still requires a frozen independent holdout, longer-budget validation and actual cost evidence.
