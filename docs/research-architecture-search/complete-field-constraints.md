# Constraints on the complete flow-matching field

This is a mathematical prerequisite for another design, not a proposed architecture or experiment. Stage20 stopped after its audited negative result. Bounds on a hidden P have repeatedly failed to imply useful task behavior; the relevant question is what the **complete input-to-velocity map** should represent.

## Assumptions and the target being characterized

Use the independent coupling already used by the toy training objective:

```text
Z ~ N(0,I), Y ~ mu, Z independent of Y
a = 1-t, X_t = a Z + t Y, U = Y-Z
v(x,t) = E[U | X_t=x].
```

Assume E||Y||^2 is finite. Unless an endpoint is explicitly discussed, 0<t<1. The Gaussian convolution gives a positive smooth density p_t(x), including when mu is an empirical target bank. All spatial formulas below are for the population minimizer of this particular coupling's squared velocity loss. A finite trained network need not satisfy them exactly. Replacing the empirical bank by its generating population changes mu and generally changes this minimizer.

Let m(x,t)=E[Y|X_t=x] and C_Y(x,t)=Cov(Y|X_t=x). Since U=(Y-X_t)/a,

```text
m = x + a v.
```

This denoiser is a complete data-space map; it is not an intermediate hidden activation.

## Posterior differentiation gives the Jacobian restriction

The posterior measure has density, relative to mu,

```text
q_t(dy|x) proportional to exp[-||x-t y||^2/(2a^2)] mu(dy).
partial_xj log q_t(y|x) = (t/a^2) [y_j-m_j].
```

Differentiating the posterior mean therefore gives, with J_ij=partial_xj f_i,

```text
Jm = (t/a^2) C_Y
Jv = -I/a + (t/a^3) C_Y.
```

Thus Jm is symmetric positive semidefinite, Jv is symmetric, and every eigenvalue of Jv is at least -1/a. In particular, m is globally monotone:

```text
[m(x,t)-m(z,t)] dot (x-z) >= 0.
```

The velocity may expand strongly. These identities do not require negative velocity eigenvalues, nonexpansive denoising, or a small operator norm.

There is an explicit potential. Define the finite positive integral

```text
L_t(x) = E_mu exp[(t/a^2) x dot Y - (t^2/(2a^2)) ||Y||^2]
Phi_t(x) = (a^2/t) log L_t(x).
```

The exponent is at most ||x||^2/(2a^2), so L_t exists for every x even without an ordinary moment-generating function for mu. It is a log-MGF of a Gaussian-tilted target measure, up to an x-independent constant. Direct differentiation gives

```text
m = grad_x Phi_t,       Hess_x Phi_t = (t/a^2) C_Y >= 0
v = grad_x [Phi_t/a - ||x||^2/(2a)].
```

Equivalently, if score_t=grad_x log p_t,

```text
score_t = -(x-t m)/a^2
v = x/t + (a/t) score_t.
```

The symmetry is global integrability on R^d, not just a bound on the symmetric part of a generally nonsymmetric Jacobian. Conversely, an arbitrary convex potential at each t is not automatically the posterior potential of one fixed mu across time. The identities are necessary restrictions, not a complete characterization of an admissible target distribution or a guarantee of better approximation.

## PSD does not supply an upper Lipschitz bound

Already for Y~N(0,1) in one dimension,

```text
Jm = t / [(1-t)^2+t^2].
```

At t=.9 this is 45/41 >1. Hence the correct denoiser is not generally nonexpansive.

Finite target moments alone do not even ensure a finite global upper Jacobian bound for a fixed target. One explicit one-dimensional example has atoms y_n=2^n, n>=0, with probabilities proportional to exp(-y_n^2); it has finite moments of every order. At a fixed interior time, posterior weights are proportional to

```text
exp[B x y_n - A y_n^2],   B=t/a^2, A=1+t^2/(2a^2).
```

Choose x_n=A(y_n+y_(n+1))/B. The two adjacent atoms have equal posterior weight. Any other atom y has relative weight exp[-A(y-y_n)(y-y_(n+1))], so its total contribution vanishes as n grows. Consequently C_Y(x_n,t) is asymptotic to (y_(n+1)-y_n)^2/4 and is unbounded. Any useful global upper bound must state additional assumptions on the target and time interval. The lower bound -1/a also deteriorates near t=1.

## Straight supervised pairs do not imply zero field acceleration

For each sampled pair, U is constant and X_t=Z+tU has zero second time derivative. The conditional mean field mixes these different velocities at the same location. Its characteristics need not have constant velocity or be affine in t.

Write C_v=Cov(U|X_t=x), and define matrix divergence by
`[div A]_i=sum_j partial_xj A_ij`. Free transport of the joint (X_t,U) law gives its first two moment equations:

```text
partial_t p + div(p v) = 0
partial_t(p v) + div[p(v v^T + C_v)] = 0.
```

Subtracting v times the continuity equation gives the complete material acceleration:

```text
partial_t v + Jv v = -(1/p) div(p C_v).
```

Conditioning U=(Y-x)/a and using the Jacobian identity verifies the coefficients:

```text
C_v = C_Y/a^2 = [I+a Jv]/t.
```

These equations hold locally on the smooth interior, with the transport derivation also valid in weak form for an empirical target. The right side is generally nonzero. Imposing zero material acceleration would impose an extra condition that the present independent-coupling target need not satisfy. A straight-path/optimal-transport argument cannot silently replace this coupling.

### Exact Gaussian check of the sign and coefficients

For Y~N(0,q), q>0, in one dimension, define D=a^2+t^2 q and k=(t q-a)/D. Then

```text
p_t = N(0,D), v=k x, C_Y=a^2 q/D, C_v=q/D
Jm=t q/D
partial_t v + Jv v = (k'+k^2)x = (q/D^2)x
-(1/p) partial_x(p C_v) = -(q/D)(-x/D) = (q/D^2)x.
```

The two sides agree. For q=1 at t=.5, v=0 but material acceleration is 4x. Characteristics are x(t)=sqrt(D(t))*x(0), not affine functions of t. In anisotropic Gaussian examples different coordinates have different scale factors and the spatial curve can also bend. The displayed rational identities were independently simplified symbolically to zero; no training or test framework was added.

## Endpoints and the actual JiT output convention

At t=0, independence gives m(x,0)=E[Y], v(x,0)=E[Y]-x, Jm=0 and Jv=-I. The formula C_v=[I+aJv]/t must be interpreted by a limit there; direct conditioning gives C_v=Cov(Y).

At t=1, direct conditioning gives m(Y,1)=Y and v(Y,1)=Y because E[Z]=0. For the project's smooth positive Gaussian-mixture populations, the interior local limits are m(x,t)->x and v(x,t)->x, with Jm->I and Jv->I. General atomic or restricted-support targets do not justify those interior derivative limits or a globally smooth endpoint extension. Expressions dividing by a should not be evaluated at t=1.

The repository's [JiT implementation](../../src/denoiser.py) predicts x and divides both its prediction residual and pair target residual by delta=max(1-t,.05), using the [SwiGLU configuration](../../configs/swiglu.yaml). Its loss is weighted x-prediction MSE with weight 1/delta^2. It also samples sigmoid-Gaussian times, whereas the toy samples uniform times. These conventions differ from the toy's direct velocity head. At the ideal conditional-mean solution, the capped decoded field is

```text
v_delta = (m-x)/delta = (a/delta) v.
```

It therefore differs from the true interpolation velocity in the capped endpoint band. This distinction concerns the objective/output convention, not an FFN advantage. In the uncapped case, merely writing `m_theta=x+a*N_theta` and then decoding `(m_theta-x)/a` gives exactly N_theta; that is no new architecture. A native zero x-head instead initializes velocity to -x/a (or -x/delta), whereas a zero velocity head initializes it to zero. Cross-head comparisons cannot be called controlled FFN comparisons without accounting for these differences. An x-prediction head alone also does not guarantee a symmetric PSD Jm.

## Why the previous hidden constructions do not establish these properties

- For the professor's K1 map F(h)=h+s(h)P h, `JF=I+sP+(P h)(grad s)^T`. Symmetric P does not make the last term symmetric. Bounding eigenvalues or singular values of P also omits the clock response.
- A monotone diagonal hidden activation has a diagonal Jacobian in its own coordinates. Independent input/output projections, normalization, conditioning and the surrounding residual network do not automatically preserve symmetry or monotonicity of the final data-space map.
- Even an exact hidden gradient-flow endpoint need not itself be a gradient map. Its variational equation is `dJ/du=H(u)J` with symmetric H(u); products of symmetric exponentials are generally nonsymmetric when their factors do not commute. An internal gradient vector field and an integrable endpoint map are different claims.
- Stage9's tied covariance-like field constrained a hidden operator inside the existing architecture. Its negative result tests that construction. It does not directly test enforcing `Jm=(Jm)^T>=0` on the complete denoiser, and its internal energy descent did not establish the identities above.

Before another ODE-related design is eligible for an experiment, its derivation should identify the complete predicted map and prove the intended property through **all** projections, gates, time conditioning and residuals. It must distinguish exact auxiliary flows from their implemented truncations; state endpoint/time-domain assumptions; and preserve the existing coupling and objective unless a separately controlled change is explicitly selected. A restriction should permit the expansion and nonzero acceleration that the true field requires.

That is a decision boundary, not a winning architecture. A mathematically valid construction would still need a clear computational purpose, the strong SwiGLU references and a generic control separating the property from its ODE parameterization. These derivations authorize no new architecture, loss, oracle labels or training run.
