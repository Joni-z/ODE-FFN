# A Gaussian model of denoising and its precise relation to ODELayer

Date: 2026-10-25. These are worked consequences of Gaussian conditioning, squared regression, and matrix polynomials. They are not claimed as new mathematical theorems or as evidence of an ODE-FFN performance advantage.

The formulas use column vectors and population distributions. This note analyzes the professor's original layer supplied in the conversation, rather than substituting a later variant. The accompanying verification program performs symbolic identities and deterministic small-matrix checks; it does not train a model.

## 1. A distribution that can be solved exactly

Let X be clean data and E be independent standard Gaussian noise in R^d:

```text
X ~ N(0, Sigma),   Sigma >= 0
E ~ N(0, I),       X independent of E
Z_t = t X + b E,   b = 1-t,   0 <= t < 1
C_t = Cov(Z_t) = t^2 Sigma + b^2 I
```

C_t is positive definite for t<1 even when Sigma is singular. Write Sigma = Q diag(lambda_i) Q^T with orthogonal Q and lambda_i >= 0. A subspace-supported example has r positive eigenvalues and d-r zero eigenvalues. Thus its data dimension is r while the noise occupies all d directions. The choice of coordinates is immaterial: an orthogonal rotation preserves every result below.

The principal example is `Sigma=diag(4,1,0)`: an elliptical Gaussian in the first two coordinates of three-dimensional space. This isolates the structural question without introducing mixture assignment, numerical integration error, or neural optimization.

This is the independent coupling used in the JiT interpolation, not an optimal-transport coupling. The idealized formulas below use an unclipped denominator. JiT's practical denominator clipping is addressed in Section 12.

## 2. Why the optimal prediction is a conditional mean

For any measurable prediction f(Z_t,t) with finite risk, let D*(z,t)=E[X|Z_t=z]. Conditioning on Z_t gives the orthogonal decomposition

```text
E ||f(Z_t,t)-X||^2
  = E ||f(Z_t,t)-D*(Z_t,t)||^2
    + E ||X-D*(Z_t,t)||^2.
```

The cross term is zero because E[X-D*|Z_t]=0. Therefore the conditional mean is the unique squared-loss minimizer up to sets of Z_t probability zero. A positive time-only weight changes the weighting of different times, but not the unrestricted optimum at each fixed time.

For the pair velocity U=X-E and any image prediction f, the interpolation implies

```text
U = (X-Z_t)/b
v_f = (f(Z_t,t)-Z_t)/b
||v_f-U||^2 = ||f(Z_t,t)-X||^2/b^2.
```

Thus ideal x-prediction with v-loss, as in JiT, has the same pointwise optimal D*. Finite network capacity can make the choice of prediction and time weighting important; the Bayes identity does not establish equivalence between practical neural models.

## 3. Deriving the Gaussian posterior without assuming full-rank data

Cross-covariance is Cov(X,Z_t)=t Sigma. Set

```text
A_t = t Sigma C_t^(-1)
R = X-A_t Z_t.
```

Then Cov(R,Z_t)=t Sigma-A_t C_t=0. The pair (R,Z_t) is jointly Gaussian, so uncorrelatedness implies independence, including when R is degenerate. Since E[R]=0, this proves

```text
D*(z,t) = A_t z = t Sigma C_t^(-1) z
Cov(X|Z_t=z) = Sigma-t^2 Sigma C_t^(-1) Sigma
             = b^2 Sigma C_t^(-1).
```

The last equality follows from C_t=t^2 Sigma+b^2 I and the commutation of Sigma and C_t. The posterior covariance is independent of z in this Gaussian case.

As a useful independent check, the denoising-posterior Jacobian identity under Gaussian corruption is

```text
J_D*(z,t) = t/b^2 * Cov(X|Z_t=z) = A_t,  0<t<1.
```

This is a rescaling of the standard posterior-mean derivative identity, not a new identity specific to ODELayer. Its symmetry and positive semidefiniteness concern the complete data-space denoiser.

## 4. The optimal velocity and unavoidable pair-label variance

The squared-loss optimum for pair velocity is

```text
v*(z,t) = E[X-E|Z_t=z]
        = (D*(z,t)-z)/b
        = B_t z
B_t = (t Sigma-b I) C_t^(-1)
A_t-I = b B_t.
```

To verify the matrix formula, expand

```text
t Sigma-C_t = b (t Sigma-b I).
```

The posterior pair-label covariance is

```text
Cov(U|Z_t=z) = Cov(X|Z_t=z)/b^2 = Sigma C_t^(-1).
```

Consequently the minimum image-pair MSE is `tr(b^2 Sigma C_t^(-1))`, and the minimum velocity-pair MSE is `tr(Sigma C_t^(-1))`. Oracle field MSE measures excess error above this Bayes risk; it should not be confused with the training loss against a randomly paired U.

For reference, allowing a nonzero mean m changes the formulas to

```text
D*(z,t) = m+t Sigma C_t^(-1)(z-tm)
v*(z,t) = m+(t Sigma-b I) C_t^(-1)(z-tm).
```

The centered case suffices to analyze P without conflating translation with directional transformation.

## 5. What each covariance direction requires

In a covariance eigen-direction of variance lambda, define

```text
c_lambda(t) = t^2 lambda+b^2
a_lambda(t) = t lambda/c_lambda(t)          # image-prediction gain
beta_lambda(t) = (t lambda-b)/c_lambda(t)   # velocity derivative
```

The exact relations are

```text
a_lambda-1 = b beta_lambda
beta_lambda = (1/2) d/dt log(c_lambda)
beta_lambda changes sign at t=1/(1+lambda), when lambda>0.
```

For a zero-variance direction, `a_0=0` and `beta_0=-1/b`. It is entirely removed by optimal image prediction and continuously contracted by the sampling field.

For a positive-variance direction, the velocity contracts early, reaches zero at the stated threshold, and expands later. For example, at t=1/2 with Sigma=diag(4,1,0):

| Direction | Clean variance lambda | Noisy variance c | Denoising gain a | Velocity derivative beta |
|---|---:|---:|---:|---:|
| First data direction | 4 | 5/4 | 8/5 | 6/5 |
| Second data direction | 1 | 1/2 | 1 | 0 |
| Normal direction | 0 | 1/4 | 0 | -2 |

Thus `D*(z,1/2)=(1.6 z1,z2,0)` while `v*(z,1/2)=(1.2 z1,0,-2 z3)`. Even this simple target does not request uniform contraction or equal singular values in the original z coordinates.

The factor a can exceed one because Z_t contains a scaled-down signal. This does not say that denoising always amplifies noise. In normalized coordinates `Y=Z_t/t=X+(b/t)E`, the corresponding gain is `t*a_lambda=t^2 lambda/c_lambda`, which lies in [0,1]. Coordinate normalization matters when interpreting a spectral constraint.

For a linear subspace S with projector Pi and normal projector N=I-Pi, `Sigma=lambda Pi` gives

```text
D*(z,t) = a_lambda(t) Pi z
v*(z,t) = beta_lambda(t) Pi z - N z/b
        = -z/b + [a_lambda(t)/b] Pi z.
```

This factorization separates a fixed geometric operator (the subspace projector) from time-dependent scalar coefficients. It is an exact structural description, not a reason that an ODE parameterization must outperform an ordinary linear map.

## 6. The sampling flow map is a different object

To generate samples, solve `dZ/dt=B_t Z`. All B_t commute in this Gaussian example because they are functions of the same Sigma. Moreover,

```text
B_t = (1/2) C_t^(-1) dC_t/dt.
```

Therefore the exact flow from external time u to time t, with u,t<1, is

```text
Phi_(u->t)(z) = C_t^(1/2) C_u^(-1/2) z
             = exp( integral_u^t B_r dr ) z.
```

Direct differentiation verifies the ODE and Phi_(u->u)=I. Its covariance is

```text
Phi_(u->t) C_u Phi_(u->t)^T = C_t.
```

Since C_0=I, `Phi_(0->t)(e)=C_t^(1/2)e`. As t approaches 1, it converges to `Sigma^(1/2)e`, which has the target distribution, even for a singular Sigma. At t=1 a singular target is reached as a limit; the smooth invertible-flow statement only applies before that endpoint.

Crucially, D*(z,t) is not Phi_(t->1)(z). At t=0 the optimal image prediction is zero, while the sampling endpoint is `Sigma^(1/2)z`, usually nonzero. D* averages possible clean endpoints conditional on the current noisy observation; the flow map transports an individual state along the marginal velocity. Straight conditional training paths also need not equal trajectories of this marginal flow.

In the isotropic subspace case, the finite-time generator is

```text
log Phi_(u->t) = (1/2) log[c_lambda(t)/c_lambda(u)] Pi
                + log[(1-t)/(1-u)] N.
```

The two directional coefficients generally evolve independently. A single scalar times a fixed matrix does not reproduce this whole family.

## 7. What the original ODELayer actually computes

The professor's code computes its scalar once and reuses it in the recurrence:

```text
s(h) = 0.2+0.8 sigmoid(g(h)/10)
F_K(h) = T_K(s(h)P) h
T_K(q) = sum_(k=0)^K q^k/k!.
```

Here P has no bias and `0.2<s(h)<1` for finite g. By induction, term k is `s(h)^k P^k h/k!`. For each initial h, freezing this duration gives a truncated evaluation of `exp(s(h)P)h`.

This is an internal hidden-feature computation. It is not automatically a step of the external sampling ODE. Nor is it the second-order Taylor solver of `dh/du=s(h)Ph` with the gate reevaluated along the evolving state; that nonlinear ODE also has a gate-derivative term in its second derivative.

The infinite matrix-exponential series converges for every finite P. Constraints on P address amplification, truncation accuracy, optimization, or the function class; they are not required merely to make the infinite series exist.

## 8. The exact P correspondence at one fixed external time

The following comparisons deliberately use one layer directly in data coordinates (`h=z`) with a constant internal duration s. They do not assert the same restriction for an entire Transformer with embeddings, conditioning, activations, and output projections.

### 8.1 K=1 and direct clean-data prediction

Matching `F_1(z)=(I+sP)z` to D*=A_t z gives exactly

```text
P_D(t) = (A_t-I)/s = (b/s) B_t.
```

This is the cleanest algebraic link to the physical velocity generator. P equals a scaled B_t only under the stated identification and at one fixed t. With unrestricted real P, K1 represents every linear denoiser at that time, just as a free linear weight A does. It has no greater linear expressivity.

If instead the layer is meant to be an Euler sampling step of duration Delta, matching `I+sP=I+Delta B_t` gives `P=(Delta/s)B_t`. If it directly outputs a velocity through `I+sP`, the matching equation is instead `P=(B_t-I)/s`. The output's meaning must be specified before interpreting P.

### 8.2 Exact exponential and clean-data prediction

For t>0 and positive definite Sigma, A_t is positive definite, so one exact choice is

```text
P_D(t) = log(A_t)/s.
```

It is generally not B_t or a scalar multiple of B_t. If Sigma has a nullspace, A_t is singular, so no finite matrix exponential can equal A_t. Strong compression may approximate that map, or a separate projection can provide the rank loss.

For the actual Gaussian sampling flow between two fixed times, a valid exponential generator is instead

```text
P_flow(u,t) = [log(C_t)-log(C_u)]/(2s).
```

These three matrices—velocity generator, denoiser logarithm, and integrated flow generator—are distinct.

### 8.3 K=2 is a polynomial parameterization, with an important obstruction

Put Q=sP and let R=I+Q+Q^2/2. Then

```text
R = [(I+Q)^2+I]/2.
```

Matching the optimal image map A_t=I+b B_t therefore requires

```text
(I+sP)^2 = I+2b B_t.
```

When the right-hand side is positive semidefinite, a real symmetric choice is `P=[sqrt(I+2b B_t)-I]/s`. It is a matrix function of B_t rather than B_t itself. In a normal direction the right-hand side equals -1, exposing the real scalar/symmetric obstruction immediately.

If P is symmetric, every eigenvalue of R is at least 1/2, since

```text
1+q+q^2/2 = 1/2+(q+1)^2/2,   q real.
```

So a symmetric K2 layer directly predicting the image cannot exactly remove a normal direction. Making a negative eigenvalue very large in magnitude eventually increases this gain instead of decreasing it: at q=-4 the polynomial gain is 5 while exp(q) is approximately .0183. This is a truncation issue, not the behavior of an exact dissipative linear flow.

The lower bound MUST NOT be claimed for unrestricted real P. For example,

```text
Q = [[-1,-1],[1,-1]]
I+Q+Q^2/2 = 0.
```

Its complex conjugate eigenvalues are -1+i and -1-i. A real matrix can therefore annihilate a two-dimensional subspace through polynomial cancellation. Its exact exponential remains nonzero and invertible. Such a fit would not establish that the layer approximates the exponential accurately.

There is nevertheless a useful unrestricted obstruction. If R=A_t, then Q commutes with A_t because A_t is a polynomial in Q. Hence Q preserves each eigenspace of the symmetric A_t. On an eigenspace with denoising gain a,

```text
(I+Q)^2 = (2a-1)I.
```

If a<1/2 and that eigenspace has odd dimension m, taking determinants would require the square of a real determinant to equal `(2a-1)^m<0`, an impossibility. For a<1/2 and even multiplicity, real 2x2 blocks give a construction: `Q=-I+sqrt(1-2a) J`, where `J^2=-I`. For a>=1/2, the scalar real root `Q=(-1+sqrt(2a-1))I` suffices. Combining eigenspaces gives a necessary and sufficient condition for this symmetric target A to have a real K2 preimage: every eigenvalue below 1/2 has even multiplicity.

In the three-dimensional example at t=1/2, the zero-gain eigenspace has dimension one. Thus no real P, symmetric or otherwise, lets one direct K2 layer equal that denoiser at that time.

This is not solely an artifact of exactly singular data. Replacing the normal clean variance by .01 gives the distinct normal denoising gain `2/101`, still below 1/2 with multiplicity one; the same direct-layer obstruction holds. The exact rank-loss statements elsewhere do require zero variance and do not apply unchanged to this thickened distribution.

## 9. One fixed P and one scalar clock do not encode all external times

First suppose P is meant to be the velocity generator, `B_t=s(t)P`. With a positive scalar s(t), the directional signs of this matrix cannot change independently. Our Gaussian signal direction changes from contraction to expansion while its normal direction keeps contracting. Thus this identification fails across time even though every B_t shares an eigenbasis.

A stronger finite-polynomial statement holds for a direct denoising layer, and also allows an input-dependent gate. Assume:

- Sigma has at least one positive eigenvalue and one zero eigenvalue;
- P is one fixed finite real d-by-d matrix;
- K is finite and `s(z,t)` is differentiable in z and continuous in t;
- `F(z,t)=T_K(s(z,t)P)z` has no additional embeddings, offsets, projections, or layers, and the time interval is contained in (0,1).

Suppose F equals the exact linear denoiser A_t z for every z and every t in a connected open time interval. Equality almost everywhere for continuous F implies equality everywhere because Z_t has a positive density on all R^d. Differentiating at z=0 gives

```text
A_t = J_F(0,t) = T_K(s(0,t)P).
```

Since A_t is singular for all those times,

```text
det T_K(s(0,t)P) = 0.
```

But `det T_K(qP)` is a polynomial in the scalar q, equal to 1 at q=0. It is not the zero polynomial, and so has only finitely many real roots. A continuous s(0,t) taking values in this finite set must be constant. Then A_t would also be constant. This contradicts the nonconstant positive-variance gain `a_lambda(t)`; its derivative is

```text
d a_lambda/dt = lambda [1-(1+lambda)t^2]/c_lambda(t)^2,
```

which cannot vanish on an open interval when lambda>0.

Therefore no single such layer exactly represents the entire subspace-Gaussian denoiser family. The theorem does not quantify approximation error and does not rule out the original multi-layer FFN or a full JiT. It prevents identifying one fixed hidden P with the complete physical denoising family without further structure.

## 10. Input-dependent gates and hidden coordinates change the interpretation

For the actual original layer, differentiation gives

```text
J_F(h) = T_K(sP) + [P T_(K-1)(sP) h] grad(s)^T,   K>=1.
```

The second term has rank at most one, but need not be symmetric or small. For K1 it is `(Ph)grad(s)^T`; for K2 it is `[(P+sP^2)h]grad(s)^T`. Restricting P alone therefore does not impose the same spectral restriction on the full layer Jacobian. The sigmoid supplies the pointwise bound `||grad(s)|| <= .02 ||grad(g)||` with the professor's defaults, but it does not bound the learned g network's gradient.

Even replacing T_K by exp does not make the input-dependent-duration map automatically invertible:

```text
det J_[exp(s(h)P)h] = exp(s tr(P)) [1+grad(s)^T P h].
```

The second factor can vanish. Fixed-time ODE flow properties cannot be transferred automatically to an input-dependent stopping time or the intervening SiLU in ODEFFN.

A concrete noninjective example in one dimension is `P=-1` and `s(x)=.2+.8 sigmoid(5(x-1))`. For `F(x)=x exp(-s(x))`, exact differentiation gives `F'(1)=0` and `F''(1)=-exp(-.6)<0`. This is a strict local maximum, so the map is not injective in a neighborhood. This example supplies more than a zero-Jacobian observation alone, which would not by itself prove noninjectivity.

If hidden coordinates are h=Lz for a fixed invertible linear L, the corresponding velocity generator is `L B_t L^(-1)`. A symmetric data-space B_t need not be symmetric in these hidden coordinates. It is self-adjoint in the metric `G=L^(-T)L^(-1)` instead. Arbitrary learned nonlinear embeddings and non-square input/output projections weaken a direct interpretation of hidden P further.

The complete original block is `F_2(SiLU(F_1(h)))`. Its Jacobian is the product of the two native layer Jacobians and the SiLU derivative. Neither this product nor a Transformer containing it is a single `T_K(sP)`, a single exponential, or the data-space B_t.

There is also a temporal distinction even if h=z and P is deliberately identified with B_t. The actual external Gaussian ODE has the local expansion

```text
Z_(t+Delta) = [I+Delta B_t+(Delta^2/2)(B_t^2+B'_t)] Z_t
              + O(Delta^3).
```

Freezing B_t and using T_2(Delta B_t) omits B'_t. It is not a second-order method for a time-varying field in general. In a normal direction the exact flow gain is `(b-Delta)/b=1-Delta/b`; the frozen K2 gain is `1-Delta/b+Delta^2/(2b^2)`. Here `beta'_0+beta_0^2=0`, so the apparent extra quadratic term actually departs from the exact normal flow. This argument concerns a literal physical-generator interpretation; the learned hidden P remains free to serve a different role.

## 11. A separate, exact connection to optimization

At one fixed t, fit a free linear image predictor A by population squared loss. Up to its Bayes-risk constant,

```text
L(A) = (1/2) tr[(A-A_t) C_t (A-A_t)^T]
grad_A L = (A-A_t) C_t.
```

For constant s and K1, `A=I+sP` is an affine reparameterization of every real A. Euclidean gradient flow on P induces

```text
dA/dtau = -s^2 grad_A L.
```

Thus in this setting it changes the learning-rate scale, not expressivity or directional preconditioning. This claim does not describe weight decay, a trained gate, or AdamW exactly.

For K2,

```text
L_P(H) = D_P A[H] = s H+(s^2/2)(P H+H P)
dA/dtau = -L_P L_P^*(grad_A L).
```

This is a nontrivial parameterization-induced preconditioner, but it can be degenerate: at `P=-I/s`, L_P is zero. In one eigen-direction the loss derivative is

```text
d/dp [ (c/2)(1+sp+s^2 p^2/2-a)^2 ]
  = c (1+sp+s^2 p^2/2-a) s(1+sp).
```

It vanishes at p=-1/s even when the target gain is not 1/2. Having a polynomial derived from an exponential does not by itself guarantee a favorable optimizer geometry.

## 12. Boundaries, practical JiT clipping, and what follows next

At t=0 the conditional mean and Gaussian velocity have well-defined values `D*=0` and `v*=-z`. Formulas dividing by t should not be used there. At t=1 a subspace target has singular covariance and the normal velocity coefficient diverges; sampling reaches the singular endpoint as a limit.

JiT clips its conversion denominator, by default to .05. If the same denominator d(t)=max(1-t,.05) is used for both the predicted velocity and paired target, the training loss is `||f-X||^2/d(t)^2`, and D* remains the unrestricted optimum. The converted field is then `(D*-z)/d(t)`, which differs from the exact interpolation velocity by the factor `b/d(t)` in the clipped region. The exact flow in Section 6 describes the unclipped idealization. It is not a proof of the endpoint distribution of a clipped numerical sampler. No JiT code was needed or inspected for this note; these distinctions follow from the paper's algorithm and the specified algebra.

The derivation supports the following conclusions:

1. There is a precise K1 connection `P=(b/s)B_t` for one direct data-space denoising layer at one time. A hidden P in the actual architecture has not thereby been identified with that B_t.
2. The target requests directional and temporal structure. Equal singular values or an everywhere negative physical generator do not follow from it.
3. Finite Taylor maps and exact exponentials have different contraction and rank properties. Symmetrizing K2 would remove some polynomial-cancellation mechanisms rather than merely stabilize them.
4. Even a Gaussian target generally needs more than one scalar coefficient to describe its directional time evolution. This motivates separating learned geometry from temporal gains; it does not prove a particular neural architecture is better.
5. This Gaussian target is linear. A time-conditioned linear oracle represents it exactly without an internal ODE. It is a calibration problem, not a convincing benchmark for ODE superiority.

No new training is launched on the strength of these facts alone. The next justified work, before another architecture sweep, is to use the analytic targets to distinguish three questions: whether a proposed block can represent the required maps, whether its parameterization makes those maps easier to fit at matched capacity, and whether any benefit remains for a nonlinear target. A useful experiment would need a concrete prediction beyond what a free time-conditioned linear map already provides. A new training recipe or architecture is not selected in this note.

## Sources and relation to existing project notes

- [Li and He, *Back to Basics: Let Denoising Generative Models Denoise*](https://arxiv.org/html/2511.13720), Sections 3.2 and 4.3: prediction space, loss space, interpolation, and conversion to a sampling field. The Gaussian calculation and P representability arguments above are worked derivations, not results attributed to JiT.
- [Manor and Michaeli, *On the Posterior Distribution in Denoising: Application to Uncertainty Quantification*](https://arxiv.org/html/2309.13598v2), Theorem 2: the posterior-mean Jacobian and covariance identity for Gaussian corruption. Scaling its observation gives the version used here.
- Existing project notes already derived the general posterior/velocity identity and the K2 real-matrix cancellation example: [mathematical reassessment](../research-2026-10-21-reassessment/mathematics.md) and [objective bridge](../research-2026-10-23-direction/objective-bridge.md). This note specializes and completes the Gaussian conditioning-to-flow calculation, separates the three possible P interpretations, and proves the direct-layer restrictions with explicit scope. It does not relabel earlier calculations as a new discovery.
