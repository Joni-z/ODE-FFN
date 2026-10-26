# Gaussian physical-time flow: a useful identity, not a selected architecture

**NO to selecting a hidden ODELayer design from this formula alone.** The exact Gaussian response concerns physical data coordinates and flow-matching time. It does not identify the appropriate latent generator, feature basis or internal solve duration. One raw-affine error decomposition could falsify a narrower motivation, but is neither implemented nor selected here. No clock-collapse outcomes enter this review.

## Exact field and characteristics

Assume independent `Z~N(0,I)`, `Y~N(mu,Sigma)`, fixed `Sigma` symmetric positive definite, and `0<=t<=1`. Set `a=1-t`, `X_t=aZ+tY`, `U=Y-Z`. Gaussian conditioning gives

```text
C(t)=a²I+t²Sigma
Cov(U,X_t)=tSigma-aI
A(t)=(tSigma-aI)C(t)^-1
v*(x,t)=mu+A(t)(x-t*mu).
```

Since `C'=2(tSigma-aI)`, `A=.5*C'*C^-1`. All these matrices commute: they are functions of the same fixed Sigma. The principal square root therefore gives the deterministic characteristic

```text
x(t)=t*mu+C(t)^(1/2)*z0,
x(t1)=t1*mu+C(t1)^(1/2)*C(t0)^(-1/2)*(x(t0)-t0*mu).
```

Differentiating the first expression verifies the field. This is a characteristic of the conditional-mean velocity, not one of the random straight training pairs. Commutativity is essential to this simple matrix differentiation; arbitrary learned time-dependent matrices do not inherit it.

For a covariance eigenvalue lambda>0, the variance, characteristic scale and instantaneous rate are

```text
c_lambda(t)=a²+t²lambda
r_lambda(t)=sqrt(c_lambda(t))
alpha_lambda(t)=(t*lambda-a)/c_lambda(t).
```

Every mode first contracts and then expands: its rate changes sign at `t*=1/(1+lambda)`, where its variance reaches `lambda/(1+lambda)`. It starts at variance1 and ends at lambda. A mode with lambda<1 thus expands late despite net endpoint contraction. Different eigenvalues turn at different times.

The rate itself need not be monotone:

```text
alpha_lambda'(t)=[lambda-((1+lambda)t-1)²]/c_lambda(t)².
```

For lambda<1 it first decreases, then increases; for lambda>1 it increases, then decreases. For lambda=1 it increases from−1 to1, although the characteristic scale still falls to `1/sqrt(2)` and returns to1. Identical source/target Gaussians therefore have nontrivial intermediate dynamics under this independent coupling.

With positive definite Sigma, `A(0)=-I`, `A(1)=I`, so `v*(x,0)=mu-x` and `v*(x,1)=x`. If Sigma is only positive semidefinite, interior formulas remain valid, but zero modes have rate `-1/(1-t)` and collapse at the endpoint. Do not invert C(1), claim an invertible endpoint transition, or transfer the nonsingular endpoint limits to those modes.

## A stronger projection identity, with a narrower interpretation

Gaussian Y is unnecessary for the **best affine approximation**. For any independent target Y with finite mean mu and covariance Sigma, the same covariance equations hold. For t<1, and also at t=1 when Sigma is positive definite, the least-squares affine projection of `v*(X_t,t)=E[U|X_t]` under the true path law is exactly

```text
v_aff(x,t)=mu+A(t)(x-t*mu).
v*(x,t)=v_aff(x,t)+r(x,t),
E[r(X_t,t)]=0,
E[r(X_t,t)*(X_t-t*mu)^T]=0.
```

Gaussianity makes r vanish. For a mixture, r is generally nonzero; the square-root characteristics above then describe only the Gaussian affine reference, not the full learned transport. These covariance statements also hold for a finite training bank using its own empirical moments, which need not equal the evaluation population moments.

This provides a legitimate affine-component motivation, but no automatic improvement. A global-covariance linear term may expand where a resolved mixture mode requires contraction; the nonlinear residual must cancel that term locally. Isolating the affine reference can therefore leave a harder residual rather than simplify optimization. Covariance alone does not determine posterior assignment or the nonlinear field.

## Why this does not select a hidden spectral block

The exact generator A(t) acts on `x-t*mu` in data space; its eigenvalues are rational functions of covariance eigenvalues. The original P acts on learned, normalized, time-conditioned hidden features. Those features are not certified Gaussian, their distribution changes during training, and their internal clock is not physical t. The backbone already has time-dependent FiLM. A formula for A(t) establishes neither a missing representational direction nor failure of that conditioning path.

A single fixed scalar generator with monotone positive elapsed time would have a fixed contraction/expansion sign, unlike the Gaussian mode above. That restricted construction cannot reproduce the whole physical-time characteristic. **This is not a lower bound for our model:** it predicts velocity through two solves, SiLU, residuals and FiLM, rather than directly enforcing that restricted characteristic.

Replacing a hidden Taylor polynomial by an exponential does not implement the physical square-root transition without also identifying its coordinates, spectrum and time dependence. The stopped diagonal-exponential, spectral-constraint, projection and head settings are not reopened. No oracle covariance, rotation, analytic initializer or new objective follows from this derivation.

## One possible falsifier, not a proposed training branch

The narrow prediction would be: **if failure to fit physical-time affine dynamics explains much of the parent’s gap, its excess error over matched SwiGLU should lie substantially in the raw-input affine component.** The existing good Gaussian-control performance already makes a blanket inability-to-fit-linear-fields story weak.

At each fixed-time common panel, form `Q=[1,X]` and residual matrix `E=v_model-v_oracle`. With the orthogonal projector `H=Q Q^+`,

```text
E_aff=H E;  E_nonlin=(I-H)E
||E||F²=||E_aff||F²+||E_nonlin||F².
```

Use the identical H for the parent and both SwiGLU references, retaining every time and signed contribution to their error gap. This is a projection of **error onto raw affine functions**, not the earlier fit of a final head to learned features. It uses oracle residuals descriptively; its coefficients cannot become training labels or initialization. Per-time fitted coefficients do not constitute one learnable time-shared branch.

A small affine excess would refute this explanation of most current error. A large one would localize a fitting deficit, not distinguish capacity, optimizer conditioning or nonlinear-feature interference. Moreover, H is unchanged by any invertible rotation of X, so this diagnostic cannot identify a preferred spectral basis. Same-panel subtraction would be a privileged fitted counterfactual, not achieved model improvement or holdout evidence.

Thus the Gaussian identity sharpens one question about error structure. It does not yet justify an additional time-linear branch, an ODE-specific architecture, or another experiment sequence. The useful-architecture objective remains unachieved.
