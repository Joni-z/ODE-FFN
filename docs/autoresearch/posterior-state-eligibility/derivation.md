# Finite posterior-state evolution and its ordinary realization

Zhizhe Zhang · October 28, 2026

## Target
Derive an exact finite posterior state that retains the third-moment term
missing from a covariance-only homotopy. Determine whether integrating it
provides a distinct computation beyond an ordinary static nonlinear map.
This is algebraic eligibility, not evidence of improved image generation.

## Status
**COHERENT AS STATED** for a fixed, known finite Gaussian-mixture prior.
[Fixed numerical checks](results/summary.json) pass. The mathematical representation alone does not
select a new ODELayer, learn an unknown image prior, or justify training.

## Invariant object
The normalized posterior law
`q_lambda(dx|y) = exp(-lambda*||x-y||²/2) q(dx) / Z_lambda(y)`.
For lambda>0 this is conditioning under Gaussian observation noise of variance
1/lambda. At lambda=0 it is the prior, defined by the same normalized tilt,
not conditioning on a proper infinite-variance Gaussian random variable.

## Assumptions and notation
- The fixed prior is `q = sum_j pi_j N(mu_j, Sigma_j)`, with finitely many
  components, positive normalized pi and positive-semidefinite symmetric
  Sigma. Atoms and singular components are allowed. No parameter is fitted.
- y is a d-vector, lambda>=0 is noise precision. The first evolution holds
  **y fixed**. Component posterior mean/covariance are b_j/S_j; posterior
  weights are w_j. Global mean/covariance are m/C.
- `kappa = E[(X-m)*||X-m||² | y]` is a vector, not the full third tensor.
  Transpose products use column-vector convention; I is the d-by-d identity.
- The later FM conversion uses `Z_t=tX+(1-t)e`, independent standard Gaussian
  e and interior 0<t<1. A moving z(t) requires an extra observation term.

## Derivation strategy and map
1. Condition each prior component and normalize its likelihood.
2. Differentiate its finite state and combine all component moments.
3. Recover the complete global precision derivative, including kappa.
4. Transform the actual observation and precision to FM coordinates.
5. Eliminate the solver and expose the exact ordinary endpoint map.
All displayed equalities below are identities under the stated fixed prior.
No Gaussian approximation of a non-Gaussian posterior is introduced.

## Main derivation

### 1. Exact component conditioning

Define `A_j=(I+lambda Sigma_j)^-1`. This inverse exists even if Sigma_j is
singular. Completing the square, or conditioning a joint Gaussian, gives

```text
S_j = A_j Sigma_j
b_j = mu_j + lambda S_j (y-mu_j)
ell_j = log(pi_j) - .5 logdet(I+lambda Sigma_j)
        - .5 lambda (y-mu_j)^T A_j (y-mu_j)
w_j = softmax(ell)_j
m = sum_j w_j b_j
C = sum_j w_j [S_j + (b_j-m)(b_j-m)^T].
```

The observation-density factor common to all components cancels from w.
For singular components, the expressions follow from Gaussian conditioning
or the continuous positive-noise limit; no inverse of Sigma_j is needed.

### 2. Closed finite state at fixed observation

Since A_j is a function of Sigma_j, it commutes with Sigma_j. Differentiate
the inverse and use `y-b_j=A_j(y-mu_j)`:

```text
S_j' = -S_j²
b_j' = S_j (y-b_j)
ell_j' = a_j = -.5 [trace(S_j) + ||y-b_j||²]
w_j' = w_j [a_j - sum_k w_k a_k].
```

The prime is d/dlambda with y fixed. The last equation is normalized
component-weight evolution. Initial states at lambda=0 are Sigma_j, mu_j
and pi_j. Log weights can be integrated instead of probabilities, retaining
normalization by softmax without probability clipping. These states close
because the *entire fixed finite family* is retained, not because global
mean and covariance determine every posterior.

### 3. Where the missing third moment goes

Put `d_j=b_j-m`. Gaussian component moments give

```text
kappa = sum_j w_j [d_j*||d_j||² + d_j*trace(S_j) + 2 S_j d_j]
m' = sum_j (w_j' b_j + w_j b_j')
   = C (y-m) - .5 kappa.
```

The second equality also follows by differentiating the normalized tilt:
`m'=-.5 Cov(X,||X-y||²|y)`. It is the same identity as
[cycle22](../cycle22-denoiser-dynamics/derivation.md), now represented by
finite component state. A single Gaussian has kappa=0; a mixture need not.

The existing counterexample is exact: at y=0,lambda=1, posterior masses
(.5,.5) on (-1,1) and (.8,.2) on (-.5,2) both have m=0,C=1. Their kappa
values are 0 and 1.5, and their precision drifts are 0 and -.75. Each is
realized by a fixed prior proportional to those masses times exp(x²/2).
Carrying only m/C would predict the same drift and therefore cannot implement
both posteriors. Carrying finite component state retains the missing term.

### 4. The full FM chain rule

For fixed z and interior t, set `y=z/t`, `lambda=t²/(1-t)²`. Then
`D(z,t)=m_lambda(y)` is the clean conditional mean, and

```text
J_y m = lambda C
J_z D = (lambda/t) C
lambda_dot = 2t/(1-t)³
partial_t D = lambda_dot [C(y-m)-.5 kappa] - (lambda/t) C y.
```

The final term arises because y changes with t. Holding y fixed while
varying lambda is not holding an FM input z fixed. Along an arbitrary
differentiable z(t), the complete derivative is

```text
dD/dt = lambda_dot [C(y-m)-.5 kappa]
        + lambda C [z_dot/t-z/t²].
```

The same result follows by differentiating the component states along that
path, with `y_dot=z_dot/t-z/t²`:

```text
dS_j/dt = -lambda_dot S_j²
db_j/dt = lambda_dot S_j(y-b_j) + lambda S_j y_dot
dell_j/dt = lambda_dot a_j - lambda (y-b_j)^T y_dot.
```

For the uncapped marginal velocity `z_dot=(m-z)/(1-t)`, this simplifies to
`dD/dt = t/(1-t)³ [C(y-m)-kappa]`. It is generally nonzero. When the original
denominator cap is active, use the general chain rule with the capped
velocity; do not apply that uncapped simplification. Formulas using z/t are
not evaluated at t=0; neither boundary singularity is a hidden-layer design.

### 5. Eliminate the internal solver

**Atoms.** With all Sigma_j=0, component values stay mu_j. Cancel the common
term `-.5 lambda ||y||²` from logits:

```text
w_j = softmax_j[(lambda y)^T mu_j + log(pi_j)-.5 lambda ||mu_j||²]
m = sum_j w_j mu_j.
```

This is prototype dot-product attention with a precision-dependent key bias.
The conventional attention scale can be absorbed into its query. It is not
necessarily a standard self-attention layer over existing image tokens.

**Shared covariance.** Let `A=(I+lambda Sigma)^-1`, common across j. Then

```text
q = lambda A y
beta_j = log(pi_j)-.5 lambda mu_j^T A mu_j
w_j = softmax_j[q^T mu_j + beta_j]
m = (I-A)y + sum_j w_j A mu_j.
```

This is a time-conditioned linear term plus prototype attention. Its
parameters are the same fixed prior parameters; no integration is required.
This identity is not permission to add a noisy-input skip to JiT silently.

**Unequal covariances.** The common quadratic term no longer cancels. Logits
are input quadratics, while b_j are input-affine experts. The exact endpoint
is therefore a quadratic-gated mixture of affine experts. It is **not in
general ordinary linear-query/fixed-key attention**: the Hessian of
`log(w_j/w_k)` is `-lambda(A_j-A_k)`, which may be nonzero. Explicit quadratic
features and conditional affine values realize the endpoint without an ODE.

## Remarks and interpretation

The finite state correctly repairs this fixed-family closure. However, the
proposed exact posterior endpoint has an ordinary direct expression using
the same prior parameters. Component covariance factors depend on lambda,
not on the input, and can be prepared/cached at fixed times in either
implementation. Calling a numerical integration of these known states a
new ODELayer does not by itself introduce a distinct operation. No measured
cost/quality advantage is established here.

The ordinary attention definition used for the operator comparison is
[Vaswani et al., Section 3.2.1, Eq. 1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1),
as a classical definition, not a frontier recommendation. The mixture
conditioning and equivalences above are derived here, with no novelty claim.

## Boundaries and non-claims

- Known-prior conditioning is not learning an unknown image conditional
  mean, and a hidden token is not automatically an observation of that prior.
- A direct expression does not prove every numerical approximation, learned
  posterior model or recurrent state is useless. A distinct approximation or
  restriction would require its own cost/error and task evidence.
- Reject only this solver repackaging as an ODELayer nomination. Do not infer
  an image-quality win, universal attention equivalence for unequal covariance,
  or a covariance/mixture rescue of the sealed Gaussian reference.
- Preserve all old training/evaluation, strong ordinary controls and stop rules.

## Open risks and check scope

The fixed numerical checks cover PSD components, mixture-weight normalization,
the third-moment contraction, fixed-observation versus moving-FM derivatives
and unequal-covariance logits. Their pass is mathematical identity evidence
only. The separate [actual hidden-input check](hidden-input-check.md) explains
why these known-prior statistics are not automatically JiT hidden statistics.
