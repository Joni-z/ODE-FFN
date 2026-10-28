# Gaussian reference: derivation package

Zhizhe Zhang · October 28, 2026

## Target and status

**COHERENT AS STATED** for the conditional mean under the fitted positive
Gaussian prior. Its approximation of real image data is empirical. Derive
the predictor and its derivatives before assessing its quality; do not turn
the closed-form model into a theorem about neural ODE architectures.

## Invariant object, assumptions and notation

The invariant object is a normalized density `X~N(mu,Sigma)` and its conditional
mean under the unchanged coupling `Z_t=tX+(1-t)e`, with independent standard
Gaussian e. Here d=3072, t is in [0,1], and Sigma is strictly positive definite.
The orthonormal d-by-r basis U retains r training covariance eigenvalues l_i.
The positive scalar v is the average discarded variance; it is not a learned
clock, an eigenvalue floor or the synthetic oracle covariance.

## Derivation map

1. Estimate a constrained Gaussian density from training moments only.
2. Condition its joint Gaussian law at a fixed t.
3. Act in the retained subspace and its isotropic complement.
4. Differentiate the actual predictor; retain the sampler's existing cap.

## 1. Density fit — cited estimator, not a new theorem

For training mean mu and MLE covariance C (denominator n), probabilistic PCA
uses the leading r eigenpairs and the mean discarded eigenvalue v:

```text
v = (trace(C)-sum_i l_i)/(d-r)
Sigma = v I + U diag(l_i-v) U^T.
```

These are the classical maximum-likelihood estimators in
[Tipping/Bishop, Section 3.2, Eqs 7–8](https://www.cs.columbia.edu/~blei/seminar/2020-representation/readings/TippingBishop1999.pdf).
This statement concerns that Gaussian likelihood, not the optimal conditional
mean of the true CIFAR distribution. No FM labels or validation scores select
the estimator, rank or tail. Positive v and l_i make Sigma positive definite.

The stored scalar count is `d+(d+1)r+1`. With budget 185744, the largest rank is
59 and count 184380; rank 60 requires 187453. Orthonormality reduces the generic
dimension of the parameter family to 182610. Stored values, free dimensions,+float64 bytes and neural training computation must not be conflated.

## 2. Conditioning — exact identity for this prior

Joint moments give `E[Z_t]=t mu`, `Cov(X,Z_t)=t Sigma` and
`V_t=Cov(Z_t)=t² Sigma+(1-t)² I`. V_t is positive throughout the closed interval.
Gaussian conditioning therefore gives

```text
D(z,t) = mu + t Sigma V_t^-1 (z-t mu).
```

Put `a(l,t)=t*l/(t²*l+(1-t)²)` and `a0=a(v,t)`. Since U is orthonormal,
the production expression for `q=z-t mu` is

```text
D = mu + a0*q + U diag(a(l_i,t)-a0) U^T*q.
```

It uses two d-by-r matrix applications, or `2dr=362496` matrix MACs per image
at r=59, in float64. The remaining scalar work, casting and outer sampler are
separate. The isotropic reference uses r=0. No FFN or hidden feature dynamics
is present. An independent Woodbury solve checks the actual stored covariance
instead of reusing this spectral-gain formula.

## 3. State and time derivatives — exact identities

At fixed t, `J_D=t Sigma V_t^-1` is symmetric positive semidefinite, with
eigenvalues a(l_i,t) and a(v,t). It need not be contractive: for variance l>0,
its largest scalar gain over t is `(1+sqrt(1+l))/2`, achieved at
`t=1/sqrt(1+l)`. Thus even a correct Gaussian denoiser can have gain above 1.
This does not prescribe a constraint on any neural model's internal P.

For fixed z, let `A=t Sigma V_t^-1` and `V_t'=2t Sigma-2(1-t)I`. Then

```text
A' = Sigma V_t^-1 - t Sigma V_t^-1 V_t' V_t^-1
partial_t D = A'*(z-t mu) - A*mu
partial_t a(l,t) = l*(1-t²*(1+l))/(t²*l+(1-t)²)².
```

At t=0, D=mu; at t=1, D=z. These formulas are checked by independent dense
conditioning, Woodbury solves and state/time automatic differentiation on
fixed data-free Gaussian fixtures. Numerical agreement is not an image-quality
result or a proof that the real data are Gaussian.

## 4. Exact density does not imply an exact sampled endpoint

For the uncapped Gaussian marginal flow, the solution from a standard Gaussian
start e is `z_t=t mu+V_t^(1/2)e`, and at t=1 it has covariance Sigma.
Commutativity of V_t and V_t' verifies this solution directly.

The existing implementation instead uses
`v_cap=(D(z,t)-z)/max(1-t,.05)` and finite float32 Heun128. The cap changes
the field near the endpoint: at t=1 the capped velocity is 0, whereas the
uncapped limiting velocity is z. We keep that existing sampler. Directly
sampling `mu+Sigma^(1/2)e` would answer a different evaluation question and
is not substituted in this study. Small affine fixtures check the same
capped, discretized recurrence.

## Interpretation and non-claims

The Gaussian reference has no ODELayer P. A favorable score would show only
that this specified statistical model is competitive on these banks. It
would not prove the neural models learned a Gaussian law. An unfavorable
score would not prove that every Gaussian or affine predictor is inadequate:
rank, density-fitting objective and prior restrictions differ from neural FM.
An Inception feature-mean term is not a pixel-mean error. None of these
identities explains that term causally or nominates a new architecture.

## Open risks

Finite covariance precision, nearly degenerate eigenspaces, float64 predictor
versus float32 output/sampler and finite exposed evaluation banks are recorded.
No sign-sensitive eigenvector comparison is used. No failed reference is
rescued by rank, tail, mixture, prediction-target or sampler tuning.
