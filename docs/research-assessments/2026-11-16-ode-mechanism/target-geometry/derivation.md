# Derivation package: the FM target before the hidden generator

## Target

Derive the geometry of the optimal clean-data predictor and velocity under
the repository's independent Gaussian-source flow matching. Determine what
this implies for an ODE-inspired FFN and what still requires measurement.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.** The constrained object is the
whole Bayes predictor. A constraint on an internal P does not follow from it.
The covariance identity is established prior work, not our new theorem.

## Invariant object

The full conditional-mean predictor m(z,t)=E[X|Z_t=z]. P is an internal model
parameter; it is not m, its Jacobian, or the physical velocity field.

## Assumptions

- X has finite second moment; epsilon is independent standard Gaussian.
- Z_t=alpha_t X+beta_t epsilon, with alpha_t>0 and beta_t>0 at the times used.
- Regression uses squared error with a positive scalar weight depending on t.
- For the repository convention alpha=t, beta=(1-t)sigma0, sigma0>0, 0<t<1.
- Conditioning, when present, is fixed. CFG extrapolation is a different map.
- Hidden feature time s and generative interpolation time t are distinct.

## Notation

C=Cov[X|Z_t=z]; score=grad_z log p_t(z); J_m and J_v differentiate complete
outputs with respect to z at fixed t; I is the data-space identity matrix.
All displayed Jacobians use column-vector differentiation, even where code
stores batches as row vectors.

## Derivation strategy

Differentiate the Gaussian likelihood, then the normalized posterior mean.
Convert the predictor to velocity using the unchanged native loss. Finally
state the observable comparison and construct controls against overinterpretation.

## Derivation map

1. Gaussian likelihood gives score=(alpha*m-z)/beta^2.
2. Differentiating the normalized posterior gives J_m=alpha*C/beta^2.
3. The straight path gives v=(m-z)/(1-t), hence J_v=(J_m-I)/(1-t).
4. Hidden-P assumptions do not transfer through input conditioning and readout.
5. Measure complete Jacobian error; symmetry/PSD alone do not certify samples.

## Main derivation

### Step 1 — identity: differentiate the likelihood

The likelihood is proportional to exp(-||z-alpha X||^2/(2 beta^2)). Its
log derivative is (alpha X-z)/beta^2. Averaging under the posterior gives

```text
score = (alpha*m-z)/beta^2
m = (z+beta^2*score)/alpha.
```

### Step 2 — proposition: covariance is the full predictor derivative

Differentiating the normalized posterior expectation subtracts the product
of its two first moments. Coordinate by coordinate,

```text
d m_i / d z_j = alpha/beta^2 * (E[X_i X_j|z]-m_i*m_j)
J_m = alpha/beta^2 * C.
```

Gaussian smoothing and the moment assumption justify the interior-time
derivatives. For positive alpha, J_m is symmetric positive semidefinite.
Equivalently m is the gradient of
psi=||z||^2/(2 alpha)+(beta^2/alpha)log p_t(z), which is convex under this model.
PSD does not impose an upper eigenvalue bound of one.

This is [Wan et al., ICML2025](https://proceedings.mlr.press/v267/wan25e.html),
Appendix C Proposition C.2; the earlier [arXiv v2 Section3.1 Proposition3.2](https://arxiv.org/html/2412.18730v2#S3.SS1)
states the same identity and attributes it to generalized Tweedie identities.

### Step 3 — identity: convert to the native velocity convention

For Z=tX+(1-t)sigma0 epsilon, the conditional path derivative satisfies

```text
v*(z,t) = (m(z,t)-z)/(1-t)
J_v* = t/[sigma0^2*(1-t)^3] * C - I/(1-t).
```

Thus J_v* is symmetric but need not be PSD. Both expansion and contraction
are permitted. The equivalent score expression is
v*=z/t+sigma0^2*(1-t)*score/t.

The copied JiT loss in root denoiser.py forms z=t*x+(1-t)*e, predicts x_pred,
and compares (x_pred-z)/a with (x-z)/a, where a=max(1-t,t_eps). It is exactly
image squared error weighted by 1/a^2. Its conditional Bayes optimum remains m.
This does not make finite image- and velocity-prediction networks equivalent.
The Gaussian source scale must remain explicit; synthetic studies use one.

### Step 4 — counterexamples: do not constrain the wrong object

1. A hidden SPD generator followed by a negative linear decoder has a negative
   full derivative. Hidden flow structure alone does not imply output PSD.
2. For scalar P=.5 and s(h)=.2+.8*sigmoid(20(h+1)), the exact conditional map
   h*exp(P*s(h)) has negative derivative at h=-1. Bounded positive P and clock
   do not eliminate the derivative of the input-dependent conditioning.
3. An anisotropic Gaussian prior can have a Bayes denoiser eigenvalue above
   one. A universal contraction/spectrum-flattening rule would exclude its target.

### Step 5 — counterexamples: geometry does not certify generation

For Gaussian data and unit Gaussian source, v*=a(t)z where
a(t)=(2t-1)/[(1-t)^2+t^2]. Adding a constant bias leaves its Jacobian identical
but shifts the final mean. Conversely adding omega*Kz, K skew, preserves the
isotropic covariance path and final Gaussian law while changing the Jacobian.
Therefore target Jacobian matching is an explanatory hypothesis about the
regression solution, not a necessary and sufficient condition for generation.

## Remarks and interpretation

The prior matched order result motivates a measurable question: does complete
velocity-Jacobian error reproduce the opposite Ring8/Product16 sampling effects?
The registered primary metric is mean ||J_v-J_v*||_F^2/d on common interpolation
states. This measures average squared response error to a unit isotropic direction.
Symmetry and the negative spectrum of sym(J_m) are secondary descriptions.

## Boundaries and non-claims

The identities hold for independent isotropic Gaussian corruption and the
conditional squared-error optimum. They do not apply automatically to arbitrary
couplings, non-Gaussian corruption, CFG extrapolation, or a hidden P. Interior-time
formulas must not be numerically substituted at t=1. SVD and eigenvalues are
different for a nonsymmetric learned matrix; test skew separately.

## Open risks

A derivative diagnostic on interpolation states may not explain errors on
generated trajectories. A matched direction would justify a causal intervention,
not establish it. A failed direction rejects this registered explanatory metric
at this checkpoint/budget; it does not prove all target geometry irrelevant.
