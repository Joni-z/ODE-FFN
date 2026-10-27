# Prediction-space compatibility before another hidden ODE design

## Why this question follows the failed local-flow screens

Cycles04–06 did not supply a useful new ODE FFN. They separate exact flow algebra, accurate integration and actual regression quality. The old [head review](../../research-architecture-search/head-control-design-review.md) also left a clean/shifted/direct comparison unselected: output-envelope error was negligible on its full-dimensional tasks, and an uncapped zero clean head has infinite initial population loss under uniform time. Those conclusions remain intact.

The present question is different: does the original ODE/SwiGLU ranking depend on the joint prediction-space and input-geometry setting associated with JiT? All previous main targets have positive within-component variance in every ambient direction. Their full-support geometry does not test a target confined to a low-dimensional linear subspace. Adding a diagnostic benchmark does not erase their negative results or nominate a new architecture.

Three primary sources motivate this specific compatibility check:

- [Li and He, JiT v2, Section3.3](https://arxiv.org/html/2511.13720v2#S3.SS3) embeds2D data through a fixed unknown orthogonal projection, then varies ambient dimension under matched prediction/loss comparisons. This supplies direct precedent for the benchmark geometry, not a result about ODE FFNs.
- [Gagneux et al., v3, controlled Fourier data](https://arxiv.org/html/2603.06454v3) distinguishes parameterization from weighting and shows that dimensionality effects depend on architecture. It is an ICLR2026 DeLTa workshop paper, not a claim that one prediction head always wins.
- [Liang et al., Residual-Stream Burden, v2](https://arxiv.org/html/2609.33895v2) studies spectral concentration, residual-state capacity and decoupled paths. It is an October2026 preprint under review. We do not adopt its hyper-connections or transfer its ImageNet claims to our local FFN.

## Two populations with the same latent structure

Fix Q in R^(128 by2), Q^T Q=I, unknown to the model. Let H follow the existing radius3, eight-component2D ring with within-component covariance.04I. With N=I-QQ^T and independent standard G, define

```text
thin:  Y=QH
thick: Y=QH+.2*N*G.
X_t=(1-t)Z+tY, Z standard Gaussian independent of Y.
```

The laws differ only by orthogonal target variance q_N=0 versus.04, not by the latent components or basis. Normal energy and total target energy necessarily change; this is not an equal-energy or every-notion-of-intrinsic-dimension comparison. Float32 stored targets approximate the ideal subspace up to rounding, which is measured in preflight.

The posterior factorizes in tangent and normal coordinates. Write a=1-t, z=Q^T x, x_N=Nx. The latent posterior is the existing2D Gaussian-mixture posterior. In normal coordinates,

```text
d_N=a²+t²*q_N
k_N=t*q_N/d_N
m_N=k_N*x_N.
```

For thin data k_N=0, including its natural continuous extension at t=1. Thus the complete posterior mean has exactly rank-two spatial response for every interior time; the normal velocity response is still−I/a. For thick data the normal posterior-mean derivative is k_N I and becomes identity at t=1. A full-rank empirical covariance by itself would not disprove a nonlinear manifold; here the distinction follows from the actual specified support.

For the2D component means mu_i and posterior probabilities p_i proportional to `exp(-||z-t*mu_i||²/(2d))`, d=a²+.04t²,

```text
m_H=(t*.04/d)*z+(a²/d)*sum p_i*mu_i
Jm_H=(t*.04/d)I+(a²*t/d²)*Cov_p(mu_i)
m=Q*m_H+k_N*x_N
Jm=Q*Jm_H*Q^T+k_N*N.
```

These give a closed oracle in128D without revealing Q, components or oracle labels to the model.

## One capped regression target, three heads

Use delta=max(a,.05) and the same target `c_target=(a/delta)*(Y-Z)` in every arm. Its Bayes field is `c*=(m*-x)/delta`. With identical raw native network f_theta:

```text
clean:   c_theta=(f_theta-x)/delta
shifted: c_theta=f_theta-x/delta
direct:  c_theta=f_theta.
```

Clean and shifted share their zero-raw initial decoded field, but have parameter derivatives J_theta f/delta and J_theta f, respectively. Direct has zero initial decoded field. They are not an initialization- or optimizer-equivalent three-way test. The two skip heads share the same raw normalized information bottleneck and differ in time-dependent output scaling. No learned projection, skip or new loss is added.

The cap deliberately avoids the infinite expected zero-clean initial loss of unweighted uniform-time velocity supervision. It changes the endpoint-band regression field relative to old uncapped studies. No new value is pooled into their leaderboard, and this is not a faithful reproduction of JiT's logit-normal time sampling.

## A complete-network bottleneck bound, independent of P

Every native raw network first maps x to A*x+b in R^64, with no later access to raw x. For any learned A, the intersection `ker(A) intersect range(N)` has dimension at least128−64−2=62. Along its orthonormal directions, the Gaussian normal input is independent of A*x and of the latent target. A function of A*x therefore cannot recover those coordinates.

Let beta=(k_N−1)/delta and gamma=k_N/delta. Conditional normal input variance is d_N. The squared oracle-field risk per ambient coordinate has the universal population lower bounds

```text
direct:  MSE >= (62/128)*beta²*d_N
clean/S: MSE >= (62/128)*gamma²*d_N.
```

This follows by conditioning on A*x,t and retaining only the62 independent unobserved normal directions. It applies to arbitrary nonlinear computation after A, so modifying hidden P, adding Taylor terms or solving an internal ODE exactly cannot remove it. It does not bound empirical finite-bank error pointwise or identify all other optimization/capacity effects.

The final LayerNorm also has one exact uniform-shift null direction, so each raw spatial Jacobian has rank<=63. Projecting onto the126D normal space and using the best-rank approximation bound gives the per-anchor Frobenius-error floors

```text
direct:  ||Jc_theta-Jc*||_F² >=63*beta²
clean/S: ||Jc_theta-Jc*||_F² >=63*gamma².
```

These are architecture-wide local derivative bounds, not learned-head envelope bounds. For thin data gamma=0: the skip heads remove this particular obstruction. Direct thin risk has uniform-time population floor `(62/128)*(29/30)=0.468229...`. The result predicts a generic head effect, not an ODE advantage. In thick data gamma is nonzero, especially near clean time; a clean prediction head is no universal cure.

## Sampling caveat

Heun32 integrates the same capped decoded field for all heads. This field is not the original uncapped probability-flow velocity in the final5% of time. Even the exact thin oracle leaves normal standard deviation `.05/e` in continuous time, rather than reaching an exactly singular endpoint. Save oracle-Heun32 samples and metrics as a numerical/decoder reference. Generated SW2 cannot be interpreted as pure network error or an unbiased estimate of exact target recovery.
