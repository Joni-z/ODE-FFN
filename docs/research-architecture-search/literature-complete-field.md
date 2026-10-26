# Complete-field geometry and the JiT parameterization gap

Reviewed 2026-10-24 after Stage20 failed. No architecture or training selected.

**The original FM motivation does imply a precise constraint on the complete denoiser. It does not imply that a hidden P should have equal singular values, that the velocity Jacobian should be positive, or that a conditional hidden ODE is the appropriate implementation.** Before designing another block, distinguish that theorem from the prediction parameterization actually used by JiT and by our synthetic model.

## What the code trains

Use Y for clean data, Z for independent standard Gaussian noise, X for the noisy input, and a=1-t.

| Property | Local JiT path | Current synthetic architecture studies |
|---|---|---|
| Path | `X=t*Y+a*noise_scale*Z` | `X=t*Y+a*Z` |
| Direct network output | Clean-data prediction `m_theta` | Velocity prediction `v_theta` |
| Base regression | `||(m_theta-Y)/delta||²`, `delta=max(a,t_eps)` | `||v_theta-(Y-Z)||²` |
| Training times | Logit-normal | Uniform |
| Conditioning | Labels, with label dropout | Unconditional |
| Sampling velocity | `(m_theta-X)/delta`, with optional CFG | Direct network output |

Sources: [local JiT denoiser](../../src/denoiser.py), lines73–86 and `_forward_sample`; [synthetic trainer](../../synthetic/train.py), lines315–327. The [image configuration](../../configs/swiglu.yaml) uses noise scale1, `t_eps=.05`, and logit-normal parameters `(-.8,.8)`. Optional soft-Lipschitz/perceptual losses exist; only the base FM term is considered here.

The [official JiT implementation](https://github.com/LTH14/JiT/blob/main/denoiser.py#L45) agrees. The toy studies did not port JiT's complete prediction/training setup. For `a<.05`, JiT obeys `m_theta=X+delta*v_decoded`, while the underlying path's Bayes velocity still uses a. Clipping changes the endpoint field.

## The complete-field constraint, derived in our coordinates

For the independent Gaussian path and `0<t<1`, the posterior is

```text
p(Y|X=x) proportional to exp(-||x-t*Y||²/(2*a²)) p1(Y).
m*(x,t) = E[Y|X=x]
v*(x,t) = (m*(x,t)-x)/a.
```

Differentiating the normalized posterior expectation gives

```text
J_m* = (t/a²) Cov(Y|X=x)          symmetric positive semidefinite
J_v* = -I/a + (t/a³) Cov(Y|X=x).
```

The denoiser is the gradient of a convex log-partition potential. It is therefore monotone at fixed t; the velocity is conservative with `J_v* >= -I/a`. Negative velocity eigenvalues are valid. There is no universal upper spectral cap, isotropy requirement, or contractivity guarantee.

Our [existing derivation](covariance-operator-mathematics.md) also applies to the empirical cached target measure, so this condition cannot distinguish generalization from memorization. For isotropic source variance `sigma_Z²`, `J_m*=t*Cov/(a²*sigma_Z²)` and `J_v*=-I/a+t*Cov/(a³*sigma_Z²)`; the toy source has variance1. Fixed class conditioning preserves the argument; general couplings/non-Gaussian sources need another derivation.

A hidden symmetric generator does not ensure a symmetric complete Jacobian after projections and composition. An arbitrary PSD matrix field need not integrate to one potential. Conservative instantaneous velocity does not establish optimal transport: successive Jacobians need not commute. CFG with w>1 forms `(1-w)*m_uncond+w*m_cond`, which need not preserve PSD.

## Three primary sources that change the interpretation

**JiT: prediction space is an architectural choice.** Li and He's *Back to Basics: Let Denoising Generative Models Denoise*, inspected v2 (7 January 2026), separates prediction space from loss space in Section3.2/Table1. Its large-patch results support clean-data prediction in that setting. They do not establish a covariance-constrained architecture or an internal-ODE advantage. Head parameterization must remain distinct from an FFN comparison. [Paper, Section3.2](https://arxiv.org/html/2511.13720v2#S3.SS2).

**Parameterization has no universal winner.** Gagneux et al., *Training Flow Matching: The Role of Weighting and Parameterization*, inspected v3 (17 September 2026), distinguishes `D=N_theta` from `D=Id+(1-t)N_theta` separately from time weighting. Section5 finds architecture/locality and data regime affect their ordering. JiT's conclusion therefore cannot automatically transfer to our low-dimensional network. Status: **ICLR2026 DeLTa workshop**, not main conference. [Paper, Section3/Table1 and Section5](https://arxiv.org/html/2603.06454v3#S3); [record](https://arxiv.org/abs/2603.06454).

**Measure the complete denoiser Jacobian, but retain signed geometry.** Graikos, Jojic and Samaras, *On the spectral properties of generative denoiser Jacobians*, is a 28 September 2026 preprint. AppendixA, Eq.(24), gives exactly the covariance identity above. Its learned-model study associates stronger principal responses with improved generation and studies input-perturbation regularizers; it does not justify shrinking every direction. A material caveat: Section3.2 admits learned Jacobians need not be symmetric/PSD, yet uses a real-positive spectral convention and Algorithm1 returns `||Jv||`. That cannot identify negative or skew components. Our 2D/16D setting permits exact complete Jacobians, so there is no reason to adopt that approximation or its positivity convention. Its regularizers would also change our objective. Full primary methods/appendix were inspected; no acceptance claim is made. [Paper, Section3.2](https://arxiv.org/html/2609.36210v1#S3.SS2); [AppendixA](https://arxiv.org/html/2609.36210v1#A1).

## What follows before another design

For an existing synthetic checkpoint, define the implied denoiser without retraining:

```text
m_theta = x+a*v_theta
A = J_m_theta = I+a*J_v_theta.
```

At fixed t, errors simply scale by a, including Jacobian errors. The useful question is how much lies outside the symmetric PSD set. With `S=(A+A^T)/2`, `K=(A-A^T)/2`, matrix algebra gives

```text
distance(A, PSD)^2 = ||K||_F² + sum_{eigenvalue(S)<0} eigenvalue(S)^2.
```

This local distance is not an achievable architecture improvement. Pointwise projection need not integrate to a denoiser or improve sampling. [Stage9](analysis-covariance/analysis-report.md) rejected a hidden covariance block; it neither enforced nor disproved the complete-output condition.

Changing direct prediction from v to m would be a separate intervention. Under the same unclipped path, keeping velocity loss requires `||m_theta-Y||²/a²`; plain denoising MSE changes time weighting. A fresh zero m-head gives initial velocity `-x/a`, whereas the current zero v-head gives zero velocity. Keeping `m_theta=x+a*N_theta` exactly preserves the original velocity parameterization and does not itself add the constraint. Time distribution, denominator clipping, noise scale, initialization, and auxiliary losses must therefore remain explicit before attributing any gain to an ODE block. The exact oracle is suitable for evaluating these identities; this review does not select oracle-supervised training.

The next question is whether complete-output violations materially explain existing errors. Neither answer selects an architecture or revives Stage9/Stage20.

All three papers' relevant primary full text was inspected. Only this review file was written.
