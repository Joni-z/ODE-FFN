# Clean-data head versus velocity head: bounded design review

**Decision: no immediate training selection.** The JiT/toy prediction-space mismatch is real and must limit our claims. A small comparison can test whether the existing ODE/SwiGLU ranking depends on that parameterization. It is not itself a new ODE architecture, and neither the literature nor the final-LayerNorm bound predicts an ODE benefit. The current read-only error/radius analysis should determine whether this is an informative compatibility check or a diversion.

The [local denoiser](../../src/denoiser.py), lines76–85, divides both residual prediction and target by `delta=max(1-t,.05)` under the [image configuration](../../configs/swiglu.yaml). JiT also uses logit-normal times. Keep those facts distinct from the toy's uniform-time, direct-velocity objective; see the [primary-literature review](literature-complete-field.md). This review changes no source, labels or experiments.

## One common target and three explicit heads

Keep the existing independent standard-Gaussian source, cached target bank, unconditional inputs, uniform t, and seed0. Write

```text
a=1-t, x=a Z+t Y, U=Y-Z, delta=max(a,.05)
c_target=(Y-x)/delta=(a/delta)U
N_theta(x,t)=the unchanged network's raw output.
```

For the proposed head comparison, every arm would minimize the **same** squared error to c_target. Use the same raw model tensors, RNG, data batches, optimizer and native FFNs. No extra head parameter or random draw is necessary.

| Head | Decoded field c_theta | Raw network target | Initial field when N=0 |
|---|---|---|---|
| Native clean-data X | `(N-x)/delta` | Y, with weight `1/delta^2` | `-x/delta` |
| Shifted velocity S | `N-x/delta` | `Y/delta` | `-x/delta` |
| Direct velocity V | `N` | `(Y-x)/delta` | 0 |

All three share the ideal decoded conditional mean `c*=(m*-x)/delta`, where m*=E[Y|x,t]. Their finite network classes and optimization are different. All require fresh training if this comparison is selected: the old direct-v trajectories used U, not c_target, in the endpoint band.

X versus S holds the initial decoded field fixed. If G=D_theta N, their parameter Jacobians are G/delta and G. Their initial tangent kernels therefore differ by the factors `1/(delta_i*delta_j)` between examples. At a shared zero-output initialization, for loss one-half squared error,

```text
g_X = -G^T Y/delta^2
g_S = -G^T Y/delta
g_V = -G^T (Y-x)/delta.
```

Thus X/S is not a pure function-class or pure conditioning isolation: reproducing the same subsequent field requires the input-dependent relation `N_S=N_X/delta`, which is not a single scalar rescaling of the finite network's parameters. Heterogeneous time factors are not removed by Adam's response to a common gradient scale; clipping also acts differently.

S versus V has the same parameter response G at shared raw parameters but a different initial field and translated function class. It measures the consequence of the known subtraction together with that initialization change. The three-way comparison identifies these distinctions more honestly than a native X versus old zero-V comparison alone.

Conditional label noise is common in decoded field space: `Cov(c_target|x,t)=C_Y/delta^2`. At a shared raw parameter state, conditional gradient-noise covariance is `G^T C_Y G/delta^4` for X and `G^T C_Y G/delta^2` for S/V. This is not a theorem about Adam convergence or a prediction of the winning FFN.

## Why the obvious uncapped alternative is not clean

Using a instead of delta preserves the original U labels, but native zero-X and its matched shifted-V control start with error `-Y/a`. With uniform t,

```text
E ||initial error||^2 = E||Y||^2 * integral_0^1 da/a^2 = infinity.
```

This is an infinite population initial loss, not just a possible numerical inconvenience. A finite batch hides the divergence. Restricting the time interval, changing its sampler/weight, or enforcing a special endpoint skip would add another intervention. Do not select this uncapped zero-head comparison.

With the .05 cap, the corresponding initial loss is exactly `39*E||Y||^2`. The zero direct-V arm instead starts at `(29/30)*E||U||^2`. Over half of the native X/S initial expected loss comes from the last 5% of uniformly sampled times. These statements follow by integrating `1/delta^2` and `a^2/delta^2`; they do not describe losses after learning. Uniform time is essential to this calculation, and this proposed test would not reproduce JiT's full setup.

The identity `m_theta=x+a*N_theta`, decoded by `(m_theta-x)/a`, remains exactly N_theta. Calling it a clean-data head introduces no model or control. Likewise, replacing velocity loss by unweighted clean-data MSE changes time weighting and is outside this comparison.

A different possible question would retain original U labels while using `(N-x)/delta`. Its raw target becomes `x+delta U`, which differs from Y in the capped band. That is an output-parameterization control under the old objective, but **not** faithful capped JiT clean-data prediction. Do not mix the two questions or add this fourth arm.

## The bounded-output issue is real, but does not select a repair

The toy ends with `W*LayerNorm(h)+b`. For hidden width w, affine LayerNorm parameters gamma,beta and positive epsilon, its normalized vector z satisfies `||z||<=sqrt(w)`. Consequently the raw output has the global bound

```text
||N_theta(x,t)|| <= sqrt(w)*||W diag(gamma)||_2 + ||W beta+b||.
```

This holds for every input at fixed finite parameters. It does not bound the input Jacobian. A Gaussian oracle `v*=k(t)x` is unbounded when k!=0, so the raw bounded head cannot equal it globally. There is no positive L2 approximation floor from this observation alone: the bound can grow with the parameters, and the far-tail probability can be tiny. Existing error concentration and attained amplitudes must be checked before attributing the observed gap to this class mismatch.

Importantly, the native X and shifted S heads impose the **same fixed asymptotic slope** `-I/delta`, with a bounded remainder. That is appropriate for a bounded clean-data target, whose posterior mean stays bounded. Our synthetic Gaussian mixtures have unbounded within-component support. For a scalar component variance q and standard Gaussian source, the component's true velocity slope is

```text
k_within(t)=(t*q-a)/(a^2+t^2*q).
```

For q=.04 at t=.9, it is about -1.509, while `-1/delta=-10`. This is an illustrative component calculation, not a fitted result. Therefore a native clean-data head need not repair the correct toy tail behavior and may impose a worse asymptotic slope than the bounded direct head. Neither boundedness nor the image-paper result alone justifies this training test.

## Smallest admissible ranking diagnostic, if existing evidence warrants it

The smallest informative screen is **two existing FFNs by three heads on Product16: six cells**. Use the unchanged vector-K1 parent and nominal128 SwiGLU, which has the relevant stronger same-task reference and comparable dense budget. Keep frozen FFN settings and common non-FFN tensors; do not add the original scalar arm, another width or a new operator to this first question. The vector parent is an existing extension of the professor's scalar-clock layer, not the identical original layer.

This screen can answer whether the relative parent/SwiGLU gap changes under X versus S versus V. Report each absolute error and the within-head ratio `E_ODE/E_Swi`; comparing those ratios across heads describes an interaction, not an ODE-specific theorem. A head change that helps both similarly supplies no architecture lead. A relative rank change while both degrade also does not establish useful progress.

The six cells could not nominate a useful architecture by themselves. Any promising interaction would still require the declared Ring8 transfer, both existing SwiGLU budgets, original generation-quality guards and separate practical cost checks. Gaussian unit calculations suffice for mathematical/preflight checks here; adding Gaussian training cells or the scalar family would enlarge the question without resolving the head confounds.

Before even this bounded screen is selected, the current read-only analysis should show why the output representation matters on the exposed input distribution. If the relevant errors are neither amplitude/tail-related nor sensitive to the known prediction-space burden, do not run it merely to complete a grid. It is a setup-compatibility question, not permission to rescue a stopped ODE family.

## Evaluation and endpoint interpretation

Training and primary regression evaluation would use c_target and its oracle `c*=(a/delta)v*`. Compare FFNs only under the same head and this same target. The existing five diagnostic times are at most .9, so delta=a there; their velocity/Jacobian comparisons remain in the original units.

True-velocity reconstruction is `v_reconstructed=(delta/a)c_theta` for t<1. It amplifies error in the capped band, is undefined at t=1 without an additional endpoint rule, and may have divergent uniform-time error for a model with nonvanishing residual there. Report any reconstruction only on an explicitly fixed interior domain or predeclared endpoint-band probes. Do not silently exclude the cap band from a purported full-domain metric.

Sampling the actual JiT-style decoded field c_theta is a separate practical measurement. Even its oracle c* does not generally transport along the original p_t path in the cap band:

```text
partial_t p_t + div(p_t c*) = (1-a/delta)*partial_t p_t.
```

Thus a capped-oracle sampling reference is needed to distinguish cap/solver effects from model error if such sampling is evaluated. Conversely, sampling the reconstructed uncapped field needs an explicitly justified endpoint and solver protocol; it must not acquire an ad hoc second clamp and then be called true-flow generation. No generation claim follows from head-regression loss alone.

## Gaussian-reference alternative: coherent algebra, still unselected

For source variance s0^2 and an isotropic Gaussian target reference variance q>0, define

```text
D=a^2*s0^2+t^2*q
L=(t*q-a*s0^2)/D
c_in=1/sqrt(D), c_skip=t*q/D, c_out=a*s0*sqrt(q)/sqrt(D).
```

The standard preconditioned denoiser `m=c_skip*x+c_out*N(c_in*x,t)` induces, without a denominator cap,

```text
v=L*x + [s0*sqrt(q)/sqrt(D)]*N(c_in*x,t).
```

For s0=q=1 this is the proposed `v=L*x+N(x/r,t)/r`, with r=sqrt(a^2+t^2). All displayed velocity coefficients remain finite, and N=0 gives the exact reference Gaussian velocity. Its residual scale squared equals the reference conditional velocity variance. It can preserve original U labels and uniform-time loss, unlike the capped clean-data comparison. These are standard Gaussian/EDM-style preconditioning facts, not a new ODE architecture.

It changes the initial field, known linear skip, output scale and input normalization together. A same-frame ODE/SwiGLU comparison could measure their relative ranking, but could not attribute a gain to any one of those ingredients. Separating them would require further controls; do not append them to the head grid.

An isotropic empirical covariance is not the within-mode covariance of a mixture. The unit-Gaussian L at t=.9 is about +.976, while the illustrative q=.04 component slope above is -1.509. A bounded residual cannot correct the wrong affine slope globally. The proposed reference therefore needs evidence that it removes a substantial part of the actual field burden, rather than a covariance-matching intuition. No oracle covariance, fitted nonlinear teacher or extra parameter sweep is justified here. Both this alternative and the six-cell head screen remain unselected pending that evidence.
