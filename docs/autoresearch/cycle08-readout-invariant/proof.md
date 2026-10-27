# A fixed readout imposes an affine law on part of the sampled state

## Assumptions

Consider one frozen native predictor with ambient dimension D=128 and final normalized hidden width w=64. All weights are finite, LayerNorm epsilon is positive, and its raw mapping is locally Lipschitz in x, continuously dependent on external time. These conditions hold for the finite affine/SiLU/sigmoid/native-normalization networks in this cohort. The head and LayerNorm affine parameters are shared across external time. The sampler starts from a fresh standard Gaussian independent of the frozen model. The cap is delta(t)=max(1-t,.05).

For the normal-space bound the target is the registered population `Y=QH+sqrt(q)*NG`, with Q orthonormal128-by2, N=I-QQ^T, independent standard Gaussian G and q=0 or.04. H has the finite-second-moment Ring8 law. The bound is not asserted for arbitrary image data or empirical target measures.

## 1. The raw velocity lies in a fixed affine subspace

Let z be the centered, variance-normalized hidden vector before learned LayerNorm scaling. Its coordinates sum to zero and `||z||²=w*var(h)/(var(h)+epsilon)<=w`. Let C=I-11^T/w. Then Cz=z and

```text
raw(x,t)=W*(gamma*z+beta)+b=B*z+c0
B=W diag(gamma) C,      c0=W beta+b.
```

C has rank w-1, so rank(B)<=63. If L has orthonormal columns spanning ker(B^T), then `L^T raw=L^T c0` for every state/time. This does not freeze gates or differentiate a conditional generator; it holds for arbitrary hidden computation producing z.

The raw output norm is bounded by `sqrt(w)||B||_2+||c0||`. Direct decoding is therefore bounded. The two skip decoders have at most linear growth because delta>=.05. Local state Lipschitzness and this uniform growth bound give a unique global solution on the finite interval by continuation and Gronwall. Its Gaussian-source projections have finite second moments, so the covariance and W2 calculations below are well-defined.

## 2. Projected dynamics are affine, although the full field is nonlinear

Write y=L^T x, b_L=L^T c0. Direct decoding gives `y'=b_L`, hence `y(1)=y(0)+b_L`. The skip decoders give

```text
y'=-y/delta(t)+alpha(t)*b_L
alpha_clean=1/delta,    alpha_shifted=1.
```

Variation of constants yields `y(1)=R*y(0)+d*b_L`. The common homogeneous factor is `R=exp(-integral_0^1 dt/delta)=.05/e`. Clean has d=1-R. Shifted has `d=.05*(1+(log(20)-1)/e)`. Standard-Gaussian initial covariance becomes R²I in these coordinates; the forcing affects only the mean. Direct instead has R=d=1.

For Heun, projection commutes with both linear combinations of stage values. With step h and `a_i=-1/delta(t_i)`, `f_i=alpha(t_i)b_L`, projecting the predictor and corrector gives exactly

```text
y_predict=(1+h*a_i)*y_i+h*f_i
y_(i+1)=[1+h*(a_i+a_(i+1)+h*a_i*a_(i+1))/2]*y_i
        +h*[(1+h*a_(i+1))*f_i+f_(i+1)]/2.
```

Thus the scalar recurrence in the registered code holds even when hidden activations change nonlinearly between the two stages. Its32-step R,d need not equal their continuous-time values. No fitted coefficient or solver approximation is used in this identity.

## 3. The target-normal intersection gives a population W2 obstruction

Let L_N be an orthonormal basis of `ker(B^T) intersect ker(Q^T)`, of dimension k_N. The rank-nullity inequality gives `k_N>=128-rank(B)-2>=63`. Generated coordinates there are Gaussian with mean `mu=d*L_N^T c0` and covariance R²I, while target coordinates are Gaussian with mean zero and covariance qI.

For any coupling of these marginals, the expected squared distance is the squared mean distance plus `k_N*(R²+q)` minus twice the cross-covariance trace. Coordinatewise Cauchy-Schwarz bounds that trace by `k_N*abs(R)*sqrt(q)`. The bound is attained by sharing a standard Gaussian between the two isotropic marginals. Their exact squared W2 is therefore

```text
||mu||²+k_N*(abs(R)-sqrt(q))².
```

An orthogonal projection cannot increase squared Euclidean transport cost for any coupling, so this is a lower bound on full ambient W2². For the numerical sampler use its exact discrete R,d. For q=0 the target is a point mass in this projected space and the same argument applies.

Ignoring the nonnegative mean term gives a universal lower bound63*(abs(R)-sqrt(q))² for this architecture/head and population. Retraining may change B's orientation, but not the dimension argument. Altering hidden P alone cannot remove it. This statement concerns the128D cohort; the older2D/16D readouts may have full row rank and no such null space.

## Limits

The theorem is an affine-readout restriction shared by the ordinary baselines. It is not a new ODE architecture, a complete error attribution, or a result about all FM data. Full W2, finite-direction empirical SW2 and finite-sample energy are distinct quantities. Finite precision only gives approximate projected identities; the registered numerical gates test that approximation on the actual saved trajectories. No covariance target, basis or null-space information is passed to training.
