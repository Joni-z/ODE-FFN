# Exact discrete defect transport, without dropping signed interactions

Fix32 learned Heun maps M_i and oracle Heun maps O_i with the same time grid, cap and starting noise. The state-dependent learned clocks/frames are included in M_i. Let x_(i+1)=M_i(x_i), define S_32=identity and S_i=O_31 composed with ... composed with O_i, and put p_i=S_i(x_i).

Then p_0 is the oracle endpoint and p_32 the learned endpoint. Adding adjacent differences gives the exact discrete identity

```text
E = x_32 - oracle_endpoint = sum_i Delta_i
Delta_i = p_(i+1)-p_i
        = S_(i+1)(M_i(x_i)) - S_(i+1)(O_i(x_i)).
```

This is a nonlinear finite-difference transport of the local step defect `d_i=M_i(x_i)-O_i(x_i)` through a fixed suffix. No Jacobian linearization, frozen hidden generator or continuous-solver approximation is required. It holds for arbitrary defined discrete maps; finite-precision branches also telescope by their recorded values. Norm ratios of Delta_i and d_i describe those finite displacements, not worst-case amplification.

For each source and any orthogonal tangent/normal projection R,

```text
||R E||² = sum_i <R E, R Delta_i>.
```

Average over the fixed source bank and divide by the subspace dimension. Terms can be negative and their sum is the paired endpoint MSE. Summing ||Delta_i||² instead would discard cross interactions. Grouping the signed terms into the four prespecified bins preserves the identity. Full-space energy is the dimension-weighted sum of tangent and normal energies.

Paired endpoint MSE depends on the chosen common-source coupling. Different transports can produce similar distributions but assign the same source to different modes. Therefore it is neither the benchmark's empirical SW2 nor a lower bound on it. Nearest-centroid disagreement makes this limitation visible without becoming a new selection metric. The oracle endpoint is the existing capped Heun reference, which itself differs from the exact target law.

## Independent normal-space prediction

For N=I-QQ^T, the registered mixture oracle has normal field beta(t) N x, where

```text
a=1-t; delta=max(a,.05)
gain(t)=t*q/(a²+t²*q), using the native zero extension when the denominator is zero
beta(t)=(gain(t)-1)/delta.
```

The scalar factor of one Heun step of size h=1/32 is

```text
r_i=1+h/2*(beta_i+beta_(i+1)+h*beta_i*beta_(i+1)).
G_i=product_(j=i..31) r_j;   G_32=1.
```

Oracle tangent and normal dynamics are separable. In real arithmetic, `N Delta_i=G_(i+1) N d_i` exactly, independent of the nonlinear tangent suffix. This also gives `N E=sum_i G_(i+1) N d_i`. Native float32 rounding introduces small deviations checked by the registered tolerance. The scalar calculation uses the actual grid and declared endpoint extension, not a continuous-time substitute.

No derivative cross-term orthogonality or contraction assumption is used. A high on-path field defect can reflect approximation and visited-state differences jointly; this diagnostic does not identify an optimizer cause. None of the prefix/suffix branches is proposed as an inference method or selected by quality.

## A forced component already identified by the output layer

For the frozen raw output, write B=W diag(gamma) C, c0=W beta+b, C=I-11^T/64. Let T be the orthogonal projector onto ker(B^T) intersect ker(Q^T), as in cycle08. The clean-head projected model field is `(c0*T-x*T)/delta`, independent of hidden P/frame computation. Let R_i be the product of clean-head homogeneous Heun factors over steps0throughi-1, with R_0=1. Its particular solution is constant c0, so

```text
model_path_i * T = R_i * source*T + (1-R_i)*c0*T
bridge_i * T = G_i * [R_i*source*T+(1-R_i)*c0*T].
```

This predicts the complete suffix-bridge sequence in a subspace of dimension at least63. It remains a fixed-readout restriction when changing hidden P; retraining the whole network changes its orientation/bias, not the dimension argument. Under native floating-point execution the registered tolerance checks the approximation.

Because T is contained in the normal projector N, normal paired endpoint energy splits orthogonally into T and N-T components. Signed temporal contributions split the same way. The finite-bank forced-energy fraction is therefore a concrete diagnostic of a shared structural component, not a causal attribution of every difference between independently trained models. It does not turn paired MSE into SW2 or reopen a stopped head/readout setting.
