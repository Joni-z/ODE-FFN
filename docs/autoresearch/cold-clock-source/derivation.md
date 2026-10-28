# A time-conditioned mixing family, with a limited claim

Let h be one FFN input and t the external corruption time. P1 and P2 are
learned, time-independent matrices. c(t) is the existing JiT conditioner;
it has no image dependence under the current fixed-label protocol. Each
scalar sj(t) is the original bounded clock MLP applied to c(t).

```text
Wj(t) = I + sj(t) Pj
u = W1(t) h
F(h,t) = W2(t) SiLU(u)
```

This is exactly an ordinary conditional matrix-interpolation FFN. The ODE
name does not add another operation. What changes relative to a constant
core is the learned family across objective time; what changes relative to
an input clock is the removal of its feature-feedback derivative.

For fixed t, the full input derivative is

```text
J_h F = W2(t) diag(SiLU'(u)) W1(t).
```

For fixed h, its explicit time derivative is

```text
partial_t F = s2'(t) P2 SiLU(u)
            + W2(t) diag(SiLU'(u)) s1'(t) P1 h.
```

Both derivatives of the shared conditioner and both clock MLPs must be kept.
Along a changing feature path, the total derivative also contains J_hF*h'.
An input clock instead has local derivative I+s(h)P+(Ph)grad(s(h))^T.
This distinction is exact; whether removing that feedback helps is empirical.

The family is not merely a diagonal rescaling of one fixed mixing matrix.
For example, P=[[0, 1],[1, 0]] gives W(s)=[[1, s],[s, 1]]. The cross ratio
W1 1* W2 2/( W1 2* W2 1)=1/s² changes with s. Left/right nonzero diagonal rescaling
of a fixed 2x2 matrix preserves that cross ratio. Thus such pointwise modulation
cannot represent this entire family at one matrix. This is not a separation
theorem for a full multi-layer ordinary transformer; existing AdaLN and all
strong ordinary controls remain part of the comparison.

No Gaussian observation law is assigned to normalized hidden features. No
posterior-optimality, full-model PSD, stability or FID guarantee follows.
The commuting identity [W(s), W(r)]=0 within one local layer describes shared
mixing directions, not a semigroup law for the whole nonlinear predictor.
The constant-clock optimizer transport also cannot simply be applied to a
trainable, time-dependent sj across every batch.

At zero final clock weights, all sj are the same 0.6 and both cold arms have
the identical full parameter state. That pairing removes a default-versus-
matched initialization contrast. It does not make their parameter tangents
or subsequent gradient routes equal. The registered experiment, rather than
this algebra, decides whether the time-only restriction is useful.
