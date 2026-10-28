# Derivation package: constant-clock training coordinates

Zhizhe Zhang · October 28, 2026

## Target

Derive what is held fixed when a constant-clock professor block is compared
with an ordinary SiLU block. Specify how changing P to an effective W changes
Adam and joint clipping. These identities justify controls, not a quality claim.

## Status

COHERENT AS STATED for the real-valued parameter maps and the stated optimizer
conditions. Actual float32 forwards/resumes are separately verified. The image
advantage and any training benefit remain empirical questions.

## Invariant object

The model function represented by the shared backbone theta and effective
FFN matrices W1,...,W6, together with its update under a named optimizer. Raw P
spectra are coordinates; they are not the invariant function or sample metric.

## Assumptions and notation

- Each local constant c is fixed and positive; formal controls use float32(.6).
- The local map is first-order `h+cPh`, with no bias or internal normalization.
  Each complete core is two such maps around SiLU; dropout is inactive/zero.
- L is the same differentiable minibatch loss. H is the shared-backbone
  gradient, G_j the gradient with respect to effective W_j.
- Adam starts with zero moments, common betas, bias correction, epsilon>0,
  and zero weight decay. eta is the native learning rate.
- A common clipping factor kappa is computed in P coordinates when deriving
  an equivalent W update. The actual ordinary control instead uses its native
  W-coordinate global clipping, deliberately preserving that difference.
- All identities below are real-arithmetic identities. Native finite-precision
  initial W values are rounded once and their errors are recorded.

## Strategy and map

1. Rewrite the complete function using a bijective affine parameter map.
2. Pull back minibatch gradients and identify the joint clipping norm.
3. Transform bias-corrected Adam moments and its update to W coordinates.
4. Separate matched initialization from matched optimization and image quality.

## Main derivation

### 1. Function class and initialization — identities

Set `W_j=I+c_jP_j`. Then

```text
F(h) = (I+c2P2) SiLU((I+c1P1)h) = W2 SiLU(W1h).
P_j = (W_j-I)/c_j.
```

Every square W has exactly one P when c>0. Thus this constant-clock class is
the ordinary bias-free SiLU class of the same width. It does not establish
equivalence to an adaptive clock or a common training trajectory. Mapping
native P_initial to W_initial gives a common initialized function; ordinary
default zero-centered linear initialization is a different initial condition.

### 2. Gradient and clipping geometry — identities

For a perturbation dP, `dW=c dP`, so

```text
grad_P L = c G_W;  grad_theta L = H.
N_P^2 = ||H||^2 + sum_j c_j^2 ||G_j||_F^2.
N_W^2 = ||H||^2 + sum_j ||G_j||_F^2.
```

The native clip kernel applies its scalar function to N_P or N_W, respectively.
Its numerical epsilon is retained in the fixture. A common clip limit does
not give the same clipping factor in the two coordinates. With unclipped
plain gradient descent, effective W steps scale as eta*c^2 rather than eta.

### 3. Adam in effective coordinates — conditional identity

For one fixed c>0 and the same clipped W-gradient sequence, linearity of the
first moment and homogeneity of the second moment give

```text
m_P_hat = c m_W_hat;  v_P_hat = c^2 v_W_hat.
W_next = W - eta*c * m_W_hat/(sqrt(v_W_hat)+epsilon/c).
```

The square, square root and division act elementwise. Bias correction preserves
the relations. With the same P-coordinate clipping factor applied to both
representations, equivalent direct-W Adam uses rate eta*c and epsilon/c for
the six W matrices, native settings for the shared backbone. This equivalence
requires the same clipping factor and corresponding initial optimizer state.

The formal ordinary control uses the unmodified native rate/epsilon and its
own coordinate norm, so its trajectory is not claimed equivalent. Those
optimizer-coordinate differences are the comparison, not a new function class.
No scaled-rate or transformed-clipping training arm is selected; the equivalent
update is only a small data-free verification witness.

## Interpretation and non-claims

An ordinary matched-init control separates the constant representation from
default initialization. A same-c frozen native checkpoint compares parameters
learned under adaptive clocks with parameters trained under a constant clock.
It changes the whole learned model, not exclusively P. Smaller raw P norms,
equal singular values or common initialized outputs do not order FID.

## Open risks

Roundoff, finite optimization and joint clipping remain in the actual runs.
Neither these identities nor a frozen inference comparison proves that
adaptive clocks are useful during training, that a constant model will match
the native endpoint, or that an ODE-specific successor has been established.
The [registered experiment](protocol.md) tests only those finite image controls.
