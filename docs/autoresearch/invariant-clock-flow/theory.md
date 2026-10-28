# Exact invariant-clock feature flow

Zhizhe Zhang · October 28, 2026

## Target and status

**COHERENT AS STATED** for the hidden autonomous field below; numerical and
architecture checks are pending. The construction is classical action-angle
prior art, reviewed [here](literature-review.md). The question is whether its
constraint helps FM, not whether its ODE can be solved accurately.

## Invariant object, assumptions and notation

Let hidden width d=2m, and split h into pairs h_j in R². Put
`I_j(h)=||h_j||²/2`, `J=[[0,-1],[1,0]]`. Finite learned omega_j define a
skew pair matrix P=diag_j(omega_j J). A C¹ clock network maps all actions to
`s(I)=.2+.8 sigmoid(MLP(I)/10)`. The rate vector is `theta_j(I)=omega_j s_j(I)`.
No frequency sign, orthogonal basis, clipping penalty or posterior meaning
is assumed. Inner feature time u is distinct from external FM time t.

The object is the actual autonomous field

```text
dh_j/du = theta_j(I(h)) J h_j.
```

Every displayed equality below concerns this feature field, not the complete
FFN, backbone, clean denoiser or sampling ODE. The surrounding time condition
is fixed during an inner solve.

## Derivation map
1. Prove pair actions are conserved along this actual field.
2. Freeze only those conserved inputs and integrate each planar rotation.
3. Differentiate the complete input-conditioned endpoint, retaining gate feedback.
4. Compare ordinary same-budget controls and state what the constraint cannot prove.

## 1. Conserved clocks and exact solution — identities

Since `h_j^T J h_j=0`, `dI_j/du=0` for every pair. Therefore the whole rate
vector theta(I(h)) stays constant along its own trajectory. With R(a) the
ordinary 2D rotation,

```text
Phi_u(h)_j = R(u theta_j(I(h))) h_j.
Phi_0 = identity;  Phi_u(Phi_v(h)) = Phi_(u+v)(h).
Phi_u^-1 = Phi_(-u).
```

These statements hold at zero pairs without polar angles or division by
radius. Smooth bounded clocks and finite omega give a locally Lipschitz
field of at most linear growth; the explicit norm-preserving solution exists
for every real u. This is not freezing an arbitrary original input clock
whose value could change along the claimed state-dependent field.

## 2. Complete endpoint Jacobian — identity

At a fixed h, let B=dtheta/dI, U the d-by-m block matrix with column j equal
to J h_j in pair j, and V the m-by-d block matrix with row j equal to h_j^T.
Let R be block diagonal with rotations R(u theta_j). For any variation v,

```text
D Phi_u(h)[v]_j = R_j v_j + u R_j J h_j sum_k B_jk (h_k^T v_k)
D Phi_u = R (I_d + u N),  N=U B V.
V U = 0;  N²=0;  det(D Phi_u)=1.
```

The determinant follows from Sylvester's determinant identity and VU=0,
while detR=1. In particular volume is preserved. B is the complete learned
clock derivative, including omega; P's imaginary eigenvalues alone omit it.

For nonzero pair radii r_j, U and V have orthogonal normalized column/row
spaces. The nonzero singular values of N equal those of diag(r) B diag(r).
This remains true after omitting zero-radius rows/columns. In orthonormal
left/right singular coordinates, I+uN consists of scalar shear blocks and
identity directions. Thus, with sigma=||N||_2,

```text
||D Phi_u||_2 = (sqrt(u² sigma²+4)+|u| sigma)/2.
```

This gives at most linear growth with inner time at fixed h; it does not
give gain<=1. It is a local derivative identity, not global sampler stability.

## 3. A bounded-clock noncontractive witness

For one pair take omega=1 and
`theta(I)=.2+.8 sigmoid(5(I-.5))`. At h=(1,0), theta=.6 and B=1.
The derivative has the singular values of `[[1,0],[1,1]]`, with largest
value `(1+sqrt(5))/2=1.618...`, although ||Phi(h)||=||h|| and detD=1.
This clock is exactly realizable using SiLU(z)-SiLU(-z)=z in the specified
MLP. Normalized features or purely imaginary P eigenvalues therefore do
not certify nonexpansiveness. The full learned feedback must be measured.

For several pairs the map is not generally symplectic. In positive-action
coordinates it updates angles by u theta(I) while actions stay fixed;
symplecticity requires B symmetric, equivalently theta locally a gradient.
The planned vector MLP does not enforce this, so call it volume-preserving
feature transport, not a Hamiltonian network.

## 4. Ordinary implementation and matched controls

The production endpoint at u=1 uses only pairwise sin/cos and the action-clock
MLP. This is an ordinary gated rotation, with the same parameters as its ODE
description. No repeated solve, matrix exponential or polar coordinate is used.
Use two such layers with the professor's middle SiLU.

For the same a_j=theta_j(I), two controls have identical parameters/MLP work:

```text
Euler(h)_j = h_j + a_j J h_j
NormalizedEuler(h)_j = (h_j+a_j J h_j)/sqrt(1+a_j²)
                    = R(arctan(a_j)) h_j.
```

Euler changes each pair amplitude by sqrt(1+a_j²). NormalizedEuler preserves
the same actions. At unit duration it is itself the exact flow of a modified
invariant-rate field with rate arctan(theta); its natural variable-step formula
is not the original rotation semigroup. Hence a tie with this control can
support the conserving family, not an exclusive exact-integrator mechanism.

All three start at omega=0, so the two-layer FFN initially equals SiLU(h).
Their inherited clocks and initial parameter tangents are paired. Clock
parameters initially have zero gradient because omega=0; a separate nonzero-
frequency probe and the first updates must verify active routes. Ordinary
W2 SiLU(W1h) with W1=W2=I and fixed0.6P with P=0 match the initial function,
not the constrained model's complete trainable tangent space.

## 5. Why per-pair actions, and what remains a hypothesis

A clock depending only on total norm can become constant on exactly
normalized input. Different pair energies can vary at the same total norm.
The MLP may couple those energies, retaining this variation without a learned
frame. This is a structural witness, not a proof of complete-network capacity
or superiority over SwiGLU. The old Product/sign-code laws share radial
statistics while differing in dependencies; their existing banks are relevant
controls, not a new dataset selected to favor the method.

At d=64, choose clock hidden width128 arithmetically: two clocks cost
16384 dense MACs per token, near the original K1's16512. The clocks/frequencies
store16768 scalars per FFN. These are tentative analytic counts to verify in
the actual native constructor; pointwise trig/radius work and wall time are
separate. A rounded SwiGLU width must be selected by the native factory/count,
not by quality. No budget, step, initializer or clock-range sweep is selected.

## Boundaries, approximation and non-claims

- The feature law is not the outer FM field or a clean posterior evolution.
  Amplitude preservation may discard useful computation and hurt denoising.
- Middle SiLU, residual updates and the learned backbone need not preserve
  norm, volume, invertibility or symplecticity.
- No Taylor-accuracy, state-frame, clock-constant or stopped-setting rescue.
  This is one newly constrained field with an acknowledged ordinary realization.
- Exact validity and controlled amplitude do not predict quality; strong
  ordinary and same-budget controls must decide continuation.

## Open risks
Restrictive fixed pairs, initially dormant clocks, nonnormal gate feedback,
pointwise overhead and unequal parameter tangents need explicit tests and
reporting. The protocol must be fixed before cold outcomes and preserve all
old datasets, metrics, sources, matrices and submitted Report1.
