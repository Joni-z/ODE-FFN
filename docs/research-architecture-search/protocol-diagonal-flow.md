# Controlled expanded-feature diagonal-flow study

Registered after the audited Stage17 failure and the plain-SiLU reference,
before any diagonal-family training. The old shared-P/width axis is stopped.
This is a different architectural family, not a one-factor explanation of
the old dense ODEFFN. It reallocates internal dense-operator/controller work
to wider projected features and cheap coordinatewise transformations.

## Operator and lineage

```text
z = E h + bE
F(z;lambda) = z * R(lambda*s(z))
s(z) = .2+.8*sigmoid(z)                 adaptive, fixed
s(z) = .6                              static control
lambda = tanh(eta), eta initially0      one parameter/channel/solve

FFN(h) = R_out F2(SiLU(F1(z))) + b_out
R_K1(y)=1+y; R_K2(y)=1+y+y²/2; R_exp(y)=exp(y).
```

Each internal map retains the professor's conditional linear-flow polynomial
with P diagonal. Its clock is chosen from the solve input and held fixed.
The exponential solves that conditional linear system exactly; it is not the
nonlinear ODE obtained by continuously recomputing s along its trajectory.
Internal solve time is distinct from FM time. The fixed local controller
can be written as t_func(z)=10z with the old tau10, but the old learned dense
controller is removed. There is no learned clock slope/bias/temperature.

E/R_out supply all cross-channel mixing in this FFN. The latent expansion,
diagonal generator and local fixed clock change several choices relative to
the previous dense ODE; no causal single-factor claim is made across families.
Within this new family, the following controls isolate the activation changes.
The exact block is also an MLP with learned coordinatewise activation. ODE
activations have prior art; see [the primary-source review](literature-diagonal-flow.md).

## Seven arms and exact internal controls

All arms use width64/depth3. All diagonal and ordinary SiLU arms use hidden128.

| Label | Model | Scalar maps |
|---|---|---|
| ffn_silu | ordinary SiLU FFN | no maps |
| diagonal_static | diagonal ODE FFN | static-clock K2 |
| diagonal_k1 | diagonal ODE FFN | adaptive K1 |
| diagonal_k2 | diagonal ODE FFN | adaptive K2 |
| diagonal_exp | diagonal ODE FFN | adaptive exponential |
| swiglu_budget | SwiGLU | nominal96, actual64 hidden |
| swiglu | SwiGLU | nominal128, actual85 hidden |

All eigenvalue coordinates start at zero. The four diagonal variants have
the same parameter counts as one another and the same actual E/R_out
initialization as ordinary SiLU. They add two hidden-sized eigenvalue vectors
relative to SiLU. The extra coordinates
consume no RNG draws. Complete initial functions equal the SiLU model for
arbitrary inputs and a common nonzero output head, not just the formal zero
head. Adaptive K1/K2/exp additionally have identical complete initial parameter
Jacobians and kernels. Subsequent trajectories can differ.

Static K2 is exactly absorbable into the two affine projections, including
their biases. Its function class therefore equals ordinary same-width SiLU;
it is an active parameterization control. K1 is an affine identity–SiLU
mixture, function-equivalent to an ACON-type gate with a different regular
identity-limit parameterization. It is the necessary generic activation
control. A K1 gain alone does not establish a benefit from higher terms or
exact exponentiation.

Per FFN, ordinary SiLU uses16576 parameters/16384 dense MACs. Diagonal variants
use16832 parameters/the same16384 dense MACs. Over three blocks the added
parameters are768. SwiGLU128 uses16554/16320; SwiGLU96 uses12480/12288. Pointwise
sigmoid, polynomial and exponential work is additional. Record wall time;
equal dense MACs are not a speed claim.

## Mathematical predictions and limits

At |lambda|<=1, each adaptive exponential scalar map has derivative at least
.8/e. Each K2 map has derivative at least.1. These conservative analytic
bounds include the input-clock derivative. They establish scalar monotonicity,
not invertibility/stability of the middle SiLU, projected FFN, complete velocity
network or generated FM trajectory. K1 can fold at sufficiently negative
lambda; this is an explicit difference from K2/exp, not a hidden restriction
making all arms equally monotone.

For a frozen clock, K2's diagonal multiplier lies in[.5,2.5], hence its
conditional linear-map condition number is at most5. The exponential has
condition number at most exp(2). Including clock derivatives gives the looser
single-map Jacobian bounds29 for K2 and1.5*exp(2) for exp. These bounds do not
require P's singular values to be uniform or bounded away from zero. They
still do not cover the middle SiLU, affine projections or velocity network.

The static-control equivalence, initial adaptive kernel equality and conditional
flow interpretation must pass direct algebra/autograd checks before launch.
More terms alone do not imply better field fitting. The adaptive eigenvalue
directions include cheap nonlinear features absent from a single SiLU neuron's
parameter tangent, but all three adaptive solvers share those directions at
initialization. This is neither a wide-network lower bound nor a reason to
expect K2/exp to outperform K1. Symmetry/activation-shape observations are
restricted motivation; they do not prove an FM-specific architectural advantage.

For example, a single K2/exp map's odd-part gain increases from center to
tails, whereas the local Euler map of the balanced bimodal teacher can require
the opposite ordering. Dense projections and the full composition can change
that comparison. This counterexample blocks treating central symmetry alone
as proof of target alignment.

## Training and initialization

Twenty-one cells: all seven arms on Gaussian2D, Ring8 and rotated product16.
Training seed0,5000updates,batch512,AdamW3e-4,weight decay0,gradient clip1,
original cached-bank pair labels and existing dataset versions remain fixed.
Use original validation streams, complete five-time field/Jacobian diagnostics,
validation SW2 and Heun32. Both SwiGLU controls and ordinary SiLU are rerun in
this study; their prior runs remain visible for trajectory checks, not counted
as independent repeated evidence.

Copy every non-FFN initial tensor and actual runtime buffer from the same
hash-bound Stage15 vector_mul step0 references. Verify the diagonal/SiLU
projection tensors, biases and RNG draws match exactly. No pretrained weights,
optimizer state or labels are copied. No stopped additive, clock-coordinate,
identity, sharing, spectral-cap or extra loss intervention is combined.

Run on Mac CPU with two one-thread workers and actual10s telemetry. No GPU
is reserved. Freeze a new source/protocol/plan runtime; never edit old runtimes.

## Predeclared decision

Only K2 and exp are higher-order candidates. Nominate one only if, on at least
one nonlinear task, it lowers primary validation oracle MSE by at least10%
against both SwiGLU sizes and by at least5% against adaptive K1, while also
improving ordinary SiLU and static K2 there. On every other primary task it
must be at most5% worse than the better SwiGLU. On both nonlinear tasks its
complete Jacobian error and validation SW2 must each be within10% of the
better SwiGLU. These are engineering screens, not statistical tests.

Report every arm and every endpoint/curve. If both candidates pass, order by
product16 oracle MSE, then measured dense MACs, then Ring8 MSE. All pointwise
costs remain visible; a nomination still needs isolated timing, matched-budget
confirmation, a fresh independent holdout and mechanism verification before
the architecture goal can be achieved. Existing validation or diagnostic
calibration paths are not that holdout.

If K1 alone performs well, record a generic low-cost activation lead and the
prior-art boundary; do not call it evidence for a higher-order ODE mechanism.
If neither higher-order candidate passes, stop these settings without a beta,
clock slope, temperature, width or extra-order rescue. Do not add rational,
learned-clock or additional spectral tricks after inspecting the result.

## Required verification

Before launch: conditional matrix-exponential identity; full clock-dependent
input derivatives and sufficient bounds; explicit K1 folding example; zero-
eigenvalue SiLU functions/RNG/projection pairing; adaptive initial full-parameter
Jacobian equality; exact static absorption including biases; real parameter
and linear-operation counts; full-network integration and diagonal P logging;
unchanged default trajectories against the frozen predecessor. Complete a
trainer/sampling/checkpoint/resume check and actual monitor heartbeat.

After completion: all source/data/config/reference hashes, every checkpoint,
initial eigenvalues/projection pairing, all snapshot data-generator states,
and final common-point fields/Jacobians. Preserve P=diag(lambda), its raw eta
coordinates and conditional diag(s)P as distinct quantities. Final-field
replay is separate from fresh holdout or sampling-trajectory reproduction.
