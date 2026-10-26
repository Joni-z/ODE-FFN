# Forced-response correction of the SwiGLU gate

Registered after the audited Stage19 rejection, before any forced-gate training.
The mathematical and experiment reviews permit one fixed operator comparison;
the literature review supplies interpretation and novelty limits. Existing
evidence supports preserving SwiGLU's independent gate/value features. It does
not predict that this particular response will improve them.

## Operator and relation to the original layer

```text
a, v = unchanged independent SwiGLU projections of h
b = SiLU(a); s = sigmoid(a); lambda = tanh(eta), eta initially0
dy/du = (lambda*s)*y + b, y(0)=0, u in[0,1]
gate_exact = b*phi1(lambda*s)
phi1(c) = expm1(c)/c, phi1(0)=1
FFN(h) = unchanged_output_projection(gate*v).
```

The input selects coefficients which remain frozen during this auxiliary
solve. The professor's homogeneous Taylor construction applied to augmented
matrix `[[lambda*s,b],[0,0]]` and state `(0,1)` returns b at K1, and
`b*(1+lambda*s/2)` at K2. K1 is ordinary SwiGLU. The exact response retains
all terms. This extends the original conditional-flow construction to an
affine forced system; it changes the old two-solve dense FFN substantially.
The auxiliary solve time is distinct from FM time. No external solver accuracy,
full-network stability or FM-specific approximation theorem is claimed.

One eta per gate channel is the only added trainable object. There is no new
projection, controller, clock slope, temperature, loss or rate-bound setting.
The new arms use nominal hidden128, actual85, with255 extra parameters across
three blocks and exactly the baseline dense MACs. Pointwise cost is additional.

## Five arms

| Label | Gate | Nominal hidden |
|---|---|---:|
| swiglu_budget | ordinary SiLU |96|
| swiglu | ordinary SiLU |128|
| forced_static | b*phi1(.5*lambda) |128|
| forced_k2 | b*(1+.5*lambda*sigmoid(a)) |128|
| forced_exact | b*phi1(lambda*sigmoid(a)) |128|

Static gains absorb exactly into columns of the output projection. They retain
the ordinary SwiGLU function class while adding active redundant parameters.
Adaptive K2 is the generic multiplicative-gate control. K2 and exact have the
same complete initial parameter Jacobian, including their eta directions;
both preserve every ordinary SwiGLU parameter and its initial derivative.
The new eta tangent is `.5*b*sigmoid(a)*v` before output projection. Modulo
ordinary output gains and gate-bias tangents, this is a generic GLU feature.
Thus K2 success alone cannot establish a uniquely ODE or all-orders mechanism.

Exact relative gate gain is in `[1-exp(-1),exp(1)-1]`; K2 gain is in `[.5,1.5]`.
They have different response curves and ranges, both positive. This comparison
tests those response terms, not a pure numerical solver-order intervention.
Bounded conditional rates do not bound the augmented generator norm because
the forcing makes a shear, nor the whole SwiGLU product/Jacobian.

## Fixed training and initialization

Fifteen cells: all five arms on Gaussian2D, Ring8 and product16. Use training
seed0,5000updates,batch512,AdamW3e-4,weight decay0,gradient clip1, original
cached-bank pair supervision and existing data versions. Keep width64/depth3,
the five diagnostic times, validation field/Jacobian measures and Heun32 SW2.
Do not use exact-mean supervision or a changed optimizer.

Construct ordinary SwiGLU projections before allocating zero eta vectors,
without extra random draws. Pair every shared initial tensor, buffer and RNG
with nominal128 SwiGLU, using a nonzero probe output head as well as formal
zero-head initialization. Use the same hash-bound Stage15 step0 references
for non-FFN initialization in every arm. These contain no trained weights.
Both baseline reruns must reproduce their historical trajectories; they are
compatibility checks, not independent replications.

Freeze new source/config/protocol, use two one-thread Mac CPU workers and
actual10s telemetry, and prevent idle sleep. Reserve no GPU. Preserve all old
runtimes and the submitted Report1. Stop failed settings without width, slope,
rate-bound, temperature, coupling or additional-order rescue.

## Decisions fixed before outcomes

An adaptive arm is a lead only if, on at least one nonlinear task, its primary
validation oracle MSE is at least10% lower than both SwiGLUs and at least5%
lower than static. On every other primary task it must be within5% of the
better SwiGLU. On both nonlinear tasks, its full velocity Jacobian error and
validation SW2 must each be within10% of the better SwiGLU. These engineering
thresholds do not establish statistical significance.

K2 passing establishes at most a generic adaptive-gate lead. Exact passing
establishes at most the same unless it also improves K2 primary MSE by at
least5% on its qualifying nonlinear task, with the same other-task guards.
A static-only gain supports redundant reparameterization. If both adaptive
arms fail, stop the family as specified. Report every fixed endpoint and curve.

Any nominated lead still needs a fresh independent holdout, nonlinear transfer,
isolated CPU/MPS timing where supported, and a budget guard against a slightly
wider ordinary SwiGLU (nominal129,actual86, which has more parameters/MACs than
the new nominal128 model). Existing exposed validation paths do not count as
that confirmation. If both adaptive arms pass, prefer lower measured practical
cost at comparable quality; do not credit indistinguishable higher terms.

## Required verification

Before formal launch: safe phi1 value/first/second derivative at zero and
near the series boundary in float32/float64; augmented matrix-exponential and
clock-dependent derivative identities; static output-column absorption;
complete initial functions and common-parameter gradients; adaptive full
initial Jacobian equality; actual linear MACs/parameter counts; CLI/config
propagation; nonzero trained eta; full trainer/checkpoint/resume/sampling and
real monitor heartbeat. Default predecessor trajectories must remain exact.

After completion: native source/data/config/reference hashes and all checkpoint
states; all-arm snapshot data RNG pairing; exact native final field/Jacobian
replays; complete curves and every recorded eta/lambda channel. Log latent
P, conditional rates, forcing, relative response, gate derivatives and input
projection trajectories separately from the full velocity Jacobian. Final
endpoint replay alone does not verify a fresh holdout or sampling trajectory.
