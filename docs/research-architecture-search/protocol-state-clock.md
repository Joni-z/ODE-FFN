# Original vector field: frozen versus state-dependent clocks

Registered before any state-clock training. The head-envelope/time-panel
diagnostics do not support a tail-range repair, and the coupled SwiGLU proposal
is stopped. This study returns to the original P and controller. It asks one
untested question about where the same controller is evaluated inside a solve.
The mathematical and experiment reviews state the limited rationale; they do
not predict a quality gain or claim a uniquely ODE-specific mechanism.

## Exact operators

```text
s(z) = .2+.8*sigmoid(t_func(z)/10),  D(z)=diag(s(z))
f(z) = D(z)Pz
first = f(h)

Euler:        h+first
Frozen K2:    h+first+.5*D(h)P*first
Midpoint:     h+f(h+.5*first).
```

Frozen K2 is the professor's correct Taylor polynomial for the conditional
linear field with D(h) held fixed. Midpoint instead approximates the nonlinear
field f(z), with controller reevaluation. Its second coefficient includes
`diag(Ph)*Js(h)*f(h)` in addition to `D(h)P*f(h)`. Neither model is relabeled as
an erroneous implementation of the other. Stage5 shared controllers across
two solves; Stage7 used a different nonlinear drift with a frozen clock. Those
studies did not test this exact contrast.

Use vector clocks only: scalar positive clocks change time along a fixed
linear orbit, whereas vector clock feedback can change directions. The FFN
retains two original maps separated by SiLU; there is no new controller,
spectral constraint, rate bound, temperature, loss or prediction head.

## Initialization and six arms

All ODE arms use width64, depth3, gate-hidden32, reference gate-hidden64 and
the same P/controller tensors and RNG construction. Zero only each final
controller affine to make D=.6I initially; do not zero P. Frozen K2 and midpoint
then have identical initial hidden/full functions and input/P derivatives.
Their controller-parameter responses with a nonzero output head generally
differ because intermediate controller features differ. The formal zero
velocity head masks these backbone responses initially. Euler's hidden
features and output-head Jacobian are not equal to the K2 pair.

| Arm | Operator | Dense MACs per FFN |
|---|---|---:|
| state_euler | original vector Euler, constant-clock initialization |16384|
| state_frozen | cached-clock K2, same initialization |24576|
| state_midpoint | nonlinear midpoint, same initialization |32768|
| swiglu_budget | SwiGLU nominal96, actual64 |12288|
| swiglu | SwiGLU nominal128, actual85 |16320|
| swiglu_compute | SwiGLU nominal256, actual170 |32640|

All three ODE arms have the same parameter count. Midpoint executes a second
controller evaluation as well as a second P application; cached K2 does not.
SwiGLU256 is a close computation control, with0.39% fewer dense FFN MACs than
midpoint and more parameters. Actual operations, parameters and runtime must
be reported separately; no dead computation is added to equalize cost. Cached
K2 is a mechanism control, not a proposed compute-efficient architecture.

Implement midpoint as the shared frozen expression plus
`(s(h+first/2)-s(h))*(Ph+P*first/2)`. This is algebraically identical to explicit
midpoint and bit-pairs its constant-clock forward path with frozen K2. All
controller/state derivatives remain live. Independent tests must compare it
against the explicit two-stage expression, not just its own decomposition.

## Fixed training and diagnostics

Eighteen cells: the six arms on Gaussian2D, Ring8 and product16. Training seed0,
5000updates, batch512, AdamW3e-4, weight decay0, clip1 and original cached-bank
pair velocity supervision remain unchanged. Keep the existing time sampler,
data versions, five-time diagnostics and Heun32 generation. No JiT-head,
endpoint clipping, Gaussian-reference or extra-label intervention is combined.

Copy every non-FFN initial tensor and runtime buffer from the same hash-bound
Stage15 step0 references. Both established SwiGLU reruns must reproduce their
historical saved states. The larger SwiGLU is newly trained under this same
protocol. Record every fixed checkpoint/curve, raw P spectra, first/midpoint
clocks, clock changes, finite midpoint correction and the omitted clock
Taylor term. These internal statistics do not establish a full-field property.

Use two one-thread Mac CPU workers, actual10s resource telemetry and idle-sleep
prevention. Reserve no GPU and freeze a new source/config/protocol runtime.
All previous runtimes, results and submitted Report1 remain unchanged.

## Decision before outcomes

Only state_midpoint can be nominated for this state-feedback hypothesis. It
must lower primary validation oracle MSE by at least10% against all three
SwiGLUs and at least5% against each ODE control on a nonlinear task. On every
other primary task it must be within5% of the better SwiGLU; on both nonlinear
tasks its full velocity Jacobian error and validation SW2 must each be within
10% of the better SwiGLU. These are practical screening rules, not significance
tests. Every control and endpoint remains visible even if it is unexpectedly
strong; a control gain does not establish midpoint's mechanism.

A nominee still needs an independent fresh holdout, matched-cost/parameter
interpretation, isolated timing and broader transfer before the goal can be
achieved. If midpoint fails, stop this setting without another step size,
solver stage/order, clock range/slope/temperature, controller width, initializer
or output-head rescue. Do not reopen the stopped SwiGLU gate family.

Before launch verify nonlinear Taylor/JVP and local-order identities, explicit
midpoint values/gradients, constant-clock legacy compatibility, initialization
pairing and its tangent limitation, actual dense work, full trainer/sampling/
checkpoint/resume and a real monitor heartbeat. Default predecessor models
must retain identical three-update trajectories. After completion audit every
checkpoint/diagnostic, snapshot data RNG, source/data/config/reference hash
and native final field/Jacobian replay before deciding on a candidate.
