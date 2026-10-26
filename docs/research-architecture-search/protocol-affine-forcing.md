# One bias-completeness test of the original ODELayer

Status: design registered before implementation and before any new training.
Selected for one bounded falsification after Stage22's native/replay/source
audits passed and both independent reviews gave conditional GO. Launch still
requires code review and full preflight. Date: 2026-10-25.

## Question and limits

The original projection is bias-free, although both practical GLU value
projections and the shared backbone contain biases. None of the earlier original
ODELayer controls added an independent value bias inside a solve. This is an
uncontrolled component difference worth testing once. It is not a new ODE
mechanism or evidence that this difference causes the existing performance gap.

Stage22's ordinary sigmoid GLU is worse than SwiGLU on both nonlinear tasks.
Its fixed-time raw-input mixture representation theorem did not imply an
optimization advantage for the complete normalized, time-conditioned network.
In particular, the present test is not justified by claiming that posterior
ambiguity, sigmoid features, or missing mode offsets explain the earlier errors.

Keep the original two K1 maps and their middle SiLU. For each map compare:

```text
parent:       F(h) = h + D(h) P h
static bias:  F(h) = h + D(h) P h + c b
affine drift: F(h) = h + D(h) P h + D(h) b
c = shift + scale/2 = 0.6
```

Here `D(h)=diag(.2+.8*sigmoid(g(h)/10))`. Both new arms start with `b=0`
and the already used constant-paired `D=cI` initialization. They have identical
initial functions and complete parameter Jacobians, including with a nonzero
receiving head. The parent lacks the new bias parameter columns. Common
parameters, buffers and RNG draws must match the current homogeneous parent.
Use `c=shift+scale/2` in code, rather than a separately rounded constant.

The difference is precisely `(D-cI)b`; its matched-input Jacobian is
`diag(b) J_s`. At the paired start the difference and its first parameter
derivatives vanish. An initial tangent-kernel advantage cannot explain an
affine-versus-static difference. A nonzero trained bias or conditional term
does not establish that this term caused a training benefit.

For input-frozen D, the affine arm is the K1 step of `z'=DP z+Db`.
If `Pr=b` is solvable, every Taylor order is a translation of the homogeneous
solve: `T_K(DP)(h+r)-r`, with a correspondingly shifted controller input.
The full two-solve/SiLU/LayerNorm network has no established equivalence or
strict capacity separation. The augmented spectrum adds a zero eigenvalue;
this does not bound the forcing magnitude or the complete network Jacobian.

## Fixed comparison

Run exactly six arms on Gaussian2D, Ring8 and Product16:

| Label | Model | Change |
|---|---|---|
| vector_parent | original vector K1 | Current homogeneous parent |
| static_bias | same | Add `c*b` in each solve |
| affine_forcing | same | Add `D(h)*b` in each solve |
| swiglu_budget | SwiGLU96 | Established smaller generic control |
| swiglu | SwiGLU128 | Established near-budget generic control |
| sigmoid_glu | GLU128 | Stage22 known generic control |

Use training seed0, 5,000 updates, batch512, AdamW learning rate3e-4,
weight decay0, gradient clipping1, the existing cached data banks and pair
velocity labels, uniform time sampling, the original direct-velocity head,
and the same non-FFN initialization references. No labels, oracle coefficients,
rotation, or analytic weights enter training or initialization. Validation uses
the unchanged native oracle MSE, sliced-W2 and five-time/512-anchor field and
Jacobian diagnostics; sampling remains Heun32.

Width64/depth3/gate-hidden32 adds 384 parameters in either bias arm, without extra dense MACs.
The Product16 counts are parent83,664/81,984 MACs, new84,048/81,984,
SwiGLU128 and GLU128 each83,598/81,792. These are close budgets, not identical
parameter counts or elapsed time. Rerunning established methods is an exact
compatibility control, not independent evidence of robustness.

## Registered decision

Evaluate only after all18 cells are terminal and independently audited. For
either new arm, a practical lead requires all of the following:

1. On at least one nonlinear task, oracle MSE is at least10% below the parent
   and the better of all three generic controls.
2. On every task, oracle MSE is no more than5% above the better generic control.
3. On each nonlinear task, Jacobian error and sliced-W2 are each no more than10%
   above the best generic control for that metric.

A static arm meeting these conditions is a **BIAS_COMPLETION_LEAD**, which may
be useful without supporting conditional forcing. An affine arm meeting them
is a practical bias-completion lead too. Attribute an additional conditional
mechanism only if affine MSE is at least5% below static on a qualifying nonlinear
task and no more than5% worse than static on any primary task. Do not promote a
Gaussian-only improvement, a train-loss gain, active new parameters, or a
conditional-versus-static gain that still loses to the generic frontier.

These are engineering screening margins on exposed validation data, not a
statistical significance test. Any lead still needs isolated timing, a fresh
independent holdout, and nonlinear transfer evidence before architecture
acceptance. This study alone cannot complete the architecture goal.

## Required evidence and failure boundary

Before launch, verify analytic input/parameter derivatives, whole-model initial
functions and all parameter responses for the static/affine pair, no-RNG bias
construction, actual dense MACs, truthful statistics, short full-trainer/resume
identity, old-family compatibility, config propagation and real monitoring.
Freeze source, protocol, reviews and plans before exposing the final ready
receipt; never edit that receipt after it can be consumed.

Save initial and every1,000-update checkpoints, all native trajectories and
five-time endpoint arrays. Independently replay all18 endpoints and audit
108 checkpoints/90 arrays, the established generic histories, data RNG pairing,
and every reported aggregate. Monitor at10s with two one-thread CPU workers and
idle-sleep prevention; reserve no GPU for this tiny study.

For every original solve log P spectra and actual hidden-input quantities:
`DPh`, total bias contribution, `c*b`, `(D-cI)b`, the latter's mean and centered
RMS, input-normalized scales, bias norm and duration summaries. Total first-term
statistics must include the actual bias. Raw P spectra exclude forcing and
clock derivatives and cannot certify the full map.
Also log the analytic local `||diag(b) J_s||_F` contribution; it is zero in the
static arm and is not the complete network's Jacobian norm.

At each final affine checkpoint, additionally evaluate replacing every
`D(h)*b` by `c*b` with all trained tensors fixed, on the same five native
anchor sets. Report original and counterfactual field/Jacobian errors and
prediction/Jacobian changes. This is an output-dependence diagnostic; the
separately trained static arm is the training control. A fixed-c replacement
removes a possible mean shift as well as sample dependence, so do not interpret
its difference as a pure state-dependence effect.

Failure closes this fixed bias-completeness setting. No K2/order/stage, clock
range/temperature, forcing strength, initializer, width, head or objective
rescue is permitted. It does not reopen the rejected forced-gate or state-clock
families. Submitted Report1 is untouched; the work is local and not pushed.
