# Zero-order term of the professor's original ODELayer

Registered before training. Stage12 rejected moving vector durations before P.
This test returns to the explicit sum(approx[1:]) option in the supplied
professor code, retaining rates after P, K1 and the middle SiLU.

## Operator and question

```text
F_gamma(h) = gamma*h + D(h)*P*h
block(h)   = F_gamma,2(SiLU(F_gamma,1(h)))
gamma      = 1 (map) or 0 (increment)
```

The outer time-conditioned residual, scalar/vector controller sizes,
positive sigmoid durations [.2,1], tau10, objective and optimizer stay fixed.
Only the zero-order identity inside each solve is removed. This is not the
previous activation-deletion or nonlinear-drift experiment. It does not
claim that the hidden block itself must predict an FM velocity.

The exact solve Jacobian is gamma*I+DP+diag(Ph)J_s. With both P matrices
zero, the map block is SiLU(h); the increment block and its first variation
in both P matrices vanish. Removing the identity is not automatically a
benefit, especially if combined with strong P shrinkage. No such shrinkage
is added here. Existing trained increments are material, with mixed
increment/input alignment; they do not support a universal cancellation story.

## Initialization control

With constant invertible D, map and increment solve families coincide:
P_increment=P_map+D^(-1). For a state-dependent D this shift is generally
not one fixed parameter translation. A practical legacy-initialization
comparison changes initial hidden functions despite matching raw P draws.

Therefore use two distinct comparisons, fixed before seeing outcomes:

1. Legacy initialization: scalar map/increment and vector map/increment.
   Each pair has identical raw tensors, RNG draws and budgets. Its hidden
   functions differ; do not call any gain a pure optimization improvement.
2. Vector paired initialization: zero the controller's last affine weights
   and biases so D=cI,c=.6 initially. Keep the map P unchanged and add I/c
   to each increment P. All initial hidden functions then match in exact
   arithmetic. Record actual float32 reconstruction defects, rather than
   asserting bit equality. Controllers subsequently train normally.

The paired initialization is an identification control, not a temperature,
normalization or loss trick. Its two model families are not globally the
same under trained state-dependent rates. At the paired initial function,
P gradients agree; last-controller gradients differ by the identity response.
This is derived and tested in zero-order-mathematics.md. A zero last affine
makes first-controller-layer gradients zero initially in both paired arms.

## Arms, budget and input law

Eight arms: scalar_k1, scalar_increment, vector_k1, vector_increment,
vector_const_map, vector_const_increment, swiglu_budget(ffn_hidden96),
swiglu(ffn_hidden128). Scalar gate hidden64; vector gate hidden32 with
reference hidden64. Backbone width64/depth3 is unchanged. Paired controls
are confined to the preceding vector lead; scalar correspondence is checked
algebraically, without adding another initialization grid.

Use Gaussian2D, Ring8 and product16 validated data, original cached-bank
pair-target FM, seed0,5000 updates,batch512,AdamW3e-4,wd0,clip1, identical
evaluation/sampling streams and Heun32. The population oracle is evaluation
only. Two one-thread CPU workers with10s telemetry; no GPU reservation.

All four variants within a gate-size family have the same parameters and
linear MACs. Omitting additions changes pointwise work; report observed
training time as well. Raw initial/1000/final checkpoints, all trajectories,
P spectra, durations and actual activation/increment RMS ratios are retained.

## Decisions fixed in advance

Primary metric is uniform-time validation population-oracle MSE. Complete
Jacobian errors use existing five fixed-time common panels; SW2 is the
existing validation sampling readout. All three increment arms are screened
separately against their corresponding map parent and both SwiGLU sizes.

To continue an increment arm, require at least10% lower primary MSE than
both SwiGLU budgets on at least one nonlinear task, at most5% regression
against the better baseline on the other nonlinear/Gaussian controls, and
Jacobian error and SW2 at most10% above the better baseline on both nonlinear
tasks. The supplying nonlinear task must also improve over its matching
map parent. These are practical screens, not significance tests. If several
arms pass, prefer lowest product16 oracle MSE, then fewer MACs, then lower
Ring8 oracle MSE. No metric/seed/temperature is tuned after outcomes.

Failure stops these settings without a K2, temperature or shrinkage rescue.
Passing permits a longer matched budget, fresh independent holdout and
capacity/gating controls; the screen alone cannot meet the full architecture
goal. An advantage only under legacy initialization is not by itself evidence
that removing the identity intrinsically improves the trained family.

## Verification

Freeze a new runtime and verify explicit deployed file hashes. Native tests
cover actual solve returns, unchanged middle activation/budgets/RNG, complete
derivatives, zero-P degeneracy, static paired optimizer correspondence and
the paired initial last-controller gradient difference. Check complete
nonzero-head function pairing in float64 and float32 on CPU/CUDA; predeclared
absolute/relative tolerances are2e-13/3e-6 respectively, and actual defects
are logged. Defaults must reproduce the frozen predecessor's predictions,
weights and Adam states in a nonzero-head trajectory check.

Audit all initial tensors, expected P shifts, zero controller finals and
all checkpoint data-generator states. Restore recorded runtime buffers.
Recompute oracle arrays/native validation readouts and independently replay
collected final predictions/Jacobians on Mac. No old runtime or Report1 is edited.
