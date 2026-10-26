# Pruned low-rank state allocation: one fixed Phase A screen

Registered 2026-10-25 before formal training or outcome inspection. Execution
requires focused mathematical tests, independent code review and full preflight.
This is one joint state/budget reallocation, not an augmentation-only theorem.

## Evidence, operator and initialization

The finite clock-collapse diagnostic increases all 15 exposed-panel field errors;
its equal-time ratios are 20.19 on Ring8 and 2.82 on Product16. Conditional
computation therefore helps those frozen weights. Ordinary SiLU128 improves
Product16 but worsens Ring8 against the original block. Together these findings
motivate retaining clocks while testing a larger nonlinear state under a fixed
matrix budget. Neither observation proves that width is the remaining cause.

Keep the two original input-frozen K1 solves and middle SiLU. With input width d,
state width r, rank q and controller hidden width g, compute

```text
y = pad(h) + s1(h) * U1 V1 h       U1:r×q, V1:q×d, gate1:d→g→r
z = SiLU(y)
out = z[:d] + s2(z) * U2 V2 z     U2:d×q, V2:q×r, gate2:r→g→d.
```

Both clocks remain `.2+.8*sigmoid(logit/10)`. This is exactly a lifted original
K1 map after pruning unused first-map extra-input columns and second-map
extra-output rows. No learned encoder/decoder, extra bias, activation, loss,
prediction head, rate bound, regularizer or higher Taylor order is added.

Construct the existing random d=64 / gate=32 reference and consume exactly its RNG
draws. Initialize common low-rank cores from its CPU-float64 truncated SVD,
using square-root factors cast to native precision; no data or oracle enters.
Use the first g reference controller features and zero final controller maps.
For the expanded model, copy U1's core into extra rows; initialize V2's extra
columns and controller2's extra input columns to zero. Extra SiLU features are
nonzero, and new receiving columns have live responses under a nonzero head.
Do not zero both sides of the added path. Copied cores are SVD-balanced; the
entire expanded U/V pair is not globally balanced and is not rescaled.

Narrow/expanded initial functions agree algebraically, with explicit floating
GEMM tolerances. Their native zero-head kernels coincide by masking; under a
nonzero head, added parameter responses generally make full kernels different.
The original dense parent has a different initial map and feasible P set.
ANODE global-flow topology and global capacity-separation claims do not apply.

## Fixed six-arm comparison

| Label | State/rank/controller | Per-FFN dense MACs | Parameters |
|---|---|---:|---:|
| vector_parent | dense64 / gate32 |16384|16576|
| lowrank_budget |64 /32 /32|16384|16576|
| lowrank_narrow |64 /21 /21|10752|10922|
| lowrank_augmented |128 /21 /21|16128|16362|
| swiglu | nominal128, actual85 |16320|16554|
| silu128 | ordinary hidden128 |16384|16576|

The low-rank formula is `2(d+r)(q+g)` MACs, plus `2g+d+r` biases for parameter
count. q=g=21 is the integer equal-split allocation under the original budget;
it is not chosen by a rank search. On Product16 the candidate has 83022 full-model
parameters and 81216 MACs, compared with SwiGLU 83598/81792. Actual calls/counts
must reproduce these values. The cheaper narrow comparison alone cannot
establish useful allocation. Matrix MACs do not establish latency.

The already audited Stage23 SwiGLU96 and GLU128 remain hash-bound historical
generic references at no greater than the near-matched budget. They are not
new training cells or independent repetitions. All generic guards below use
the best value for each metric among those two references and the two fresh
generic arms. Larger historical SwiGLUs are visible context, not matched-cost
screening comparators.

Run 18 cells: these six arms on Gaussian2D, Ring8 and Product16. Preserve seed 0,
5000 updates, batch 512, AdamW 3e-4, weight decay 0, clip 1, existing cached banks
and pair-velocity labels, uniform times, direct velocity prediction, hash-bound
common non-FFN references and Heun32. The new family zeros its own controllers;
its configuration's legacy initialization option prevents applying unrelated
legacy return-map interventions. It does not mean random-clock initialization.

## Registered decisions and continuation

For a near-budget low-rank arm, a practical lead requires at least 10% primary
oracle-MSE improvement over the better generic control on one nonlinear task,
no more than 5% primary regression against that generic frontier on any task,
and nonlinear Jacobian/SW2 errors within 10% of the better generic value.
Also require at least 10% improvement over the original parent on the qualifying
task. Retain a successful factorized control under its true interpretation.

An augmentation-allocation claim additionally requires the expanded arm to
improve MSE by at least 10% against both lowrank_budget and lowrank_narrow on the
same qualifying task, with no primary regression over 5% against either on any
task. This is joint allocation evidence, not pure width or ODE uniqueness.

A narrow cheap lead instead requires all primary errors within 5% and nonlinear
Jacobian/SW2 within 10% of the generic frontier, with at least 20% fewer full-model
parameters and dense MACs than SwiGLU128. It must subsequently face SwiGLU84,
whose actual56 channels match its MAC budget. Neither a narrow nor other lead
completes the goal at Phase A.

Only a practical lead proceeds to a separately registered Phase B: isolated
timing, remaining mechanism/initialization or constant-clock controls where
needed, a fresh independent in-distribution evaluation and frozen-architecture
training on a fresh nonlinear target. No cheapness claim rests on concurrent
wall time. No positive control is discarded because it is not the proposal.
Failure stops this fixed allocation without rank/state/controller sweeps,
SVD rebalance, order, step, clock, activation, width, head or objective rescue.

## Required verification and interpretation

Before launch verify actual costs, reference draws/common tensors, full input
and parameter derivatives, live added paths, truthful rectangular/lifted
operator statistics, resume identity, historical compatibility, all 18 configs
and real monitoring. Freeze source/protocol/reviews before a finalized ready
receipt; later launch must be gated explicitly on successful checks.

Save six native checkpoints per cell, all trajectories and five-time endpoint
arrays; independently audit 108 checkpoints/90 arrays and 18 native field/J
replays. Historical generic reruns are compatibility checks, not repetitions.
Monitor two one-thread CPU workers every 10s, without reserving a GPU.

Record full zero-padded P spectra and actual activations/factors. Nonzero
conditional-generator eigenvalues equal those of VDU; padded coordinates can
affect the function while remaining invisible to eigenvalues. Probe-mean VDU
statistics are not per-example or complete-field guarantees; factor norms
are gauge dependent. The new layers deliberately stay outside legacy ode_layers.
Thus legacy anchored_jacobian arrays equal native for this family and are
**not** clock-derivative ablations. Native full-field/J evaluation remains valid.

The submitted Report1 and prior frozen runtimes remain unchanged. This work
remains local/unpushed. The full useful-architecture goal remains active.
