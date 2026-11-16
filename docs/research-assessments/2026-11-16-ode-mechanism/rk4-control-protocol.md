# Matched training test: does a more faithful hidden ODE help?

Registered before new training. The parameter-free modified-equation calculation
predicts the trained midpoint error well, but that does not say whether the
error helps the task. Switching a solver at frozen weights cannot answer how
training with a more accurate solver behaves. This is the next causal test.

## Change exactly one factor

Keep y'=P ISRU(y), a=1.2, internal interval[0,1], width64, depth3 and both
independent local operators around each FFN's SiLU. Replace four explicit
midpoint steps by two classical RK4 steps. Both use exactly eight P applications
per local solve. Complete parameter/buffer tensors and initialization RNG must
match before training. Extra weighted additions differ; equal matrix work is
not a claim of equal wall time.

The RK4 numerical step is copied from torchdiffeq's classical rk4_step_func at
commit657943acefa826ef04c025ebeb1ff5e9d60dc268, with its MIT notice retained.
Only an autonomous callback connects it to the existing layer. The pinned
native trainer, optimizer, objective, sampler and evaluator are reused. No
activation, spectral constraint, learned coefficient, data or loss is changed.
This is fourth-order integration with two substeps, not Taylor K8.

## Fixed training and controls

Train only two new trajectories: Ring8 and Product16, seed0,20k steps,
batch512, AdamW3e-4, weight decay0, gradient clip1. Reuse the existing shared-P
midpoint4 controls and strong ordinary SwiGLU/FreeISRU results on the same
source-bound banks. Do not narrow or replace any baseline. Check exact common
parameter/RNG initialization and native data-generator states in bounded
continuations before dispatch. Existing training/final-test scripts remain
unchanged; evaluation scripts change only their model-builder callback.

Keep the original milestone checkpoints, native32-step training evaluations,
projection/update spectra and final native test examples. Final comparisons
use fixed20k weights and the original eight selection/confirmation banks,
Heun128,4096sources per bank and256directions. Do not select checkpoints using
confirmation results or compare small-source pilot scores with these values.

## Primary hypotheses and decisions

H_quality: new RK4 has lower pooled target SW2 and lower mean paired-bank SW2
than midpoint4 on both tasks at the same training budget. Report both pooled
values and paired bank differences; they are different estimands. Conditional
bank intervals do not estimate independent-training variation.

H_fidelity: RK4 has lower normalized native-to-continuous hidden-map error
on both tasks. Use the same80fixed interpolation inputs, all six layers and
qualified two-tolerance continuous references, captured separately at each
model's actual entrances. Normalize pooled squared error by the corresponding
continuous-map displacement energy, retaining zero/unresolved cases. No
generation reference metric is substituted for actual target SW2.

Report nominal four-comparison-adjusted conditional intervals for two tasks by
two hypotheses; fixed bank pairing is retained. No arbitrary5%margin. A mixed
task result fails the strong two-task claim. If fidelity improves without SW2,
greater continuous accuracy alone is not a sufficient generative advantage.
If SW2 improves without fidelity, do not attribute the gain to fidelity. Even
both passing does not establish a useful candidate unless the strong ordinary
controls are also competitive. No image run is selected before those results.

## Execution gates and budget

First prove the copied RK4 algebra against an independent implementation,
full input/parameter gradients, fourth-order refinement, exact eight P calls,
and common initialized tensors/RNG. Check native8->16resume versus continuous16
and real common-bank evaluator entries on both tasks. A fresh frozen documented
pass must reproduce these checks. Preserve failed entries rather than retuning.

Use two one-thread MacCPUworkers. Record per-step speed, native progress and
actual completion; attach a persistent monitor and source-bound evaluators.
145physical0-3remain prohibited and4-7must be idle before use. No job is stopped
or shared, no packages are installed, and finalized reports/image code stay fixed.
