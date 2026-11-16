# ODE mechanism: current evidence

## Structural controls

State feedback and sharing P are positive within-family interventions at
this budget. Free Taylor/free midpoint coefficients match prescribed-stage
sampling, though prescribed midpoint retains numerical order 2. Sharing
primary iid SW2 is Ring8 .091/.111 and Product16 .064/.073 (shared/stage),
8/8 paired banks each, at equal dense work and fewer parameters. Larger
ordinary Product16 controls remain stronger and cheaper.

## P1 symmetry complete

Eight frozen model/task cells; original/projected group-stratified SW2:
Ring8 shared .072/.024, stage .096/.021; Product16 shared .059/.055,
stage .068/.063. Ring8 reverses, Product16 retains the sharing comparison.
Sharing lowers breaking error but worsens a preserved field component.
The selected symmetry alone is insufficient across tasks.

## P2 density qualification complete

Exact along-model-path identity:
 d log(rho/p)/dt = -[div(v-v_oracle)+(v-v_oracle) dot grad(log p)].
Expected terminal log ratio is KL for the exact continuous flow under the
stated smoothness/integrability assumptions. This is established transport
calculus, not a proposed theorem or ODE-specific performance result.

Both native oracle laws, gradients/finite differences, continuity and
Gaussian rotational/radial examples pass. All eight learned models preserve
native Heun endpoints exactly. Full native-map logdet finite differences
pass eight checks, maximum 5.25e-8. Data and CPU anchors are prebound.

The 48-row pilot qualifies 6/16 cells at 512. Only the ten unresolved cells
were refined; all qualify at 1024 under unchanged tolerances. No 2048 pilot
is needed. Maximum independent errors are 2.94e-14 (initial) and 2.14e-14
(refinement). Both queues are terminal; no GPU was used for tiny pilot
batches. Two diagnostic figures have been inspected.

On the new fixed-time field banks, Product16 shared/stage source RMS is
2.116/1.995; SwiGLU is 2.239. These local norms do not reproduce the known
sampling ordering. Do not substitute this proxy for the signed path integral.
The eight numerical sources are not a population-KL ranking cohort.

Next: all eight checkpoints, both fields, four prebound 256-source banks at
common 1024/2048 steps. Add real-entry CPU/CUDA checks and a timing benchmark,
then monitor and independently reconstruct 128 rows. If numerical gates fail
on full-source tails, retain them; never drop points or loosen tolerances.
Read density-formal-next.md before implementing this stage.

Raw: /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport
Warm CPU: /Users/mr.z/ode-workspace-prep/.venv-test/bin/python
145: /data2/zz5070/miniconda3/envs/py312/bin/python, Torch 2.8.0+cu128
Physical 0-3 forbidden; 4-7 only when idle. Never reclaim/share foreign tasks.
Final reports immutable; all new work goes outside reports/.

Full-entry benchmark evidence: coordinateCPUprojectedProduct16state29.520s
per32steps, batched62.291s. ExactbatchedJacobian errors0and observer errors
<=3.56e-15, but batching is rejected for MacCPUbecause it is slower at the
actual batch. Keep coordinateCPU; benchmark145CPUas a no-GPUfallback,
and CUDAonly when a permitted card becomes idle. Full128-row scope stays fixed.
