# ODE mechanism: current evidence

## Latest: the stage-order computation is measurable

Fresh six-model native diagnostic at2f8dd60and independent NumPyreconstruction
pass, vector error<=7.11e-15. New-coordinate local mean-map changes are median
.154Ring8/.222Product16of the original increment; mean collapse raises fixed
selection field MSE.011->.186and.020->.449. The scheduled computation is used.
The four-stage bracket predicts its small-horizon limit, cubic remainder
orders2.977/2.996; unit-horizon relative errors.408/.441remain substantial.
This is a feature-level explanation, not proof of sampling benefit.
All new full-bank numerical tasks now complete and independently reproduce.
Ring8anchored improves.111->.103under inner4->16while new changes.104->.105;
their coarse point ranking reverses. Keep this distinct from the same-function
optimizer-coordinate effect. Densityqualification is still running.
Canonical: coordinate-control/schedule-decomposition/analysis-report.md and
coordinate-control/numerical-resolution/analysis.md.

## Latest: mean-preserving coordinate control complete

New-control follow-up: two portable checkpoints preserve native
field/bank0outputs exactly and copy the original P2arrays byte-for-byte.
The existing entries accept manifest-sized two/eight-case populations;
both new and original derivative/formal regressions pass. Eight complete
native-sampler-map finite differences agree within3.68e-8at the finest stencil.
The augmented observer and its math remain unchanged. Fresh documented CPU
witness passes, exits0/0/0/0. Independent entries reproduce within1.78e-14;
eight bounded numerical scores reproduce within2.78e-17. Frozenfd7f09b
densityqueue43084and numericalqueue43096are running with live monitored
workers.1454-7occupied; noGPUallocated. Full32new density rows and four
32768-source numerical tasks remain to finish and be independently collected.

Two20kCPUtrajectories at frozen cdd79e5and all common-bank evaluations
finished. All10model/task rows independently reproduce within2.78e-17.
Newrelease pooled SW2Ring8.104/Product16.072; shared.091/.064; anchored
.111/.073. Shared wins5/8Ring8banks (interval includes0) and8/8Product16
banks. Neither paired new-versus-anchored improvement is resolved.
Four same-function probes show effective-mean/native-common-P step ratios
1.636--1.648in anchored coordinates versus1.000in zero-mean coordinates.
Exact fields and effective gradients agree before the update. These use
fresh AdamWstate; they do not replay historical momentum. Reparameterization
changes optimization without changing capacity. It modestly changes point
SW2and does not close Product16's sharing gap. Ordinary Product16SwiGLU
still wins at.044. New-control solver/density measurements remain pending.
Current evidence: coordinate-control/analysis-report.md under the canonical
2026-11-16-ode-mechanism directory. Reports1/2and native image code unchanged.

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

Formal measurement live: CPUqueue35441(two Ring8ordinary refs), GPUqueue
3447050(four active workers on4-7, two Product16ordinary cases pending).
All128rows retained, no training, source/data fixed. Throughput profile
chooses Maccoordinate and145batchedCUDA; the145CPUfallback is much slower
and notselected. ActualCPU/CUDAB256observer error max4.57e-10. Models and
budgets are unchanged. Next: monitor, independentfullcollector, paired
density/time-bin analysis.

## Full-density CPU checkpoint

Mac Ring8 Free ISRU and SwiGLU finished all32formal rows, queue35441 terminal
with exit0. Independent density reconstruction agrees within5.33e-15.
Full banks expose rare resolution-sensitive paths not seen in the eight-point
pilot: failing source observations are FreeISRU58/23and SwiGLU25/11for
original/projected fields, each outof1024. Every bank contains at least one
unresolved source. No point is removed and no full density ranking is made.
Continue the frozen GPUqueue and audit all cases before registering refinement.
Existing SW2comparisons are separate completed evidence.

## Full-source density follow-up

All128original observations completed and independently reconstruct within
4.62e-14. All16model/field cells contain resolution-sensitive source paths.
The registered per-source4096/8192/16384refinement retains every original
point and gate. Fresh CPU/CUDAwitnesses pass; native endpoints remain exact.
Both Mac Ring8ordinary controls finished, all4cells qualify independently
within1.16e-14. CUDAqueue3584295refines the remaining6cases at405a201.
The new statistical reader preserves common-path and bank pairing, exposes
only bank0time profiles and also checks SW2on the same1024inputs. No new
architecture/training is introduced. Reports1/2remain finalized and fixed.

Execution note: a Git bundle command naming a short unreferenced SHA was
refused as empty. Recreated it using the named HEADref, transferred it,
verified SHA256and added a separate405a201worktree. The running3a7d5fe
producer and environment were unchanged. Timing-only small-subset checks
favor CPUfor future tiny Ring8refinements and CUDAfor projectedProduct16.

## Product16 density explanation rejected

All four Product16 models and both fields pass independent adaptive audit,
max error4.98e-14. Stage-minus-shared KL is -0.247original/-0.204projected;
four-comparison-adjusted intervals exclude0. This rejects lower joint density
error as the explanation for the earlier large-bank SW2sharing lead. No
oracle window is selected and confirmation time profiles remain unopened.
Projected component variance ratios are1.190shared,1.146stage,1.085FreeISRU
and1.212SwiGLU, versus population target1. Pairwise sign covariance is near
the native target sampling floor. Width is a concrete discrepancy, not a
complete causal account of KL. A moment-based population KLbound and exact
target endpoint calibration are derived in product-metric-diagnosis.md.

The next structural control removes an optimizer-coordinate confound in the
anchored P+Delta release: orthogonal zero-mean contrasts keep mean(W)=P
while representing the same untied function class and parameter count.
Algebra/common-gradient/actual first AdamWupdate checks pass; full-model
implementation and matched20k controls are not launched yet.
Current analysis: density-transport/population-analysis.md. Reports stay final.

## Full population-density conclusion

All eight adaptive cases complete, all16cells qualify; independent error
4.98e-14and16mean/SDrows reproduce via fsumwithin1.12e-16. Ring8shared/stage
KLties at this evaluation precision (adjusted intervals include0); Product16
reverses the earlier SW2lead in KL. No oracle-window intervention is eligible.
Full statistics and three inspected scientific figures are published in the
canonical density-transport folder. All workers exit0and our GPUcontexts
are released. Next: implement/check the registered mean-preserving release
before the two matched training controls. No finalized report is changed.
