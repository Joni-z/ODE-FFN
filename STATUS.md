# Current project status

## Current priority: identify the ODE contribution

**The stage-flow decomposition is complete and independently reconstructed.**
The mean-preserving models actively use different stage directions: mean
collapse changes oracle field MSE Ring8 0.011->0.186, Product16 0.020->0.449
on the fixed selection field bank. Local mean-map changes are median
0.154/0.222 of the native feature increment. The four-stage bracket expansion
has cubic small-horizon remainder; unit-horizon relative errors remain
0.408/0.441. These explain feature computation, not a generation advantage.
[Analysis and figures](docs/research-assessments/2026-11-16-ode-mechanism/coordinate-control/schedule-decomposition/analysis-report.md).

All four new numerical tasks also completed, and twelve pooled/96bank scores
independently reproduce within2.78e-17. New-release Ring8 SW2is0.104at4steps
and0.105at16; anchored changes0.111->0.103. The coarse point improvement over
anchored is not invariant to inner refinement. Shared remains lower in the
pooled score. [Complete numerical table](docs/research-assessments/2026-11-16-ode-mechanism/coordinate-control/numerical-resolution/analysis.md).
The two full-density workers remain live. Fresh adaptive CPUentry witnesses
and their independent reconstruction pass; no population refinement has
started. A new145inventory finds5-7free; recheck before any CUDApreflight.
Keep existing CPUjobs intact and use any future CUDAwork only for distinct
pending observations. No foreign job is stopped or shared.

**New-control numerical/density observations are running on Mac mini.** Both portable
checkpoints reproduce their complete native field and bank0samples exactly.
P2density arrays are copied byte-for-byte. Two new-model derivative/formal
entries and the original eight-model regression pass; eight full-native-map
finite-difference checks agree within3.68e-8at the finest stencil. The density observer is unchanged.
The fresh four-command documented witness exits0/0/0/0. NumPy independently
reconstructs both formal-entry witnesses within1.78e-14and all eight bounded
numerical scores within2.78e-17. Sequential256-source timing is complete.
Frozenfd7f09b: densityqueue43084, workers43093/43094; numericalqueue43096,
workers43098/43099initially. ActualPIDs and advancing integration steps are
verified. These are observations, not additional training. Raw row/bank
outputs are not final comparisons until independent collection/refinement.
145physical4-7are occupied; noGPUis allocated.
[Exact scope and invocation](docs/research-assessments/2026-11-16-ode-mechanism/coordinate-control/observation-protocol.md).
[Readiness, timing and actual launch](docs/research-assessments/2026-11-16-ode-mechanism/coordinate-control/observations/launch-verification.json).

**The mean-preserving stage control is complete.** It represents the same
function class as anchored stage release while keeping the effective mean
matrix equal to P. Two 20k seed0 CPU trajectories and all common-bank
evaluations finished; independent metric reconstruction agrees within2.78e-17.

| Fixed20k / pooled SW2 | Ring8 | Product16 |
|---|---:|---:|
| Shared P |**0.091**|0.064|
| Anchored stage release |0.111|0.073|
| Mean-preserving stage release |0.104|0.072|
| Free ISRU |0.116|0.055|
| SwiGLU reference |0.120|**0.044**|

Shared P wins8/8Product16banks against the new release; Ring8 wins5/8and
its paired interval includes zero. Improvements over anchored release have
unresolved paired intervals on both tasks. At exactly the same frozen
functions, fresh native AdamW probes show mean-step amplification1.636--1.648
in anchored coordinates versus1.000in mean-preserving coordinates. This
establishes an optimizer-coordinate effect, not its historical causal share
of the final gap. The new control's solver/density measurements remain
pending. No new GPU/image job was launched; finalized reports remain fixed.
[Completed analysis, statistics and figures](docs/research-assessments/2026-11-16-ode-mechanism/coordinate-control/analysis-report.md).

**Product16 rejects the proposed density explanation of sharing.** After the
same reflection projection, estimated joint KL is 0.668 for Shared P versus
0.464 for Stage P, 0.289 for Free ISRU and 0.894 for SwiGLU. The paired
stage-minus-shared contrast is -0.204, adjusted interval [-0.320,-0.092].
All eight Product16 cells qualify numerically and independently reconstruct.
Shared P has broader components (relative variance 1.190 versus Stage P's
1.146; target 1). Pairwise dependence is near the target sampling floor.
[Analysis, figure and next control](docs/research-assessments/2026-11-16-ode-mechanism/density-transport/population-analysis.md).
Do not launch the rejected oracle-window branch. The full comparison is complete: all sixteen model/field cells qualify.
Ring8 paired KL intervals include zero, so neither field has a resolved
sharing lead. All workers exited0; our CUDAcontexts are released.

All 128 original density observations completed successfully. Independent
NumPy reconstructs every identity within4.62e-14. Some sources in every
model/field cell fail the1024/2048resolution gates, so those observations
require the registered refinement before density ranking.
[Full original audit](docs/research-assessments/2026-11-16-ode-mechanism/density-transport/formal-full-independent.json).

Mac FreeISRU/SwiGLU Ring8refinements are complete, queue36629exit0. All4096
source observations across their four fields now qualify, with no removal
or relaxed threshold. Independent source selection, identities and merges
agree within1.16e-14. One source required16384steps.
[CPU independent audit](docs/research-assessments/2026-11-16-ode-mechanism/density-transport/adaptive-mac-independent.json).

The six 145adaptive cases are complete, queue3584295terminal, source405a201.
All workers exited0and every cell qualifies. Only idle physicalGPUs4-7were
used; no foreign task was stopped or shared. No new training or image run was added.
The full paired analysis and figures are complete: no Ring8KLsharing
advantage is resolved; Product16favors stage matrices in KL. The rejected
oracle-window branch remains unlaunched. The next registered structural
control preserves the mean effective matrix while releasing the stages.
[Registration](docs/research-assessments/2026-11-16-ode-mechanism/density-adaptive-followup.md),
[execution](docs/research-assessments/2026-11-16-ode-mechanism/density-adaptive-execution.md),
[CUDA launch](docs/research-assessments/2026-11-16-ode-mechanism/density-transport/adaptive-cuda-launch.json).

P2numerical qualification is complete:16/16pilot cells pass, six at512and
ten at1024, with independent errors<=2.94e-14. Native Heun endpoints stay
unchanged. [Diagnostic figures](docs/research-assessments/2026-11-16-ode-mechanism/density-transport/analysis-report.md).

**Eight new 20k trajectories and their independent evaluations are complete.**
Free Taylor coefficients (8c10a8d), state-evolving midpoint (1ff0d02), and
free midpoint stages (9a73953) all exited zero. Their28selection and14
confirmation rows independently recompute. All training/evaluation queues
for these trajectories are terminal. The two released-stage-operator runs
also completed20k with exit 0; all16selection/8confirmation rows independently
recompute. The latest sharing cohort uses its own fresh prebound arrays:

| Model | Ring8 SW2 | Product16 SW2 |
|---|---:|---:|
| Shared generator |**0.091**|0.064|
| Released stage matrices |0.111|0.073|
| Free ISRU |0.116|0.055|
| SwiGLU reference |0.120|**0.044**|

Sharing wins all8paired banks per task at identical dense work and about56%
fewer parameters than released stages. Released matrices are active, and
Product16 field fitting improves while sampling worsens. This supports the
sharing/autonomy constraint in this comparison; ordinary Product16 controls
remain stronger and cheaper. [Completed sharing analysis](docs/research-assessments/2026-11-16-ode-mechanism/sharing-results/analysis-report.md).

Earlier coefficient/state cohort, on its separate confirmation arrays:

| Model | Ring8 | Product16 |
|---|---:|---:|
| Fixed exponential K8 |0.088|0.056|
| Free Taylor coefficients |0.089|0.055|
| State flow / midpoint4 |0.071|0.052|
| Free midpoint stages |0.071|0.052|
| Free ISRU |0.082|0.051|
| SwiGLU |0.091|0.034|

The positive within-family comparison is state feedback; prescribed
coefficients do not show a consistent added benefit. Those earlier comparisons
retain shared P; the later matched stage release above tests sharing explicitly.
[All scores and paired effects](docs/research-assessments/2026-11-16-ode-mechanism/completed-comparison.md).
[Interpretation and trained-state convergence](docs/research-assessments/2026-11-16-ode-mechanism/primary-findings.md).

All18resolution cases independently recompute within2.78e-17. Full-sample
internal4->16 changes fixed-midpoint SW2 Ring8 0.071->0.074, Product16
0.052->0.055. Ring8's within-family gain remains; Product16 approaches
fixed K8's0.056. [Verification and scores](docs/research-assessments/2026-11-16-ode-mechanism/numerical-resolution/COMPLETE.json).

Numerical queue25633 stopped after21successful tasks and one failed float32
C8 trajectory-equivariance witness; three symmetry tasks were unstarted.
The failed bundle is preserved. All seven symmetry cases completed in a
separate float64 bundle, queue27668 terminal. All25numerical cases and 78 scores
independently recompute within2.78e-17. The Ring8 state-flow lead over ordinary
controls disappears after C8projection on the paired stratified bank.
[Completed numerical findings](docs/research-assessments/2026-11-16-ode-mechanism/numerical-findings.md).
[Protocol](docs/research-assessments/2026-11-16-ode-mechanism/numerical-followup.md),
[launch](docs/research-assessments/2026-11-16-ode-mechanism/numerical-launch.json).
The single-generator [sharing control](docs/research-assessments/2026-11-16-ode-mechanism/operator-sharing-control.md)
has passed initial-function, gradient, native continuation and fresh
documented evaluator checks. Both seed0 runs completed20k on Mac mini CPU,
queue27486, workers27487/27488, frozen8205c5a, both exit 0. Formal initialization,
recipe and data-RNG states match the shared controls at0/15k/20k. Evaluation
watcher27667 completed all16evaluation invocations on the new common arrays;
all six ordinary/shared references received those same arrays.
[Actual launch receipt](docs/research-assessments/2026-11-16-ode-mechanism/sharing-launch.json).
Frozen probes confirm nonzero learned stage actions and ordering sensitivity;
these are not substitute retrained generation comparisons. Same-host dense
MAC accounting and CPU forward/sampling benchmarks pass for all eight cases.
The complete new-cohort analysis is linked above. Its new512-point outer
refinement independently recomputes, maximum128->512SW2 shift2.464e-4.
Full-bank refinement also completed on idle145physicalGPU4, frozen7735abe;
the job has exited. All16pooled/128bank scores independently recompute within
2.78e-17, and native CUDA/CPUscores agree within3.44e-8. Outer128->512leaves
the gap. Inner4->16gives shared/released Ring8 0.091/0.103and Product16
0.069/0.082; Ring8bank support weakens to5/8while Product16stays8/8.
[Full refinement, figures and receipts](docs/research-assessments/2026-11-16-ode-mechanism/sharing-results/full-resolution/analysis.md).
The eight-case frozen symmetry intervention is complete at 5e78e6a;
all cases exit 0 and queue 2996899 is terminal. Independent NumPy recomputes
64 bank / 16 pooled scores plus 32 resolution checks within 1.25e-16.

| New group-stratified cohort | Shared P original/projected | Stage P original/projected | SwiGLU original/projected |
|---|---:|---:|---:|
| Ring8 / C8 |0.072 /0.024|0.096 /0.021|0.101 /0.022|
| Product16 / selected reflections |0.059 /0.055|0.068 /0.063|0.040 /0.039|

Ring8 reverses the sharing comparison after projection; Product16 retains it.
The single selected-symmetry explanation fails. Original Ring8 sharing reduces
breaking error but increases preserved-field error. Product16 still has lower
stage field MSE but worse generation after projection. Ordinary controls remain
in every comparison; no image promotion follows.
[Completed analysis and examples](docs/research-assessments/2026-11-16-ode-mechanism/sharing-symmetry/analysis-report.md).
These scores use fresh group-stratified arrays and must not be numerically
pooled with the earlier iid cohort. GPU 4/5 contexts have been released; 6/7 were
occupied and skipped. No other task was stopped. The next prepared diagnostic
studies the exact density-error source div(p*e), not another response function.
[Mathematics and bounded plan](docs/research-assessments/2026-11-16-ode-mechanism/post-symmetry-decision.md).
No new training or image job was submitted.

[Coefficient invocation and readiness](docs/research-assessments/2026-11-16-ode-mechanism/execution.md),
[coefficient launch](docs/research-assessments/2026-11-16-ode-mechanism/launch.json),
[state-flow gate and invocation](docs/research-assessments/2026-11-16-ode-mechanism/state-execution.md),
[state launch](docs/research-assessments/2026-11-16-ode-mechanism/state-launch.json).
Raw root: `/Users/mr.z/ode-workspace-prep/coefficient-20261116/`.
145 physical 0-3 remain prohibited. The earlier GPU6portability attempt
stopped when a foreign task appeared; later dedicated witnesses and inference
use their own receipts. No other task was stopped.

On16 November the user asked for a falsifiable mechanism study instead of
another response search. [Research plan](docs/research-assessments/2026-11-16-ode-mechanism/research-question-card.md)
and [derivation](docs/research-assessments/2026-11-16-ode-mechanism/mathematical-basis.md):
hold ISRU fixed, compare exponential K8 with a shared-operator polynomial
whose higher-order coefficients are free but initialized identically.
Then conditionally test true state evolution at matched dense work and
check whether any gain survives internal solver refinement. Diagnose actual
velocity-error transport using the known synthetic oracle.

This supersedes the immediate arctan-only image-readiness recommendation
below as the next priority. The completed screen remains valid evidence.
Mathematical examples and native execution witnesses passed. Current formal
runs are described above. Finalized reports remain fixed.

## Completed response comparison on Mac mini CPU

[Response-family comparison](docs/research-assessments/2026-11-14-next-architecture/response-screen-proposal.md):
compared eight responses at K1/K8 on Ring8/Product16. All32 focal trajectories
and4ordinary references completed20k updates and selection evaluation on
the same CPU host/backend. All12 independent confirmation cases are complete.
Training queue3761 and analysis watcher3767 have exited. Seed 0, pinned native
training/metrics and prebound evaluation arrays remain fixed. Training source
b5fe7aa and analysis0e78ce7 are preserved. Both streams independently
recompute within5.56e-17; all96 protected Report1/2 hashes remain unchanged.

**Next candidate: arctan-D K1 for a bounded JiT GPU readiness benchmark.**
Its Ring8 confirmation SW2 is0.125849 versus SwiGLU0.141952, winning all8banks.
Product16 is0.062764 versus SwiGLU0.039954, losing all8banks. Free ISRU also
beats every confirmed ODE variant on Product16 at the same parameter count.
K8 does not consistently improve quality over K1. Softsign-K8's strong Ring8
selection score does not pass the registered Product16 field-MSE eligibility
gate. No new image training is submitted.

[Complete comparison](docs/research-assessments/2026-11-15-response-screen/current-comparison.md),
[decision and analysis](docs/research-assessments/2026-11-15-response-screen/analysis-report.md),
[paired-bank analysis](docs/research-assessments/2026-11-15-response-screen/stats-appendix.md),
[completion evidence](docs/research-assessments/2026-11-15-response-screen/completion.json).
[Execution amendment](docs/research-assessments/2026-11-15-response-screen/protocol.md).
[Launch receipt](docs/research-assessments/2026-11-15-response-screen/launch-receipt.json)
and [benchmark](docs/research-assessments/2026-11-15-response-screen/benchmark.json).
Raw output root: `/Users/mr.z/ode-workspace-prep/response-screen-20261115/`.
No other user's GPU jobs were stopped;
the user explicitly revoked earlier yp2841 preemption permission.

## Completed order-by-response comparison,15 November

Four seed 0 K8 runs, common-bank evaluations and frozen-order diagnostics
completed on 145 GPUs 4–7. All queues exited zero and released their cards.
ISRU-K8 worsens Ring8 pooled SW2 from 0.13261 to 0.16779 relative to K1;
Product16 changes from 0.07206 to 0.07321. Tanh-K8 is the positive case:
Product16 improves from 0.07006 to 0.06427, with seven of eight banks better.
High-order terms are active, and the measured K8 exponential error is small.
Their necessity for K8-trained weights does not establish an ISRU advantage.
Ring8 C8 projection still largely removes mode imbalance at either order.

[Findings and figures](docs/research-assessments/2026-11-15-order-response/findings.md),
[completion](docs/research-assessments/2026-11-15-order-response/completion.json),
[metric verification](docs/research-assessments/2026-11-15-order-response/results/verification.json).
Runtime source is frozen at 9f72bd3. The subsequent planning discussion
prioritizes the bounded CelebA checkpoint diagnosis described below. The
rotation-training control remains an optional small synthetic check.
No new image run is submitted; finalized Reports1/2 are preserved.

## Completed professor follow-up,15November

Five seed0 trajectories, all15k/20k common-bank evaluations, and the independent
original-script SwiGLU audit completed on145 GPUs4-7. Monitors ended; no pending
jobs. Training runtime`da8b8a4` and separate analysis runtime`14c65a7` are preserved.

The main finding: Ring8 ISRU-D's symmetric error improves15k->20k while its
asymmetric error grows. C8projection reduces pooledSW2 from0.13261 to0.05268;
Free ISRU, Tanh-D and SwiGLU show similar sampling gains. Matched tanh controls
show no consistent ISRU-specific advantage. Product16 weight/update spectra
confirm that high-rank Muon updates need not yield higher-rank final weights.

[Findings, figures and next decision](docs/research-assessments/2026-11-15-professor-execution/findings.md),
[metric verification](docs/research-assessments/2026-11-15-professor-execution/results/verification.json),
and [completion evidence](docs/research-assessments/2026-11-15-professor-execution/completion.json).
Ring8 replays are separate from unavailable Lenovo checkpoints. The Muon control
has a documented cross-host discrepancy, despite exact old/new entry agreement
on145. All before/after C8 comparisons use the same weights. Reports1/2 remain final.

## Completed baseline/K1/K8 continuation

All three completed160epochs and FID50k, exit 0:0. Native SwiGLU leads at
every matching checkpoint.

| Epoch | Native SwiGLU | Wide K1 | Wide K8 |
|---|---:|---:|---:|
| 100 | 35.91785 | 37.92486 | 38.13574 |
| 120 | 29.47602 | 32.50213 | 32.85067 |
| 140 | 25.98559 | 29.62906 | 30.08220 |
| 160 | 24.30357 | 27.78868 | 28.08343 |

At160, K1 trails native by3.48511FID and K8 by3.77986. All three still improve
with added training. No experiment is extended beyond160.

[Matched curves and completed evidence](docs/celeba128-wide-20261114/continuation-160/matched-results.md).

The [next execution plan](docs/research-assessments/2026-11-14-next-architecture/execution-plan.md)
specifies a one-hour checkpoint diagnosis, numerical/readiness checks and two
previously proposed100epoch runs: frozen-D K2 versus refreshed-ISRU midpoint.
After review of the matched tanh results, that ISRU pair is retained as an
option rather than the default next experiment. It has not been implemented
or submitted. Keep seed0, original baseline and FID every20.

A later [professor-comment review](docs/research-assessments/2026-11-15-professor-comments.md)
recommends spectrum/C8-symmetry diagnostics and a matched activation control
before selecting that image pair. The authorized follow-up above now executes
this recommendation; the image pair remains unsubmitted.

Verified15November2026 from scheduler completion, progress, console and
TensorBoard. No new model code or finalized report was changed.

## Completed widened ISRU-D K8/K1

Both seed 0 runs completed 100 epochs,63,500 updates and all five FID50k
evaluations; both first-stage jobs exited 0:0. The continuation jobs are listed above.

| Model | Total parameters | Final FID50k |
|---|---:|---:|
| Fixed native SwiGLU |130,405,632 |**35.91785** |
| Widened K1 |130,402,560 |37.92486 |
| Widened K8 |130,402,560 |38.13574 |

Wide K8 improves over old small K8's43.90767 but still trails native by 2.21789.
K1 is .21088 below K8; the high-order expansion adds no final FID benefit here.
[FID curves and completed evidence](docs/celeba128-wide-20261114/results.md).

K8 job 23015406 used H100 gh011 for 6h08m11s; K1 job 23015408 used L40S gl068 for
7h15m28s, each including five FIDs. These different-GPU wall times are not a
speed ranking. Allocation-average GPU use was 83.92%/84.25% (K8/K1). All five
numbered full checkpoints and 101 P snapshots exist per arm. Final weights and
both EMAs are finite; all final P matrices match their trained checkpoints.

The persistent monitor 460860 recorded both COMPLETED states and exited
normally. Source remains frozen at 17b61a5; the qualified baseline and final
reports remain preserved. No additional experiment is submitted.

First 100-epoch completion was recorded earlier on 14 November 2026.

## Completed CelebA128 comparison

Both seed 0 native JiT-B/16 models finished 100 epochs and their FID50k
measurements. **Native SwiGLU wins this comparison: 35.91785 versus 43.90767.**
K8 FID is 7.98981 points higher (+22.24%), despite using 32.61% fewer parameters.
Its steady training epoch is 1.75 times as long. The earlier unexpanded K8 design has no
quality or training-time advantage at this 100-epoch budget.

| Measurement | Native SwiGLU | ISRU-D K8 |
|---|---:|---:|
| FID50k, checkpoint 100 | **35.91785** |43.90767 |
| Final training velocity loss |.02993375 |.03155256 |
| Full-model parameters |130,405,632 |87,879,936 |
| Training elapsed, L40S |3h12m30s |5h35m53s |
| Steady epoch time |113.73s |199.06s |
| Evaluation elapsed, A100 |34m47s |54m59s |

See [results, curves and examples](docs/celeba128-20261112/results.md).
The original result files and [comparison checks](docs/celeba128-20261112/FID100-comparison-verification.json)
verify exactly 50,000 PNGs per arm, matching evaluation settings, unchanged
checkpoint mtimes and agreement between original console and TensorBoard FID.

## Execution and fixed evaluation

Training jobs 22822267/22822268 and evaluation jobs 22993694/22993695 all
completed with exit 0:0. Training used distinct L40S GPUs on gl066; evaluation
used A100s on ga013/ga022. Both evaluation allocations have ended. Their
60-second monitor recorded both terminal results and exited normally.
No additional training is submitted for this comparison.

Evaluation uses the pinned standard torch-fidelity implementation, first
EMA .9999, seed 0, CFG0, author Heun50 with final Euler, 50,000 images and the
same 162,770 official CelebA training images with deterministic 128px crops.
The ordinary baseline meets the registered FID50k below 50 criterion.

All 101 K8 P snapshots are complete. Raw P mean spectral norm grows 1.1486 to 2.1631,
while stable rank drops 193.95 to 76.65. These describe learned P, without
identifying the cause of the FID gap.

## Evaluation support

The active code has [FID every 20 completed epochs](docs/celeba128-20261112/periodic-fid.md)
and complete numbered checkpoints 20/40/60/80/100. Regression checks verify
that evaluation preserves training RNG and parameter updates. Older 20/40/60/80
full checkpoints do not exist for the completed pair. New training budgets
include all five FID50k evaluations; provisional K8/native limits are 14h/8h.

- Development: `/scratch/zz5070/ODE_FFN/ODE-FFN`.
- Completed pair frozen source: `/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112`, commit 65864ad.
- Environment: `/scratch/zz5070/ODE_FFN/envs/jit-celeba-cu128`.
- Training outputs: `/scratch/zz5070/ODE_FFN/runs/celeba/20261112-b16-{ffn}-s0`.
- Evaluation outputs: `/scratch/zz5070/ODE_FFN/runs/celeba/evaluations/20261113-b16-{ffn}-s0-ckpt100-ema9999`.

Reports 1/2 and all raw training artifacts remain unchanged. Evidence is indexed
in [docs/README.md](docs/README.md); current research decisions are in [notes.md](notes.md).
