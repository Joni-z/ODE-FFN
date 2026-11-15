# Execution plan: frozen K2 versus state-dependent ISRU

15 November 2026. **Plan prepared; no new training authorized or launched in
this planning turn.** This makes the previous recommendation concrete. The
scope is one diagnostic pass and two new training trajectories, seed0 only.

**Priority update after the completed 15 November follow-up:** all three
professor-comment checks now have evidence, including the K1/K8 by ISRU/tanh
comparison. Prioritize M0 to locate the actual CelebA deficit. Ring8 rotation
augmentation is an optional small control and does not establish an image
architecture benefit. M1/M2 remain proposals for a reviewed run decision;
this update launches nothing. The synthetic K8 exponential probes do not
establish the same approximation accuracy on image-model activations or
demonstrate that refreshed feedback will help.

**Later15November priority review:** the new [professor-comment analysis](../2026-11-15-professor-comments.md)
recommends learned-spectrum and Ring8 C8 diagnostics, then a matched activation
control, before selecting this image pair. This plan remains preserved as a
later option; no implementation or training has been launched.

## Decision and completed reference

Test whether reevaluating the ISRU response during a local solve helps image
generation at the same parameter and matrix-work budget as frozen-D K2.
The practical reference remains the author's unchanged native SwiGLU.

All three existing runs completed160epochs with exit0:0. The final FID50k is
24.30357 for native,27.78868 for wideK1 and28.08343 for wideK8. Original console
and TensorBoard agree. [Full matched evidence](../../celeba128-wide-20261114/continuation-160/matched-results.md).
Do not extend these runs beyond160 as part of this plan.

Primary hypothesis: state-dependent ISRU improves FID relative to frozen K2.
Practical requirement: it must also beat native SwiGLU at the same epoch.
Equal parameter counts and P applications rule out those two budget increases
as explanations of the paired result. They do not equalize hidden functions
or optimizer trajectories at initialization.

## Exact two arms

Define q(z)=z/sqrt(1+1.2*z^2), d(z)=1/sqrt(1+1.2*z^2).
Every local solve shares one P across its two applications:

```text
Common:   u = P q(h);  mid = h + 0.5*u
R1 frozen:    out = h + P (d(h) * mid)
R2 refreshed: out = h + P q(mid)
```

R1 is exactly the current frozen-D Taylor K2. R2 is one explicit midpoint
step for y'=Pq(y), horizon1. Both retain the full returned state. R2 is
second order, not K8 or an exact exponential. Keep all gradients live.

The complete FFN remains Win(768->1200), solve1, SiLU, solve2,
Wout(1200->768), with independent P1/P2. Both arms have130,402,560 total
parameters and7,603,200 FFN forward dense MACs/token,1.61x native. P matrices
use the existing Kaiming initializer; Win/Wout use existing Xavier. Copy/check
identical initial tensors between arms and preserve native backbone RNG.

## M0: bounded checkpoint diagnosis

Maximum **one GPU-hour**, using existing immutable checkpoints100/140/160 and
first EMA.9999 for native/K1/K8. Do not train or alter stored checkpoints.

- Freeze1024 training image indices using seed0, deterministic128px crops,
  noise arrays and times(.1,.3,.5,.7,.9). Save the manifest and checksums.
  These are mechanism probes, not a new held-out performance claim.
- Record clean-image error and the existing JiT velocity-target error by
  time on identical inputs for all three models. Do not call this oracle
  conditional-velocity MSE: the population conditional mean is unknown.
- On K8, record relative magnitudes of terms2..8, d and q' quantiles, and
  ||Pq(h)|| relative to ||h|| across12blocks and both solves. Use streaming
  summaries and fixed token subsamples, not large activation dumps.
- Measure the local frozen/refreshed midpoint difference under saved weights.
  For bounded complete-network probes, change one block at a time at epoch160
  on a fixed128-image subset; record error changes rather than sampling new
  FIDs. Label these as inference interventions, not freshly trained models.
- Deliver one time-error plot and one layer/response plot, plus a short
  diagnosis. If the hour limit is reached, record missing measurements and
  stop the diagnostic. Do not add an unbounded Jacobian or literature study.

Diagnostics explain what changes. A favorable post-hoc checkpoint intervention
is **not** required to select the training arm: an untrained operator swap is
not a reliable test of its trained potential. This replaces the previous
qualitative "appreciable response" gate with a fixed, auditable experiment.

## M1: implementation and readiness

Only extend the FFN hook and the small option/scheduler plumbing that selects
it. Reuse author training, loss, sampling, EMA and standard FID. The intended
new selector is `isru_midpoint`; it is not implemented yet. Frozen K2 uses
the existing `isru_d` order2 path. No bespoke trainer is added.

Required checks before submission:

1. Float64 outputs and gradients of frozen K2 agree with its polynomial;
   refreshed outputs/gradients agree with an independent explicit midpoint
   expression. When d is constant, the two update rules coincide.
2. Parameter counts, initial shared tensors, backbone state and training RNG
   match; existing native/K1/K8 behavior is unchanged. Different initial
   hidden functions for R1/R2 are recorded, not concealed.
3. Batch256 forward/backward/optimizer/EMA and a brief sampler run are finite;
   save/resume and periodic-evaluation RNG regression checks pass.
4. Benchmark both variants and native on the **same GPU**, sequentially, with
   warmup excluded. A short readiness allocation is capped at30minutes.
   Discard readiness optimizer updates; formal runs start from original seed0.

Failures here block launch for a concrete repair; they do not authorize a
different activation, step size, width or optimizer. Freeze a new source
snapshot after checks. Push code and run protocol to GitHub before submission.

## M2: two formal runs, no architecture grid

| Setting | Fixed value |
|---|---|
| Dataset/model | CelebA official train162770; unconditional native JiT-B/16,128px |
| New runs | R1 frozen K2; R2 refreshed midpoint |
| Start | Both from scratch, same seed0; do not resume a trained K1/K8 |
| Width/composition | Internal1200, two P maps around SiLU |
| Training |100epochs,635updates/epoch, batch256 |
| Optimizer | AdamW5e-5, betas(.9,.95),wd0; warmup5 then constant |
| Other settings | Existing noise.5, logit time(-.8,.8), t_eps.05, EMA.9999/.9996 |
| Evaluation | FID50k at20/40/60/80/100; firstEMA.9999,seed0,CFG0,native Heun50 |
| Reference | Same deterministic162770-image128px reference and pinned torch-fidelity |
| Controls reused | Existing native/K1/K8; matched comparisons begin at100 |

No LR sweep, Muon, norm penalty, learned clock, normalization change or new
sampling recipe. EarlyFID200-plus points show progress only, not success.
Primary endpoint is100, not the best intermediate checkpoint. Do not stop
an arm for a disappointing20/40epoch FID. Stop for nonfinite training,
unrecoverable execution failure or the registered endpoint.

Retain full checkpoints every20epochs, per-epoch P snapshots, fixed sample
grids and GPU telemetry. A monitor polls job/progress/FID every60seconds and
GPU state every30seconds. Record and diagnose failures; do not silently restart.

## Resource contract

Torch only, existing environment, data and scratch layout. Each training arm
gets one GPU,6CPUs,4workers,24GiB; no automatic requeue. Use priority account
`torch_pr_637_tandon_priority` and constraint
`a100|h100|h200|l40s|rtx6000|b200`; verify effective scheduler routing before
submission. Run the two arms concurrently on separate cards when available.

Budget envelope before the new benchmark: approximately3–5 GPU-hours per arm
on H100,10–12 on L40S, including five FID evaluations. These are planning
estimates extrapolated from the existing jobs, not measured midpoint timings
or promises for every GPU type. Plan at most24 training GPU-hours for the
pair, plus1.5 diagnostic/readiness GPU-hours; queue time is excluded.

Choose each continuous allocation from measured end-to-end epoch time and
generation/metric/cleanup time:

```text
walltime = round_up_30min(1.2 *
  [100 * measured_epoch_time + 5 * measured_full_FID_time + startup])
```

The launch record must name the measurement GPU and assumptions for slower
eligible cards. If a full run cannot fit the12-hour per-arm planning cap,
return the revised budget before submission; do not silently request24hours,
change batch/epochs, or split into automatic resumptions. Save outputs under
`/scratch/zz5070/ODE_FFN/runs/celeba/<launch-date>-b16-isrud-w1200-<frozenk2|midpoint>-s0`.

## M3: fixed decisions at100 and160

Native100FID=35.91785; native160FID=24.30357.

| Outcome at100 | Decision |
|---|---|
| Both new arms >=35.91785 | End the two-arm family at100; no tuning rescue |
| Refreshed beats frozen but both lose to native | Record a local mechanism gain; do not promote it |
| Either new arm <35.91785 | Recommend continuing **both** to160 for a matched confirmation |
| Frozen wins and refreshed loses | Reject the feedback hypothesis; retain frozen only as a possible performance lead |

The two160epoch continuations are a later decision with a separate budget,
not an automatic action included in the first100epoch allocation. A tiny
100epoch lead permits checking persistence; it does not establish a finished
result. At160, refreshed must beat both frozen and native24.30357 to support
the proposed useful-feedback claim. If only frozen wins, the result supports
that design rather than feedback. If neither wins, stop this family.

Publish same-epoch FID curves, same-GPU train/sample throughput and memory,
fixed sample grids, and the two diagnostic plots. A quality gain must be
reported alongside its compute cost; this test alone does not establish a
general efficiency or ODE-specific novelty claim.

## Execution tracker

| Item | State |
|---|---|
| Existing three-arm160epoch result collection | COMPLETE |
| One-hour checkpoint diagnosis | PLANNED |
| New FFN selector, numerical checks and30min readiness | PLANNED |
| Frozen K2,seed0,100epochs | PLANNED, not submitted |
| Refreshed midpoint,seed0,100epochs | PLANNED, not submitted |
|160epoch confirmation | CONDITIONAL, not submitted |

Report1, Report2, prior runtimes, raw checkpoints and the native architecture
remain preserved. This plan adds no third candidate.
