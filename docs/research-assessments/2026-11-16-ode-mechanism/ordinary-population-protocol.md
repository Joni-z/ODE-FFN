# Ordinary controls on the current target-transport banks

17 November 2026. Registered before ordinary-model outcomes on these banks.
The internal-solver effect is already known; these controls determine its
practical value relative to existing ordinary FFNs. This does not alter the
analytic-target prediction or either running queue.

## Fixed controls and inputs

Reuse the original 20,000-update, seed 0 checkpoints, without training, optimizer
changes or checkpoint selection:

| Task | Family | Original optimizer |
|---|---|---|
| Ring8 | Free ISRU | AdamW 0.0003 |
| Ring8 | SwiGLU64 | Muon 0.02 |
| Product16 | Free ISRU | AdamW 0.0003 |
| Product16 | SwiGLU192 | AdamW 0.0003 |

These are the ordinary controls retained in the existing sharing/RK4 reports.
Their older scores use different evaluation streams and cannot rank against
the current target-transport scores. Preserve configuration/capacity labels;
this is a practical comparison, not an isolated optimization-parameter test.

Checkpoint identity comes from the existing eight-case portable source:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry/portable-v2`.
Bind config, checkpoint and dataset-manifest hashes before model evaluation.
Use its pinned native loader and builder, replay the saved 512-input FP64 field
and 64-source Heun128 sample anchors exactly, and record every parameter/buffer
hash. Construct the model in native FP32 then convert the actual buffers to FP64;
do not reconstruct the nonpersistent time frequencies at a new precision.

Use the SAME sources, targets and 256 directions as the current four-model
population-resolution study. The selection/confirmation streams are
2026112700/2026112800. Bind its portable manifest and all four asset files;
check source/target/direction arrays against the original full field assets.
No new data draws. Each task/stream retains 8 banks of 4,096 sources and targets.

## Evaluation and verification

Reuse the pinned native Heun sampler at 128 and 2048 steps, FP64 and CPU one thread.
Do not implement a replacement integrator or metric. Keep all raw endpoints,
pooled and bank SW2, native mode/gap summaries, and measured runtime.
Field MSE uses all 4,096 existing matched-stream oracle inputs. Keep parameters,
all buffers and RNG unchanged during each evaluation.

Before formal dispatch, exercise every actual model and both sampler counts
on fixed small prefixes; those are execution checks, not quality evidence.
Replay the old field/sample anchors, check native128against an independently
recorded native path, and benchmark one full4096-source128-step bank to size
the run. A fresh agent must run the documented checks verbatim against frozen
source before the formal launch. Reuse warm environment 04f11f75; no installs.
At most two additional single-thread Macworkers accompany the four current
CPUreference workers.145 physical 0-3 remain prohibited; the active 4-7 queue is
untouched. Do not restart on an observation timeout.

An independent NumPyentrypoint reconstructs every pooled/bank distance and
source-bound mode/gap statistic from saved arrays. Completion requires all
four controls, both streams and meshes; preserve any failed producer/check.
Do not select a favorable subset or substitute a coarse result for 2048.

## Comparisons and decision

Primary practical contrasts at 2048 compare each of four ODE weight sets,
both existing streams and both native/crossed internal solves with both
ordinary controls on that task: 4x2x2x2 = 32 paired contrasts. Use all eight common
banks per contrast, record every squared-distance difference and the existing
multiple-contrast-adjusted paired-bank analysis. Pool all 32,768 sources
separately; a pooled score is not the mean bank score. 128 is the prescribed
coarse comparison and mesh check, not a second opportunity to choose a winner.
Bank variation concerns these fixed weights, not independent training runs.

The cached comparison retains all 48 model/stream/mesh rows. Alongside the
registered conditional bootstrap intervals, report the paired-bank mean,
median and sample SD of squared-distance differences. As a secondary check,
use the two-sided exact binomial sign test on nonzero bank differences and
Holm correction across the same 32 primary contrasts. This tests balanced
sign probabilities over independent banks at fixed weights; it is not a
randomized architecture assignment. Do not call a bootstrap interval a p-value.
No partial cohort may produce a comparison report or figure.

A useful image candidate must improve the practical quality/cost comparison
with ordinary models on both tasks and avoid resolved mode/gap regression.
No within-ODE win alone advances a model. If ordinary controls remain stronger,
retain the quantified numerical-bias mechanism but do not present it as a new
superior FFN. No new response-function search, penalty or image run is part
of this control.
