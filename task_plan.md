# ODE mechanism investigation

Goal active: identify which ODE constraints help generative modeling and why.
Keep seed 0 and report scores to three decimals. Reports1/2 are final. Preserve
the pinned image trainer, loss, sampler, baseline and frozen running sources.
Mac mini is available; on145 use only empty physical GPUs4-7, never0-3.
Never stop/share foreign jobs. Re-poll actual handles after observation timeouts.

## Evidence already complete

- Fixed/free exponential and midpoint coefficients, state feedback, ordinary
  Free ISRU/SwiGLU controls: [primary findings](docs/research-assessments/2026-11-16-ode-mechanism/primary-findings.md).
- Shared/anchored stage training, full-bank solver/symmetry interventions,
  original16-cell density comparison. Product16 rejects lower joint KL as
  the explanation of the shared-P SW2 lead; Ring8 paired KL is unresolved.
- Mean-preserving stage training, same-function optimizer probes and native
  stage-order decomposition. Coordinate choice affects updates. The bracket
  expansion holds at small horizon, but unit-horizon errors remain material.
- New full-bank inner/outer numerical sweep; anchored/new Ring8 point ordering
  reverses under inner refinement. No image promotion follows these scores.
- New Ring8 density/adaptive observations: all1,024sources in both fields
  qualify independently within5.33e-15. Queue48821/worker48824 exited0.

## P0: finish the registered new-control comparison

1. [x] Product16formal queue641894/worker641939completed on145physical7,
   all16rows, exit0. Frozenfd7f09b source stayed clean; processes are absent.
2. [x] Copy the entire case to contrast-observations/formal-product16-cuda.
   Independent audit of all32new formal rows agrees within4.27e-14. All9
   retained CPUrows match GPUarrays within1.39e-12; no host/point splicing.
3. [x] After that full audit, recheck fallback UID/parent/start/full command.
   Only our CPU43094was terminated; actual exit-15and process absent. The
   failed CPUqueue receipt is preserved, with no unrelated failure. Source,
   model, data and gates stayed unchanged. Receipt:migration-product16-complete.json.
4. [x] Product16 CPUrefinement49633/49635finished exit0. All1,024sources
   in each field qualify: original953at2048/71at4096; projected965/59.
   Both population processes are absent. No point was dropped or gate relaxed.
5. [x] Independently verify the Product16merge, then both new cases together
   using verify_adaptive.py. Require two cases/four qualified fields. Retain
   source indices, previous estimates and selected resolutions. No density
   ranking for an unresolved field.
6. [x] Run prepared analyze_formal.py with the new two-case verification and
   preserved original eight-case reference. Require identical dataset NPZbytes,
   all20cells and all8registered paired contrasts,10,000within-bank resamples
   RNG0,99.375%nominal intervals. No confirmation time-profile export.
7. [x] Run prepared product_geometry.py on the new control; compare old/new
   conditional widths, centroids and dependence with the same target calibration.
   Render plot_formal.py's two scientific figures and inspect actual images.
   Keep small1024-source SW2 apart from primary32768-source scores.
8. [x] Publish an evidence-bound analysis bundle (analysis-report.md,
   stats-appendix.md, figure-catalog.md, actual figures), canonical findings,
   STATUS/notes and source under Zhizhe Zhang to GitHubmain. Independently
   check key means/paired intervals and all finalized-report hash bindings.

## P1: answer the mechanism question and choose the next intervention

- Interpret each hypothesis as keep/weaken/reject using the completed controls.
  Do not equate layer activity, ODE approximation accuracy or a lower SW2 with
  a general distribution improvement or an ODE-exclusive contribution.
- If a specific structural mechanism remains supported, register one matched,
  mathematically predicted and falsifiable intervention before new training.
  Otherwise report that current evidence favors generic feature/optimization
  effects; do not continue response-only search or promote a new image model.
- Preserve the full objective through analysis and the resulting research
  decision. Completion of one queue is not completion of the goal.

## Resource observations and completed recovery

The actual71/59-source CPUbatched backend was slower on the projected field;
kept coordinate CPUexecution. CUDAwas3.521xfaster in the bounded benchmark,
but the card was reclaimed before a population child started. A too-early
CPUhold cost240.9seconds; its exact in-memory state was recovered, no restart.
The corrected905868queue waited while CPUcontinued, then was cancelled after
CPUcompletion; it never started a population child. All raw attempt/identity,
source and recovery records are preserved. Never suspend CPUfor a pending
GPUallocation. No foreign job was stopped/shared; physical0-3were not used.

## Matched order-control execution

1. [x] Implement the two same-P/B schedules, one new zero64x64 B per local solve.
   Parameters are81,282/83,088, with the same eight matrix applications per solve.
2. [x] Native math/common-initialization/gradient, matrix-call and exact native
   continuation/evaluator checks pass. Independent NumPy error2.22e-16.
3. [x] Fresh documented agent exits0/0/0; all four initial checkpoints match
   shared references and data RNG. Eight old references and arrays are bound.
4. [x] Publish/freeze67b4664, then launch four20k seed0trajectories on Mac CPU,
   two workers. Preserve the registered midpoint2, AdamW0.0003, batch512 recipe.
5. [x] Native final-test replay;15k/20k selection and20k confirmation on all
   eight common4096-source banks. Independently reconstruct twelve rows.
6. [in progress] Frozen inner4/8/16 and outer128/256/512 checks, B/bracket activity and
   cost; accept or reject the registered composition-benefit claim.

The previous explanation turn made no experimental progress. This turn
implements and verifies the registered control. B gradients differ by design;
no normalization or extra learning rate is introduced. The proposal is now
implemented and validated. Queue52859and watcher52860are live; all four20k
trajectories and all common-bank evaluations finished, exit0. All241frozen study source
files match actual invocation hashes. Frozen source and all97report files
remain unchanged.
[Exact invocation and scope](docs/research-assessments/2026-11-16-ode-mechanism/schedule-order/execution.md).

## Execution bindings

Local raw root:
/Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations
Frozen observer/checkpoint loader: fd7f09b0064ceab36a13f7757fd2716d002cbba6.
Data SHA:af2e0bd2ae01c76aef97e8ce5812d8234602f140123d0547575686f194e02577.
Reference8verification: ../density-transport/adaptive-full-verified.
Reference data: ../density-transport/portable. New data: density-data.
CUDAreadiness:145doc-cuda-2; adaptive:145adaptive-doc-cuda-product16.
CPUidentity/fallback and speed/portability: migration-product16-{start,live}.json,
routing-benchmark.json and handoff-prefix-check/COMPLETE.json.
Canonical evidence: docs/research-assessments/2026-11-16-ode-mechanism/coordinate-control/.

Previous detailed execution journal is preserved verbatim at
[history](docs/history/2026-11-16-before-coordinate-density-completion-task-plan.md).
Current goal turn: progress. All observation jobs are terminal; twenty-cell
analysis, independent statistics and three inspected figures are complete.
Publication at f145682 is verified on GitHubmain. The matched order control is implemented and independently validated;
source is published and four formal trajectories are active/queued.

Observer support: portable sampling-only export, six-model native stage-action
probe, generic cost control and numerical loader now accept the two schedules.
All four actual step16checkpoints restore exactly through that loader. Fresh
full-field/bank0portable export, native geometry and cost entries all pass.
See schedule-order/observer-execution.md; no density run is launched.

All four20k trajectories and all twelve common-bank rows are complete.
Ring8palindrome/alternatingSW2.086/.108; Product16.072/.066. Opposite primary
effects reject a task-general coarse-resolution benefit. Both tasks and both
schedules remain in the registered numerical sweep. Fresh four-case actual
CPUentry witnesses pass, with sixteen NumPyscores within5.55e-17.
Observerfcbe347is published/frozen. Numericalqueue55052and workers55054/55055
are live on Mac CPU, two workers/eight full-bank tasks. Full results and
verify_numerical.py collection remain pending; do not duplicate the queue.
