# ODE-FFN autoresearch

This is a project-specific adaptation of [Karpathy autoresearch](https://github.com/karpathy/autoresearch), upstream revision `228791fb499afffb54b46200aca536f79142f117`. It uses its hypothesis/edit/run/measure/keep-or-discard cycle. It does not run upstream GPT pretraining or use bits per byte.

## Objective and authority
Develop a mathematically interpretable ODELayer successor that improves flow matching against strong ordinary controls. Follow repository AGENTS.md. The user has authorized continued experiments and normal GitHub publication; setup does not require another permission request. Preserve the researcher's decisions about future directions. Do not present a negative screen as a completed research objective.

## Immutable ground truth
Keep registered datasets, train/validation split, FM coupling, oracle formulas, metric code, native checkpoint restoration and held-out test data fixed. Model seed is 0. Stream identifiers remain the historically recorded values. Do not change evaluation or data to obtain a win. Treat source snapshots and sealed studies as read-only. Keep checkpoints, datasets, environments and raw logs outside Git.

## Experiment loop
1. Read task_plan.md, notes.md, the latest decision, and earlier stop rules. Reject a renamed rescue of an already failed setting.
2. Specify one hypothesis, the exact operator and full derivative, controls, actual parameter/inference/training cost, fixed budget, acceptance rule and failure interpretation before observing new outcomes.
3. Commit the protocol and runnable source on a dedicated branch. Publish reviewable milestones to canonical GitHub main using normal fast-forward history. Preserve every failed candidate and its evidence; do not hard-reset shared or published history.
4. Run preflight: exact native replay, buffers, initialization/stream pairing when relevant, derivatives, resume behavior when training, and immutable-file hashes.
5. Establish or replay unchanged baselines first. Execute the fixed screen with monitored processes. This project retains its historical 5,000-update training screen for scientific comparability; this explicitly replaces upstream's five-minute GPT budget. A read-only diagnostic has a registered cohort and finite wall-time ceiling.
6. Append every completed or crashed cycle to runtime results.tsv and publish a curated ledger. Record commit, metric, memory, keep/discard/crash status and description. Preserve failed source and raw logs. Never select the best seed, duration or checkpoint after looking at outcomes.
7. A lower old-ODE error alone is insufficient. Require comparisons against strong SwiGLU, ordinary matched inference/training computation, both target dependence laws, derivative errors and generated samples. Retain narrower improvements only with their exact limitations.
8. Update analysis, task_plan.md and notes.md, and push verified code/evidence promptly. Continue to the next justified hypothesis; do not generate a parameter grid merely to keep machines busy.

## Resources and monitoring
Use at most two one-thread CPU workers for these small Mac synthetic models, following prior measured performance. Check reachable GPU hosts before choosing a new execution platform; do not reserve a large GPU for negligible work. Poll the same live handle after observation expiry. Never infer completion from a stale lock file, nor restart merely because SSH or a monitor timed out.

## Evidence required for each cycle
Save input/source hashes, exact commands, process IDs and terminal exit codes, all compared outcomes, cost accounting, and a human-readable decision. Plot comparisons when they clarify the result. Distinguish mathematical identities, diagnostic observations and untested explanations. A kept experimental candidate is not automatically a publishable scientific claim.
