# Current ODE mechanism evidence

21 November. Report 4 compared the trained eight-step residual candidate with midpoint on Ring8 and Product16. Four midpoint steps in training and two in sampling gave the strongest ODE result on both tasks. Product16 still favors SwiGLU192. [Methods, checkpoints, evaluation and timings](docs/report4/tied-residual/README.md).

Report3 finalized by the user on 18 November 2026. Preserve the exact PDF,
TeX, figures and evidence under reports/report3/; finalization.json binds the
files. Historical draft status below no longer describes the report's current
status. Subsequent research belongs to Report4. No experiment is launched.

18 November readability pass: define sample banks, explain both equations and
replace abstract conclusions with the experiment's question and observation.
Section5 now fixes widths and continues ordinary controls to the existing20k
midpoint training-FLOP budget, including forward/backward/optimizer work and
the training already completed. The tied residual arm gets the same budget.
Widening, gating and linear bypasses are not part of this proposed test.
All scores and underlying evidence stay unchanged; no training is launched.

18 November Report3 revision: the user wants a clear claim and a bounded final
stage, with direct factual writing. Shared nonlinear iteration is the leading
architectural hypothesis; unique causality and a cause of image failure are not
established. The proposed comparison adds a direct eight-update tied residual
control with unit horizon to the current midpoint/ordinary cohort, matching
total training work on Ring8 and Product16. It tests midpoint staging beyond
shared recurrence; both computations admit ODE interpretations. Proposal only.
Re-rendered supporting pages show no stray characters. Correct the Figure4
caption's horizontal-axis description. Preserve all result values and Reports1/2.

Report3written17Novemberatuserrequest: complete EnglishLaTeX/PDFunder
reports/report3/, authorZhizheZhang. Three main pages plus four supporting
pages; tables/figures outside main narrative. Central sequence: Ring8symmetry
case -> sharing versus exact coefficients/solver fidelity -> analytic field
response -> common-bank and image performance. Two equations, seven tables,
four figures, and a next-two-week compute-matching hypothesis. Source numbers
come from26bound existing files; no training/sampling/model calls. All seven
pages visually reviewed; final numeric/protection check and delivery recorded
with the report. Report3remains a draft; Reports1/2are final and unchanged.

Report3preparation17November: the professor's three latest concrete requests
have direct completed responses: Product16weight/update rank and condition,
Ring8C8equivariance/mode imbalance with samples/trajectories, and matched
ISRUreplacement controls plus the eight-response screen. EarlierMuon/PDdesign
exploration is also documented. These answers do not establish a general
ODE-specific advantage.

The existing ordinary-control queue and cached follower now complete. All four
producers/independent checks pass;48comparison rows/32contrasts independently
reconstruct(max1.74e-18). Both figures viewed. Source-matched artifacts copied
without rerun to ordinary-population-results/. AtHeun2048confirmation native
state-flow/Free/SwiGLUpooledSW2is0.068/0.091/0.105Ring8and
0.047/0.051/0.034Product16. Different streams are not mixed. State flow uses
about4.1xFreeISRUdense forward MACs; no total-training-FLOPs claim.

reports/report3/outline.md now maps the three-page narrative and sources:
professor responses -> sharing versus solve fidelity -> practical generation
and image evidence. Proposed next-two-week question is whether Ring8's gain
survives compute-matched ordinary continuation. Proposal only; no new run.
Reports1/2remain final;145canceled dispatch remains stopped.

User requested closing the current work and stopping on17November. Goal is
paused, not complete. Preserve existing compute/follower processes and resume
agent work only when asked. The final cleanup publishes the cached-comparison
launch/progress record; no new model, training or experimental family is added.
Subsequent resource correction: synthetic work belongs on Mac mini. Verified
UID/command of145dispatcher1477869and SIGSTOPpaused only that process; stateTs
confirmed. The user then explicitly canceled the four remainingGPUworkers.
PIDs2120626/2130541/2138747/2139760all exited bySIGTERM(exit15), and no owned
CUDAcontext remains.104unstarted tasks stay blocked. Raw completed/partial
outputs and frozen source are preserved; canceled banks are not complete.
The new docs/research-assessments/2026-11-17-project-review/ indexes the exact
127legacy records,83later/supplementary records and162canonical analysis
sources. These are evidence records, not a count of independent training runs.
It summarizes operator design, Muon/rank, symmetry, response/order, sharing,
numerical mechanisms and matched CelebAwithout changing either final report.

Canonical directory: docs/research-assessments/2026-11-16-ode-mechanism/.
Image frozen-DTaylorK8and evolving-state synthetic ODEare different computations.

## Current-bank ordinary comparison

First ProductRK4selection2048banks0-3and all4independent children now exit0,
every prefix numerical qualification true. Runtime6996.343--7200.482s/bank;
next original workers continue banks4-7. Full primary pool is still pending.
All first four copied array/COMPLETE/independent hashes bind and reproduce.
The ordinary cached comparator now has a strict all-case terminal gate,
48source rows/32primary contrasts, fixed paired-bootstrap enumeration and
secondary exact sign/Holm checks. Statistics39905and partial audit58345exit0;
partial audit writes no comparison artifact. Full data joins/plots await all
four ordinary producers and verifiers; no partial quality report is issued.
Four-file cached analysis runtime5181693now runs under owner20622/tool67368,
child20623. It verifies original19767's live command while waiting; no output
directory exists. Once complete it writes the comparison and invokes the
separate recursive verifier, with actual terminal status retained. Frozen
source hashes and both figures must be checked before public result claims.

The old ordinary controls used earlier banks. The new ordinary-population
study reuses all4original20kseed0checkpoints on current2026112700/2800streams,
both128/2048meshes,8x4096banks. No retraining, tuning or new draws.
Author15033/check14090/1494/NumPyall actual0; old512field/64sample anchors
are exact for all4controls. Freeze29668aa/5262files; fresh exact five commands
actual0, all30input hashes unchanged, no invocation divergence. Fresh receipt
2a4ae2e9; widest4096bank/Heun12814.113s,433.063MiB. The same-checkpoint dense
MACcounts are bound separately, excluding pointwise work and cross-host timing.
Formal CPUcontroller19767and workers19768/19769are actual live97-98%CPU;
two extra one-thread workers beside the original4reference workers. All128
ordinary bank evaluations/16rows and independent checks remain required.
Primary practical family:32paired2048contrasts versus original ODEweights/
native-crossed solves. Never reuse older-bank scores as these outcomes.

## Current complete Heun128 diagnostic

Cached actual-endpoint geometry now completes all16banks and both pools,
without model calls. Independent dense reconstruction16512actual0checks720
scalars/18units,max3.33e-16; per-unit reverse-mode ADgradientmax1.44e-20.
Top10analytic residuals contain62.173%/95.167%of particle squared error but
12.274%/22.079%of absolute target-gradient contributions. Target rematching
cancels98.841%/99.056%of actual projected quadratic displacement. The exact
forecast-error split is gradient-weighted residual minus nonlinear remainder:
selection4.940e-6-8.053e-5=-7.559e-5; confirmation-7.325e-5-6.396e-5=
-1.372e-4. This is descriptive complete128accounting, not primary2048.
Raw endpoint-geometry-128.json, curated analytic-target-endpoint-geometry.json.
Original flags and all directions retained; no new model calls/samples.

CPUall-source confirmation banks0-3now finish and independently qualify;
actual runtimes4585.610--4652.162s. Controller15486continues banks4-7.
Original145controller1477869and four primary2048workers remain live.

Both ProductRK4streams retain8x4096banks and independent outputs. Analytic
unfitted target scores0.966/0.758; measured-forcing scores0.994/0.870.
Actual squared-loss changes-4.122e-4/-2.789e-4, analytic-4.878e-4/-4.161e-4.
Finite endpoint predictions remain inaccurate; scalar agreement is distinct.
Original bank numerical gates15/16. Confirmation bank0epsilon1e-4metric
failure remainsfalse; cached audit98558actual0finds AD<=5.43e-19and exact
sorting-error reconstruction. Four smaller epsilons pass, no gate was edited.

Cached continuation29314actual0reuses all old four-lambda arrays/targets/
directions after all-source endpoint replay. Analytic scores.998/.951at.125,
.966/.758at1. Confirmation finite-intervention error is1.006e-4and analytic
field-approximation contribution3.662e-5; their exact sum is1.372e-4.
These are retrospective128diagnostics; registered2048and full-source gates
remain open. No report or final architecture claim is produced from them.

Fresh CPUreference five commands actual0,46a9f7a/5244hashes unchanged;
saved GPUparity<=6.67e-15. Full-batch128/256CPUtimings46.313/95.056s,
RSS584.828MiB. Controller15486now follows completed banks with4one-thread
workers15559-15562; all16coarse banks cached, four actual workers verified
97-99%CPU. Original145controller1477869and four2048workers remain live.

## What is established

Better internal ODEaccuracy alone does not improve generation. Hidden numerical
bias reaches the complete field. At fixed weights, midpoint inference can
improve generation even where its field MSEis worse. The measured full-field
perturbation predicts all8pooled target-effect directions without fitting.
Rare particle sensitivities do not imply population sampler fragility.

Population-resolution-results/: all4weights/2streams/8x4096banks/5meshes,
40rows/720metrics/32new contrasts; all4producer+4NumPyexits0. Metric error
<=1.12e-16; recursive exact statistics agree0. Sourcea525eb2/5127files unchanged.
At2048ProductRK4selection SW2.053->.049,confirmation.048->.045,all16banks
improve. Adjusted squared-loss intervals[-5.279e-4,-3.162e-4]and
[-3.396e-4,-1.915e-4]. All40pooled comparisons favor midpoint; Ring8bank
intervals cross0. Same-weight effect; ordinary architecture ranking requires
matched fresh banks. Both figures inspected, all original failures preserved.

## Analytic field prediction now complete

learned-defect-field-results/: four frozen models/two full4,096-input streams.
BOTH modified-field defects pass through all six local flows and unchanged
network. No fitted scale or observed field change enters the predictor.
ProductRK4 scores0.962/0.962,relativeRMS0.196,alignment0.987; all8case scores
0.957-0.983. All5time bins have positive scores; low-noise-time errors larger.
All8oracle-risk signs predicted. ProductRK4field risk increases+2.335e-5/
+2.601e-5(actual),+2.418e-5/+2.356e-5(predicted), while generation improves.

Producer52369actual0in186.177s; NumPy7963actual0,error2.17e-18. All refined
numerical cases qualify. Only1/8original64-reference cases qualifies on the
full sets; all old flags remain recorded. Source055a860/5175hashes unchanged,
GPU4released. Both actual figures inspected; original/layout-only analyses
90318/56920actual0and numeric summaries identical. Final reports unchanged.

Original compiled observer now complete: all8x4096original fields pass;
maximum primal6.67e-15/tangent1.25e-16. Three paired timings give1.240x,
7.103s eager versus5.726s compiled median. Independent array reconstruction
passes, sourceba0c426/5200hashes unchanged. This is execution timing only.
Exact19-to6Pdrift factorization and separate power reuse now pass all768learned
CPUstates and all8x4096GPUfields. Fresh commands/full producers/independent
reconstructions all exit0; sourceab42e7c/5213hashes unchanged. Select power
reuse4.475x,1.539s/B4096; primary observer-only projection28.008GPUhours.
Both paired timing tables and compact receipts are in analytic-runtime-results/.

Analytic-target-protocol.md registers both analytic/measured sampler tangents
before target observations. CPU38126actual0passes all8real two-source prefixes
and analytic linear8/128/2048controls. Original24257nested-forward-ADfailure
and frozen checker remain. Sourcefd04994/5229frozen on145; authored real
ProductRK4bank witness83797and independent9229actual0, numerical qualification
true. Fresh exact Product/Ringentrypoints and NumPychecks all actual0;
readinessSHA05f9eff0. Source5229hashes unchanged, both numerical cases qualify.
Controller1477869now launches the monitored128task queue, physical4-7only;
initial1477920-1477923on primary selection banks0-3/Heun128. Sourcefd04994.
All-source16-call trajectory follow-up registered before formal observations;
main-prefix and full-source numerical qualification remain distinct.

The remaining ODE-specific link predicts target quality from this analytic
forcing through the ACTUAL learned sampler. Median eager256observer7.232s/
B4096projects primary full Heun2048transport to131.653GPUhours. Runtime
protocol fixes the same equations while reducing execution cost; FP64,256steps,
both defects,all banks and gates stay fixed. Small target-entrypoint witnesses
pass; coarse primary banks are complete and2048trajectories are running.
The final registered target-quality decision remains pending.

## Runtime and audit

Mac warm04f11f75and145warm41450454unchanged. Only empty physical1454-7;
0-3prohibited, no foreign task touched. Population queue940668/tool29874terminal0;
collector86573terminal1from source-hash Path type, cached analysis41648and50553
terminal0. All failed logs/frozen sources retained, no sampling rerun. Resource
polls show mean utilization56.733-60.420%on4cards; all contexts now released.
Final reports97files/11bindings and five native image files unchanged.

Raw: /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/.
145: /data2/zz5070/ODE_FFN/population-resolution-20261117/.
Earlier journal: docs/history/2026-11-17-after-population-resolution/.
