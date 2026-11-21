# Determine what the ODE structure contributes

## Reduced-state midpoint and solver ablations, 22 November

- [x] Implement and check three branches from the existing midpoint trainer.
- [x] Run frozen response, two-step training and reduced-state midpoint on Ring8 and Product16 on Mac mini.
- [x] Evaluate each 20k checkpoint on the fixed selection and confirmation banks.
- [x] Profile each variant and compare a second endpoint at matched training compute.
- [x] Publish the code, scores and architecture decision.

## CelebA128 midpoint transfer, 22 November

- [x] Bind native epoch/EMA/FID and checkpoint availability.
- [x] Implement and test midpoint training, paired FID and same-device timing.
- [x] Push and submit one monitored Torch trajectory through the 160-epoch comparison.

## Report 4 equal-compute study, 22 November

- [x] Profile loss, backward and each original optimizer. Bind the existing 20k midpoint budget.
- [x] Complete the four ordinary controls on Mac mini, seed 0, with progress and recovery.
- [x] Score the four endpoints against midpoint on the same confirmation arrays and publish.

## Completed Report 4 candidate training, 21 November

Train the eight-step tied residual FFN on Ring8 and Product16 on Mac mini.
Reuse the 20k midpoint recipe, seed 0, width and batch size. Compare native
held-out Heun32 SW2 on identical samples at 20k and publish the endpoints.
For the first architecture decision, prefer lower SW2 on both tasks.
This comparison matches steps and P calls. Equal total training cost is a
separate question.

Both training runs and the eight-bank Heun128 evaluations completed. [Results](docs/report4/tied-residual/README.md) favor midpoint over the new residual block and retain two-step midpoint inference.

## Completed: finalize Report3

The user finalized the current report on 18 November 2026. Preserve the exact
PDF, TeX, figures and results. Only status metadata and a checksum record change.

- [x] Record the finalization and move later work to Report4.
- [x] Bind the final files and verify the PDF is unchanged, without rebuilding.
- [x] Push to GitHub and fast-forward MBP; check the delivered final PDF.

Finalization6d5fa46is on GitHub and MBP. The PDF and finalization-record
SHA256values match across machines; both checkouts are clean. The20report
files and26bound evidence sources are recorded without rebuilding the PDF.

## Completed: improve Report3 readability

18 November follow-up: use natural academic prose, explain the exact compute
comparison and retain the professor's three-page text/four-page support layout.

- [x] Shorten abstract wording and explain each experiment before its result.
- [x] Specify fixed-width ordinary continuation to the existing state-flow
  training budget; keep widening and new response designs out of this test.
- [x] Rebuild, inspect all pages and verify unchanged evidence and Reports1/2.
- [x] Push and synchronize the revised report to MBP.

No new experiment is part of this writing pass.
Editing note: the initial patch had a stale line context and applied no changes;
the main-text section was then replaced using its exact section boundaries.
Verification passes: three main pages plus four supporting pages, unchanged
11pt font/margins,26source bindings and78table numeric cells. Changed pages
1/2/3/5/7visually inspected; pages4/6render identically to the inspected version.
No stray figure characters. Finalized reports and source data are unchanged.
Revisione55523eis pushed and synchronized to MBP. PDFhashes match, and the
remote checkout is clean. Report3remains a draft; no experiment was started.

## Completed: sharpen Report3 and bound the final research stage

18 November: this is the penultimate report. State supported findings directly,
make the remaining architectural hypothesis explicit, and avoid expanding the
experiment search. This pass revises the draft only; no training is launched.

- [x] Check the external commentary against the report, the sharing coordinate
  control and the earlier recurrence study; render supporting pages again.
- [x] Revise the opening conclusion and final two-week hypothesis, retaining
  all measured results and the three-page main-text limit.
- [x] Rebuild, verify source/protection bindings and inspect changed pages.
- [x] Push the revision and synchronize the source/PDF to MBP.

Findings: sharing is the leading structural explanation, not an isolated unique
cause; the frozen-D image result does not diagnose why it loses. No stray
characters appear in rendered pages4-7. Figure4's caption mistakenly calls its
horizontal score axis vertical; correct that wording. The older recurrence
study used scalar K2 and a different training protocol, so it is not the direct
control for the current state-flow model.

Revision verified: three main pages/four supporting pages, all78numeric cells
and26source bindings unchanged;97protected files and all finalized Report2
bindings pass. Changed pages1/3/7visually inspected. PDF420236bytes.
Revision961d424is on GitHub main and MBP. Remote/local PDFSHA256matches;
MBPworktree is clean and its finalized Report2PDFis unchanged. No new run.

## Current task: write a readable Report3, without further experiments

User explicitly requests the complete Report3and highlights the professor's
readability feedback. Write three pages of connected main text; put detailed
tables, figures and cases on clearly separated supporting pages. English,
Zhizhe Zhang, scores to three decimals. Reports1/2remain unchanged.

- [x] Bind the completed numerical results and choose only the evidence needed
  for the central sharing/coefficient/solver argument and professor responses.
- [x] Write the complete LaTeXreport with plain explanations, experiment purposes
  and observations, one Ring8case, and proposed next-two-week hypotheses.
- [x] Build PDF, check all pages visually/textually, audit numeric claims and
  verify protected report/native-source hashes. No new model evaluation.
- [x] Push the complete report/evidence and deliver PDF; copy to MBPif reachable.

Complete draft at24d6358: three main pages plus four supporting pages. All seven
pages visually inspected,26source bindings/78numeric cells checked, all fonts
embedded and no layout/reference warnings.97protected files,11Report2bindings
and5native image files unchanged. MBPfast-forwarded the full source and PDF;
remote/local SHA256matches. See reports/report3/delivery.json. No new experiment.

## Current request: professor-response audit and Report3 preparation

- [x] Check each professor question against direct existing evidence, including
  weight/update rank, Ring8C8symmetry and response replacements.
- [x] Read the now-completed four-case ordinary comparison and its independent
  verification; visually inspect both generated figures.
- [x] Publish the complete cached comparison and prepare a concise Report3
  outline with evidence sources and next-two-week decision questions.
- [x] Update current status, verify protected files and push. No new training,
  evaluation or145dispatch is authorized by this planning pass.

## Current requested wrap-up: cancel four GPU workers and review all experiments

- [x] Verify ownership/commands and terminate only the four recorded145GPU
  workers. All exit bySIGTERM; their CUDAcontexts are released. Preserve all
  completed and partial outputs. Dispatcher1477869stays stopped.
- [x] Consolidate the frozen127-entry historical ledger with the later Muon/PD,
  response, image and ODE-mechanism studies. Distinguish evidence families,
  actual training, frozen interventions, proofs, engineering and proposals.
- [x] Write an English evidence-linked project review and complete source
  inventory outside finalized reports. Give the user a Chinese scientific
  briefing with practical conclusions and remaining work.
- [x] Verify source links, numerical statements and protected report hashes;
  push the review and cancellation handoff. Do not launch new experiments.

Review verification passes:209source documents,162canonical analysis sources,
307local links,97protected files,11Report2bindings and5native image files.
Four145workers are canceled; two ordinary Maccases complete with independent
checks, two continue. The complete current-bank comparison is still pending.
The review introduces no model run, evaluation rerun or new research direction.

Goal paused at the user's request on 17 November, after closing the current
working changes. The objective remains to obtain a mathematically predicted,
falsifiable account of ODE-FFN behavior and decide which architecture merits
another image test. Resume autonomous work only on the user's instruction.
Existing registered compute queues and the fixed cached-analysis follower
remain running on Mac mini; no new experimental family is launched during
the pause. The user subsequently restricted synthetic work to Mac mini.
145dispatcher1477869is SIGSTOPpaused, blocking104unstarted jobs. The user then
canceled its four remaining workers; all exited bySIGTERM, and their CUDA
contexts are released. Do not resume145synthetic work. Preserve completed and
partial outputs; the canceled banks are not completed evidence.
Training seed0only. Preserve finalized Reports1/2and the native image baseline.

## P0 parallel: ordinary controls on the current target banks

- [x] Bind the four existing20kseed0FreeISRU/SwiGLUcontrols and current
  2026112700/2026112800streams. Older ordinary scores use different banks;
  do not rank them against the current ODEcohort. No retraining or retuning.
- [x] Reuse the pinned native loader/sampler/metric, preserve actual FP32
  nonpersistent time buffers on conversion toFP64, and replay every copied
  historical field/sample anchor. Validate both128/2048real entrypoints.
  Author15033actual0: all4old field/sample anchors replay exactly; all4models
  pass both meshes/streams and parameter/buffer/RNGchecks. Widest full4096bank
  Heun12814.454s, peak436.641MiB. Actual0/3CLI14090/1494and both NumPyverifiers
  actual0; maximum metric error4.45e-16. Fresh frozen pass is next.
- [x] Freeze documented source and obtain the fresh verbatim CPUcheck;
  launch at most two additional one-thread Macworkers after timing/memory.
  The original four all-source followers and145GPUqueue continue unchanged.
  Source29668aa/5262regular hashes and30input files unchanged; fresh exact
  five commands all actual0, no invocation divergence. Native anchors/recordings
  exact, independent error<=4.45e-16. Fresh full-bank benchmark14.113s,
  433.063MiB. Controller19767and workers19768/19769verified live at97-98%CPU;
  both started formal Ring8banks. Original controllers1477869/15486unchanged.
- [ ] Collect four controls/two streams/eight4096banks/both meshes, all raw
  endpoints and independent metric checks. Only complete common-bank groups
  enter architecture comparisons; no favorable checkpoint/bank selection.
- [ ] Compare both native/crossed ODEsolves with ordinary controls at2048,
  keeping the registered32paired contrasts and full cost/configuration labels.
  Exact design: ordinary-population-protocol.md. No automatic image promotion.
  - [x] Prepare a source-bound cached comparison that refuses incomplete
    cohorts, joins every model on identical banks, and retains all48model/
    stream/mesh rows and32primary contrasts. Reuse exhaustive bank-bootstrap
    arithmetic; report secondary exact sign tests with Holm correction.
    Actual statistics39905and incomplete-cohort audit58345exit0; the latter
    creates no output. Full joins/figures remain untested until all four
    ordinary cases complete. A separate recursive verifier covers all48source
    row joins and32primary contrasts, including the adjusted sign tests.
  - [ ] Run the postprocessor only after all ordinary producers/verifiers
    finish; inspect both generated figures and independently check joins/stats.
    Four-file analysis runtime5181693is frozen. Owner20622/tool67368tracks
    actual child20623, waiting on the original19767controller with live-command
    verification. No comparison output exists while the cohort is incomplete.
    After completion it creates both figures/all48rows and runs the recursive
    independent verifier, retaining the actual child exit in comparison-terminal.json.

## Completed causal chain

- Matched midpoint/RK4training improves hidden solve accuracy45.580x/22.245x
  without a corresponding generation gain: rk4-results/.
- Same-weight solver substitutions affect the complete field and can improve
  its error; simple own-solver compensation fails: field-exposure-results/.
- Measured field changes predict all8pooled distribution-effect directions,
  while particle predictions are tail-sensitive: target-transport-results/.
- All24fixed rare/median/probe roles pass small-perturbation derivative checks;
  extreme sensitivities vary with outer mesh: extreme-transport-results/.
- [x] Full-bank Heun127/128/129/512/2048complete:4producers+4independent exits0,
  all40rows/720metrics retained. Original128replay passes on every source.
- [x] Independent32-contrast statistics and both figures verified. ProductRK4
  midpoint benefit survives2048on BOTHstreams,8/8banks improve in each.
  See population-resolution-results/; no ordinary architecture ranking.

## Current P0: analytic defect predicts the complete learned field

- [x] Register learned-defect-field-protocol.md before learned observer outputs.
  Four frozen20kmodels/two existing4096-input streams; BOTHsolver defects,
  zero fitted scale, score against zero-change prediction on ProductRK4.
- [x] Reuse actual portable parameters AND nonpersistent time frequencies;
  copy existing x/time/oracle arrays by hash, no data draw or training.
- [x] Author CPU54925actual0on all8fixed16-input prefixes. Ring8qualifies;
  Product16fails only32->64tangent refinement. Preserve all original failures.
- [x] Register uniform128/256refinement before observations. Author11984actual0,
  all8refined numerical cells qualify; independent NumPyactual0,error3.88e-18.
  Original Product16flags remainfalse. Prefixes are numerical readiness only.
- [x] Source055a860/5175files: fresh CUDA/NumPy3852/20969actual0/0,
  all8refined cells/104reconstructed gates pass; no source changes.
  fieldsSHAa6c55e28, no invocation divergence. Receipt retained.
- [x] Full8x4096-input field predictions52369actual0on physicalGPU4;
  all refined numerical cells qualify, all original/refined flags retained.
  Independent7963actual0,error2.17e-18;5175source hashes unchanged.
- [x] Reconstruct field errors/risk accounting on all32768inputs; ProductRK4
  scores0.962/0.962,relativeRMS0.196/0.196. Both primary streams pass.
- [x] Publish the complete field evidence; both actual figures reviewed.
  learned-defect-field-results/ preserves all numerical flags and source audit.
- [x] Profile/compile the SAME256-step FP64observer per the registered runtime
  protocol before paying the projected132GPUhours for primary target transport.
  - [x] Installed Torch2.8.0+cu128/Triton3.4.0and gcc/g++present; no packages
    changed. nvccabsent; actual TorchInductorkernels validated below.
  - [x] Factor one coupled RK4step without editing the canonical observer.
    CPU66114actual0:8full-model prefix outputs/48nonzero-tangent layer checks
    match the original eagerly, bit for bit. Subsequent GPUchecks pass below.
  - [x] Original compiled author7526and fresh42278actual0. All8prefix cells
    pass; frozenba0c426/5200source hashes unchanged.
  - [x] Original full63139actual0: all8x4096fields pass, p<=6.67e-15,
    z<=1.25e-16. Independent saved-array reconstruction passes. Three paired
    resident timings show only1.240x; no graph breaks/fallback.

## P1: exact algebraic execution improvement

- [x] Follow analytic-drift-factorization.md: combine the same Pby linearity
  to reduce19to6matrix actions per coupled drift; verify all768prebound learned
  layer states locally, including both directions and zero forcing.
- [x] Check half-integer-power reuse separately; no changed real-valued equation.
  CPU80575actual0: both variants/all768states/four solver pairs pass;
  original/factored drift max8.33e-17, independent jets1.34e-15.
- [x] Publish/freeze a new runtime, authored and fresh documented CUDAchecks
  for both algebra variants; then full8x4096gates and three paired timings.
  ab42e7c/5213source hashes unchanged. Author43099/84757,fresh84411/64229,
  full75849/56124and independent91714/29494all actual0. Linear1.921x;
  power reuse4.475x,1.539s/B4096; select the latter with all equations fixed.
- [x] Only after identical numerical gates and measured full-batch speed gains
  may a factored implementation replace execution of the same target predictor.

## Subsequent causal link

- [x] Cached endpoint-geometry diagnosis while the original queues continue:
  reconstruct the target gradient, actual-displacement quadratic and exact
  sorting correction on every bank and both complete primary128 pools.
  Separate gradient-weighted tangent error from actual-endpoint nonlinearity;
  retain fixed top1/top10/top1% source-error contributions. Bind all input
  hashes, check independent AD and exact identities, add no model calls or
  samples. This is retrospective accounting, not the pending2048 decision.
  Controls44501/complete57725/independent16512actual0; all18units/720scalar
  comparisons agree<=3.34e-16, full-gradient AD<=1.44e-20. Rematching cancels
  98.841%/99.056%of actual quadratic displacement; sparse particle tails
  weigh differently in the target gradient. All original flags retained.

- [x] Register analytic-target-protocol.md before new target observations:
  both analytic/measured forcings, all models/banks,Heun128/2048and fixed gates.
- [x] Validate joint sampler kernel on analytic fixtures and all8learned
  original two-source prefixes. First CPU24257exit1after linear8/128/2048
  passes: optional outer JVPnests the analytic observer's existing forward AD,
  unsupported. Preserve source/failure; analytic learned checks use the
  registered independent two-epsilon finite differences, measured JVPretained.
  Corrected38126actual0: all8learned cases and linear8/128/2048checks pass.
- [x] Validate the real GPUbank entrypoint and its independent NumPyoutputs,
  freeze/publish and obtain the exact documented analytic-target execution.
  fd04994/5229hashes unchanged. Author83797/9229actual0/0; fresh four
  invocations30564/26506/6480/16263actual0, both numerical cases qualify.
- [x] Launch the monitored128bank-task queue on idle4-7: primary128then2048,
  followed by all original controls. All banks retained; no new training.
  Detached controller1477869; initial workers1477920-1477923physical4-7,
  primary selection banks0-3/Heun128. Actual live PID/CUDAcontexts verified.
- [ ] Collect all128bank tasks and128independent checks, pooled primary
 128/2048predictions and both-forcing diagnosis. A launch is not completion.
  Primary128all16banks+16independent checks now actual0. Prefix numerics15/16;
  confirmation bank0fails onlyepsilon1e-4target-metric FDfor BOTHforcings.
  Its hidden-reference/JVP/native gates pass; epsilon1e-5passes. Preserve the
  original failure and use the existing cached sorting/ADdiagnosis unchanged.
  First primary2048selection banks0-3and their independent checks now exit0;
  all four numerical qualifications pass, actual bank times6996.343--7200.482s.
  The next four workers continue selection banks4-7. No complete2048primary
  cohort exists yet; do not pool these first four into a quality decision.
  Read-only complete128pool audit21826/12141actual0: analytic scores0.966/
  0.758, measured0.994/0.870. Original one-bank numeric failure staysfalse.
  Cached rank audit98558actual0: independent AD<=5.43e-19; exact sorting
  correction reconstructs the failure, smaller four epsilons pass. No gate edit.
- [x] Use the registered OLDfour-lambda cached endpoints to separate analytic
  approximation error from finite-intervention error; no new model evaluation.
  29314actual0for both complete primary128groups: all endpoints/targets/directions
  replay; allfour prescribed lambdas retained. Scores atlambda.125are.998/.951,
  atlambda1.966/.758. Other model groups await their registered main outputs.
- [ ] Before final mechanism interpretation, run the pre-registered ALLsource
  reference control at the same16trajectory calls in every bank/mesh task.
  analytic-target-full-probe-control.md; main-prefix gates stay separate.
  - [x] Prepare identical CPU reference control while145cards run sampling:
    actual-path reconstruction, both-count GPUarray parity on both saved
    readiness cases, independent NumPy, memory/throughput and fresh doc pass.
    Authored72105/44072and both independent verifiers actual0; GPUparity
    <=6.67e-15. Full-batch timing72829actual0:12846.313s/25695.056s,
    peak584.828MiB. Four single-thread workers fit the measured memory budget.
    Fresh exact five commands actual0,5244hashes unchanged, no divergence;
    readinessf8afbb58. Source46a9f7a, allCPU/GPUparity gates pass.
  - [x] Run a bounded CPUfollow-up queue from independently completed banks,
    without touching the live frozen145runtime or restarting its tasks.
    Controller15486, workers15559-15562actual live/97-99%CPU, source46a9f7a.
    All16Heun128main banks cached; full128follow-up results remain pending.
  - [ ] Collect every full-source reference array and independent gate,
    preserve failures, and join both meshes/all models before final claims.
    First confirmation banks0-3and their NumPyverifiers now finish/qualify;
    controller15486continues the remaining original cohort.

Analytic field prediction now qualifies and the observed sampling benefit
persists at the prescribed meshes. Propagate that analytic forcing
through the actual learned sampler and target functional. The current measured-
delta transport is not this ODE-specific prediction. Do not skip that link or
claim ordinary-FFN superiority from a same-weight intervention.

## Owners and retained execution errors

Two atomic documentation patches missed literal context during cached-geometry
publication; no partial edit or experiment restart. Exact errors and corrected
actions are retained in target-aware-transport/ops/publication-errors.jsonl.

Population producer queue940668/tool29874and all8children terminal0.
Collector9883/tool86573terminal1because analysis source hashing passed str
instead of Path; frozen source and failed output/log preserved. Cached repair
41648actual0; layout-only50553actual0. No generated arrays or gates changed.
Read-only collector1SSHquoting failure/actual143also preserved; no computation
restarted. Canonical evidence: population-resolution-results/audit.json.
Mac authored learned prefixes54925/11984and NumPyverifier are terminal0.
All field actors are now terminal0; physicalGPU4 released. Actual256observer
median7.232s/B4096projects primary transport to131.653GPUhours. Preserve
reference/data/sampler fidelity. Original compilation only1.240x; registered
exact19-to6Pfactorization with power reuse passes all full fields and reaches
4.475x. Primary observer-only projection28.008GPUhours; measure actual sampler
throughput before treating that projection as a runtime promise.
Current raw roots: coefficient-20261116/target-aware-transport/
  population-resolution/ and learned-defect-fields/.

## Invariants and resources

Mac CPUwarm04f11f75, one thread/process, freely usable.145warm41450454,
only physical4-7when<500MiBand no compute PID;0-3prohibited. Never stop/share
foreign work. All four population contexts released at last live inventory;
recheck before any launch. No package changes or new training.
97protected files/11Report2bindings and five native image files unchanged.
CelebA160FID50k stays native24.304/K1 27.789/K8 28.083. Publish verified changes
promptly to Joni-z/ODE-FFNmain as Zhizhe Zhang; no force push.

Earlier complete journal: docs/history/2026-11-17-after-population-resolution/.
