# Cycle14 execution record

The frozen source is `7d4cac9b188632dd69fa10c395f31f0655a4c2b0`, published to
GitHub main and checked out cleanly on both hosts. The existing native queue,
models, loss, sampler and optimizer source are unchanged. No new training runs.

## Scope and paths

- Lenovo: Ring8, five existing model/settings, 15 checkpoints, 240 fresh-bank
  evaluations. Source/output root: `D:/Research/ODE-FFN/professor-validation14-v2-20261104`.
- 145: Product16, the same comparison roles and 240 evaluations. Root:
  `/data2/zz5070/ODE_FFN/professor-validation14-v2-20261104`. GPU0 is occupied by
  another user at the last preflight; do not share or preempt it. Pending.
- CPU native-anchor attempt1 failed its registered roundoff guard before any
  formal evaluation. Original files and the published failure receipt remain.
  The CUDA v2 protocol retains bitwise coordinate equality and all cohort/bank
  settings. The wrapper now saves failed anchor arrays before raising and
  explicitly records all fresh-kernel dispatch flags.

The separate native queue output is `output/`, with monitor status/history under
`output/monitor/`. Two one-thread workers per eligible GPU. A queue completion
marker means a frozen diagnostic completed; it is not a new trained fit. The
training-trial ledger and both reports remain unchanged. Native timings are
concurrent execution observations, not isolated inference benchmarks.

## Progress

Fresh v2 documented readiness and actual launch records are appended below.
A partial Ring8 result must retain Product16 as pending; do not claim all 480
registered evaluations completed.

Lenovo v2 documented witness passed once: CUDA, eight kernel fixtures, fifteen
physical checkpoint bindings, five bitwise native anchors and fifteen explicit
finite fresh dispatches. The source stayed clean at7d4cac9. Formal native queue
launched with two workers; the detach wrapper reported PID21928 and the actual
monitor records controller PID29756, children24248/14584. First independent
post-SSH snapshot showed779MiB and91%GPU utilization. This verifies execution,
not model quality.145 remains pending for a free GPU.

Lenovo queue finished all5cells with exit0 and no unstarted cases. All240
formal evaluations and15field arrays independently replayed. Original25source
files rehashed unchanged after execution; clean source7d4cac9. Runtime sealed
with366files,19,946,866byte archive SHA256
64157c4179c887ba260a9e35d70a022192cafd3bee8790b63dbba03bf28ff0cb.
Primary results, post-outcome pooling, viewed figures and decision are in
[analysis](cycle14-results-ring8/analysis-report.md). Product16 is still pending.

The pending145 capacity watch ended after ten checks over ten minutes. The
specified GPU0 stayed occupied (last5,244MiB); no CUDA witness or formal
Product16 evaluation ran. The watcher is no longer active. Its supplementary
receipt is readiness/cycle14/145-capacity-wait.json. Recheck capacity before
resuming the still-pending documented invocation; no source repair is needed.

Ring8 findings published atca1e358. The clean MBP main checkout fast-forwarded
to that commit. Report2 PDF SHA256 remains
023a6339088d3231ee653a5aa82e2debf94a00f2a7e6d18c629593c5ee6759ea,
and TeX SHA256 remains
13c46c380294ec6931af40c09e7d64ab86b01614ce847702bfc437f0cbc3d3b2.
No report compilation, new training, image experiment or ledger append occurred.

4November continuation:145GPU0 became free. A fresh independent documented
CUDA witness passed once; five exact native anchors and all15physical bindings.
Native controller3151157 completed all fiveProduct16cells/240evaluations with
exit0. Full480evaluation independent replay passes, maximumSW2error5.56e-17;
all ten native anchors are exact. Complete findings: cycle14-results-complete.
145archive196,189,444bytes SHA256
a1eb6d73faad70ffcaba38798a57c7f1fd3532e1dc08fec04295989e92eeb1f3.
Original25files per host unchanged. Ring8-only snapshot/report preserve their
earlier pending status; the full cohort is now complete.
