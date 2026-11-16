# ODE mechanism investigation

Goal active: identify which ODE constraints help generative modeling and why.
Seed 0 only. Mac mini is available. On 145 use only empty physical GPUs 4-7;
never 0-3, never stop/share foreign jobs. Reports 1/2 and the native image
trainer/loss/sampler/baseline remain fixed. Push verified changes to main.

## Completed evidence

- Fixed/free exponential and midpoint coefficients, refreshed state response,
  sharing and mean-preserving stage controls, symmetry and qualified density:
  [primary findings](docs/research-assessments/2026-11-16-ode-mechanism/primary-findings.md).
- Matched P/Bcomposition: four 20k trajectories; all twelve common-bank rows;
  exact initial functions/data RNG; six-model native feature/gradient probes.
- Full registered internal 4/8/16 and outer 128/256/512 sweeps: eight tasks exit 0,
  twenty unique new cells plus four duplicate anchors. Every cell keeps32768
  sources. NumPy reconstruction error2.78e-17. Both actual scientific PNGs
  inspected; statistical corrections and figure meanings are included.
- CPU batching preserves all endpoints. New frozen 303d213 queue 56007 finished.
  Old55052and two duplicate workers were retired only after 10 full-prefix
  rows matched exactly before/after stabilizing owned processes. Actual -15
  exits and unstarted records are preserved. All these workers are absent.

## Findings that decide the next direction

- Palindrome's Ring8 lead over alternating survives every setting: internal 16
  SW2.084/.107. Product16 keeps the opposite effect:.087/.075; shared.069.
- Leading cancellation works, and higher-order stage features are active.
  It does not produce a task-general generation gain. Product16generation
  worsens with finer hidden integration at frozen trained weights.
- Ordinary Product16Free ISRU/SwiGLU remain stronger at the original Heun128
  setting(.055/.044). No new image promotion, response search or training.
- Hidden feature time still has no demonstrated correspondence with FM time.
  That missing connection is the next mathematical question.

## Next concrete actions

1. [x] Published numerical analysis/source at2ffd09c to GitHubmain. All97
   protected reports,11Report2bindings and ended-process checks pass.
2. Derive exact Gaussian-source FMvelocity/image-predictor Jacobian identities
   and check them against the existing Gaussian-mixture oracle. Distinguish
   full output-field constraints from a constraint on a hidden P matrix.
3. Review recent primary work on continuous feature depth, tied flow models
   and noise-time conditioning. Identify which proposed link is already known.
4. Use those derivations and frozen-checkpoint diagnostics to register a single
   matched, falsifiable architecture hypothesis if a concrete bottleneck remains.
   Keep the broader research goal active; completion of this sweep is a phase.

## Authoritative paths and runtime state

Source: /Users/mr.z/ODE-FFN-upstream; published training freeze 67b4664,
numerical freeze 303d213. All scientific runtimes stay frozen and clean.
Raw: /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order
New complete raw sweep: numerical-bankwise; independent: numerical-verified;
analysis: numerical-analysis-v2. Old intentional partial run: numerical-full.
Canonical: docs/research-assessments/2026-11-16-ode-mechanism/schedule-order/
  numerical-results/{analysis-report.md,stats-appendix.md,figure-catalog.md}.
All earlier execution journals are archived under docs/history/.
Current goal turn: progress, completed full numerical inference and audit.

## Target-geometry continuation

Previous goal turn: progress, full numerical sweep/statistics published.
P0now: derive the Bayes clean-data predictor and full velocity Jacobians for
Gaussian-source independent FM, then validate against the unchanged native
Ring8/Product16oracles and the author JiTloss conversion. Target derivation:
docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/derivation.md.
The invariant object is the full Bayes image predictor, not hidden P.

Use four identified primary papers only: Wan et al. ICML2025; Li/He JiTv2;
Graikos et al. September2026denoiser Jacobians; LiFT October2026. Local
papers/literature PDFfolders and optional Zotero/Obsidian tools are absent.
The arXiv helper is unresolved, so use the primary web records/full texts.
TiedFlow remains an unresolved name in prior notes; no claim relies on it.

Then register a frozen twelve-model diagnostic on existing selection pairs,
128points at five fixed times(.1,.3,.5,.7,.9). Primary explanatory metric:
full output-Jacobian error against oracle. Symmetry/PSDare secondary; include
analytic rotation and additive-bias controls to test their insufficiency.
No new architecture, training trajectory, image or density run is part of
this diagnostic. Preserve all scientific source freezes and finalized reports.

## Executable target-geometry phase

The preceding explanation turn was no progress on the research objective:
it verified and explained existing files but produced no new scientific evidence.
Resume with the next available safe action rather than another plan-only turn.

1. [x] Write the exact posterior/Jacobian derivation and native oracle check.
   Check both tasks, five interior times, nonunit-noise Gaussian and counterexamples.
2. [x] Complete two bounded read-only paper shards and record four-paper coverage.
3. [x] Register the twelve-checkpoint output-Jacobian diagnostic before execution;
   verify restoration, precision conversion and finite-difference derivatives.
4. [ ] Run the bounded frozen observation on the warm Mac CPU environment,
   independently recompute its summaries and publish the evidence to GitHub.
5. [ ] Decide whether full-Jacobian error explains both matched schedule effects;
   use the result to choose the next falsifiable mechanism test. No new training
   or image promotion is included in this phase. The full goal remains active.

Author checks: ten native oracle cells pass (velocity2.66e-14, Jacobian1.99e-13);
nonunit-noise posterior/control examples pass. Twelve real checkpoint derivative
witnesses pass; independent NumPy reconstruction1.11e-16. These four-point
witnesses cannot decide the128-point/five-time registered question. Freeze and
run the fresh exact documented pass before the formal observation.
