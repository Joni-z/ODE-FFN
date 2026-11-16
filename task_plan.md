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

1. Publish the completed numerical analysis and source, then verify GitHubmain,
   protected report hashes and absence of our ended observation processes.
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
