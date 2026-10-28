# Invariant-clock feature flow

This is one prospective FFN constraint, not a validated improvement. The
[question](research-question-card.md), [primary prior art](literature-review.md)
and [derivation](theory.md) distinguish an exact hidden feature flow from
outer FM and from a Gaussian posterior. Existing action-angle methods already
contain the rotation construction; no mathematical novelty is claimed.

## Outcome (2026-10-29)

The fixed screen failed. Rotation field MSE is 0.2948 / 0.2550 / 0.5101 on
Product16 / sign-code / half-mixture, compared with Swi129 0.07729 / 0.04097 /
0.11923 at nearly identical parameter and dense-work budgets. Euler and
normalized Euler remain close to rotation. This setting is stopped; no image
training, learned-frame rescue or changed training budget is selected.

Read the [complete analysis](results/analysis-report.md),
[case notes](results/case-notes.md), [all endpoints](results/endpoints.csv),
[P trajectories](results/P-trajectories.csv) and [figures](results/figure-catalog.md).
Actual controller60305/d97c5d and curator1895/7c3cb7 exited0; the independent
audit passed. Raw source/states/outputs are sealed outside Git; the archive
receipt records the seal. The original research objective remains open.

## Operator eligibility

After freezing this committed source outside Git, run the data-free command
in the unchanged warm Mac mini .venv-test environment, with one CPU thread:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-invariant-clock-operator/source/verify_operator.py --output /Users/mr.z/ode-workspace-prep/autoresearch/oct28-invariant-clock-operator/fixtures.json
```

This checks literal input/parameter derivatives, full feedback gain,
independent integration, active routes, counterexamples, initial identity
pairing and actual width64 matrix work. Preserve actual exit/log/source and
every failure. Passing it does not establish native model integration,
checkpoint/resume, full experiment readiness or image-quality improvement.

## Registered training screen

The fixed [protocol](protocol.md) compares one candidate with six controls on
three unchanged 16D synthetic laws: 21 new trajectories and 15 reused historical
endpoints. The warm environment is pinned in
[synthetic-env-spec.json](synthetic-env-spec.json). These small CPU models use
two workers with one thread each on Mac mini; no GPU is reserved.

Freeze the committed source and immutable input bindings once into a new runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/invariant-clock-flow/twist_study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-invariant-clock-v1
```

Before formal training, a fresh agent must execute this exact native preflight
once, preserving its actual tool exit and log. It must not fix files or retry:

```bash
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-invariant-clock-v1/source/twist_study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-invariant-clock-v1
```

Expected sentinel: `NATIVE_INVARIANT_CLOCK_PREFLIGHT_PASS`. The command checks
actual whole-model matrix work, paired initial functions and gradients using
a nonzero head, nonzero-frequency clock routes, alternative explicit-chain
network Jacobians, physical three-update checkpoint/resume for all 21 cells,
and bitwise native predictions for all 15 old endpoints. It writes
`preflight.json` only after success. Those short update witnesses are discarded
readiness probes; formal cells start from fresh cold models.

Only an actual exit0 with no documentation mismatch permits the parent to
write `documented-pass.json`, binding that preflight's SHA256. The controller
checks this receipt and then runs the registered training and independent audit:

```bash
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-invariant-clock-v1/source/twist_study.py run --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-invariant-clock-v1
```

`output/status.json` and per-cell status report training progress.
`orchestrator/resources.jsonl` records primed process CPU use and RSS every
five seconds. The automatic audit replays checkpoint predictions, full native
Jacobian panels and Heun samples, with explicit-chain Jacobian anchors and
stored P matrices. A controller status file alone is insufficient: retain the
actual terminal tool exit before interpreting results. Keep any failed runtime
unchanged and create a new version after fixing its cause.
