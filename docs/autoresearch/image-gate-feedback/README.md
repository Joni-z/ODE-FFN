# Native gate feedback

This fixed checkpoint analysis measures complete derivatives and native clock
variation. It does not train or modify a model, calculate a new FID, or choose a
constraint from a local stability proxy. Read the [analysis](analysis-report.md), [protocol](protocol.md) and
[derivation package](../../../DERIVATION_PACKAGE.md). The fixed cohort and
independent native/AD audit are complete; no new quality result is claimed.

## Reproduction on Mac mini

Reuse the existing `image-training@475428fb` environment in the
[training specification](../cycle17-cifar-baselines/training-env-spec.json);
the provider ledger is local at `.aris/compute/macmini.md`. No installation is needed.
Only the Mac mini is used. No GPU is reserved.

After a clean source commit, prepare a previously unused runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B docs/autoresearch/image-gate-feedback/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gate-feedback
```

A fresh agent must execute this exact preflight command once, retain its exit
status and report documentation mismatches before formal analysis:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gate-feedback/source/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gate-feedback
```

Readiness requires eight bitwise native endpoint replays, actual promoted core
forwards, the state/P/held-clock derivative fixtures, rank identity, bounded-P
counterexample and Heun chain rule. Import success alone is insufficient.
Write `documented-pass.json` with `status: PASS`, actual command, consumed tool
session/exit code, `preflight_sha256` and `doc_vs_reality_mismatches: []` only
after the command exits0. Preserve a failed attempt and stop without retry.

Run the source-bound controller with at most two one-thread CPU workers. It
then invokes the independent manual-path/all-probe AD audit, retaining both
phase exits and primed five-second CPU/RSS telemetry:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gate-feedback/source/controller.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gate-feedback
```

Preserve terminal tool exits and prime/read CPU/RSS telemetry over actual
five-second windows. A yielded observation is not permission to restart a live
process. No formal result is accepted until the independent audit passes.
