# Cold clock and ordinary-coordinate controls

The [protocol](protocol.md) fixes two seed 0 cold training controls, one frozen
same-clock reference, seven new sample banks and unchanged ordinary comparisons.
Read the [coordinate derivation](theory.md) before interpreting P spectra or
calling equal initial functions equal Adam trajectories.

Both prescribed trajectories have completed 20000 updates. The [recorded progress](progress.md)
contains a timestamp and checked process identities; it is a saved snapshot.
The original controller is auditing the checkpoints before generation. No new
generation/FID result is available yet.

## Reproduction on Mac mini

Use the existing image-training@475428fb/image-metrics@dc729a50 environments
recorded in the local `.aris/compute/macmini.md`. The tracked
[training](../cycle17-cifar-baselines/training-env-spec.json) and
[metric](../cycle17-cifar-baselines/metrics-env-spec.json) specifications remain
unchanged. No package installation. Training uses two one-thread CPU workers;
feature extraction uses the existing MPS path. Do not use MBP or torch.

After a clean source commit, prepare a previously unused runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B docs/autoresearch/image-clock-training/training_controls.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clock-training-v2
```

A fresh agent must execute both commands verbatim, once each and in order;
if either fails, preserve the exact outputs and stop without fixing or retrying.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clock-training-v2/source/training_controls.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clock-training-v2
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clock-training-v2/source/training_controls.py metric-preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clock-training-v2
```

PASS requires four native 1024-prediction replays; shared cold parameters/buffers;
the nonzero active promoted output/gradient transformation; data-free mapped
Adam/clipping identities; actual forward/backward costs; both three-step
physical model/Adam/RNG/buffer resumes; 32 cached MPS feature rows, historical FID
and all 25 cached feature-bank headers. No new quality result is produced by
these witnesses. Equal initial zero-readout outputs alone are not a witness
of the active FFN identity.

Retain commands, actual exec sessions and terminal exits in `doc-validation/`.
Only after both terminal exits 0, write `documented-pass.json` with `status: PASS`,
`records` of command/exec_session/exit_code, `preflight_sha256`,
`metric_preflight_sha256` and `doc_vs_reality_mismatches: []`.

Launch the source-bound controller once:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clock-training-v2/source/controller.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clock-training-v2
```

Six phases: cold training, training-boundary audit, seven-bank sampling,
independent sample replay, features and independent feature/FID audit. Preserve
the actual controller tool exit, not just its status file. Observe a yielded
live handle; never restart on an observation timeout. Real PID/create-time and
primed five-second CPU/RSS appear in `orchestrator/resources.jsonl` and
`output/resources.jsonl`; per-arm status records actual completed updates.

The training audit checks all saved boundaries and native input digests. It
does not rerun all 40000 optimization steps. Every new prediction/sample/feature
bank is independently replayed. A finished training phase alone is not an
accepted result; preserve the full cohort, its audits and all prior failures.
