# Cold feature-clock / time-clock comparison

Two cold K1 arms start from exactly the same parameter state and FFN function.
The only architectural difference is whether the local clock consumes image
features or the existing objective-time conditioner. See the
[protocol](protocol.md), [limited derivation](derivation.md) and
[task plan](task_plan.md). Both 20k trajectories and the training audit are
complete on Mac mini; sampling is running. [progress.json](progress.json)
is a published snapshot. Training uses one CPU thread per arm and model seed0.
No new FID or quality result exists yet. A [bounded primary-work check](closest-prior-work.md)
identifies direct precedents for timestep-conditioned weights; the current
scalar matrix family must earn its value empirically.

Use unchanged Mac mini environments image-training@475428fb and
image-metrics@dc729a50. Commit source, then prepare a new exclusive runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/cold-clock-source/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1
```

A fresh agent runs each following command verbatim, exactly once, without
source inspection, fixes, retries or substitute invocations. Keep combined
stdout/stderr and actual initial/terminal tool results.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1/source/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1
```

Expected sentinel: `COLD_CLOCK_SOURCE_NATIVE_PASS`. Checks include paired
initial states and active shared/P gradients, the full conditional-matrix
parameter derivative, finite differences through the actual shared conditioner,
unchanged native prediction prefixes, physical five-update resumes, independent
active update replay and measured forward/backward matrix work at 64 patches.
Inherited active probes are not formal training starts.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1/source/study.py metric-preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1
```

Expected sentinel: `COLD_CLOCK_SOURCE_METRIC_PASS`. Replays 32 cached native
Inception rows bitwise and the native K1 FID within 1e-8, using the original
cached weights. No install or weight download. After both actual exit-zero
passes, the parent binds hashes and real tool receipts in documented-pass.json.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1/source/study.py run --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-cold-clock-source-v1
```

The controller starts two owned one-thread CPU training children, then performs
training audit, sampling, independent sampling audit, metrics and metric audit
sequentially. It logs actual commands/PIDs/exits and primed CPU/RSS every five
seconds. Per-arm progress is saved every 100 updates. Poll the same actual
handle through observation timeouts; never restart an existing live job.
All full states, physical P/clock histories and six raw sample banks remain
outside Git. The 20k endpoint is primary; all 5k/10k rows and all 35 historical
rows remain in the final comparison. No failed calendar or initializer is
reopened. Original source, sealed studies and submitted Report1 are unchanged.
