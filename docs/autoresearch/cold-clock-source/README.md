# Cold feature-clock / time-clock comparison

**Interpretation update (October 29, 2026):** the ordinary image baseline was not shown to be competent or converged. The scores below are historical fixed-budget observations and do not establish architecture effectiveness or justify rejecting a model family. See the [baseline-validity review](../baseline-validity-review.md). The sealed run and original report remain unchanged.

Two cold K1 arms start from exactly the same parameter state and FFN function.
The only architectural difference is the local clock input source. The
[complete analysis](results/analysis-report.md) reports FID **172.04** for the
time-only arm versus **174.20** for the paired feature control and **173.44**
for native K1. The 0.80% native-relative improvement misses the registered 5%
requirement; both fitting guards pass. FFN matrix work falls 49.60%, not whole-
model work. The setting stops without a rescue; no new ODE method is accepted.

All checkpoint/sample/feature checks pass. Seven inspected figures, all 41
endpoint rows, P/clock histories and fixed cases are in [results](results/).
See the unchanged [protocol](protocol.md), limited [derivation](derivation.md),
[analysis layout](analysis-layout.md), [task plan](task_plan.md) and
[primary-work comparison](closest-prior-work.md). The private runtime is sealed;
[progress.json](progress.json) records the closure and actual tool receipts.
The broader useful-ODE architecture objective remains open.

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
