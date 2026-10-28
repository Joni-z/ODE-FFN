# Matched-function SwiGLU initialization

One ordinary baseline control, not a new ODE architecture. Keep native Swi86,
all 211,412 trainable values and its 16,512 dense FFN forward MACs per token
per block. Only its initialization changes. See the [protocol](protocol.md)
for the exact function embedding, retained controls and stopping rule.

Use the unchanged Mac mini training environment `475428fb` and metrics
environment `dc729a50`. No installation, CUDA device or remote host is needed.
Commit the code and use a new exclusive runtime; never overwrite a failed run.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/image-swiglu-initialization/init_study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1
```

A fresh agent runs each of the following commands exactly once, without source
edits, retries or a substitute invocation. Preserve combined stdout/stderr,
the actual terminal tool results and any discrepancy from these expectations.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1/source/init_study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1
```

Expected sentinel: `SWIGLU_INIT_NATIVE_PREFLIGHT_PASS`. This checks literal
initial FFNs, input and main-route derivatives, active-backbone shared
gradients, opening of extra routes, four unchanged native prediction prefixes,
five-update physical Adam/RNG/buffer resume and unchanged parameter/MAC counts.
The active-backbone witness is not a formal training start.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1/source/init_study.py metric-preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1
```

Expected sentinel: `SWIGLU_INIT_METRIC_PREFLIGHT_PASS`. This replays 32 cached
MPS Inception feature rows bitwise and the native K1 FID within 1e-8, using the
original cached weight and references. It does not install or download weights.

After both real exit-zero/no-mismatch results, the parent writes
`documented-pass.json` binding both preflight SHA256 hashes and both actual
command/terminal receipts. The controller verifies these before launch.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1/source/init_study.py run --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-swiglu-init-v1
```

The controller launches one one-thread CPU child at a time: training,
training audit, Heun sampling, sampling audit, metrics, metric audit. It saves
actual commands/PIDs/exits and primed CPU/RSS every five seconds. Training
reports every 100 updates. Poll this existing process; an observation timeout
does not authorize a restart. Phase budgets are in the registered protocol.

Training saves seven full states, six preceding-update states, all prediction
panels and gate/value/output matrix histories. The audit reconstructs every
input draw and checks every stored final update. Sampling and feature audits
replay all three complete 1,024-image banks, without modifying formal arrays.
The metric audit also replays all 2,048 reference features. The original
32 historical outcomes remain in the final table. A successful pipeline alone
is not a positive quality conclusion; read the final FID and both loss checks.
Original source, data, checkpoints, sealed studies and submitted Report1 remain
unchanged. Raw artifacts and execution logs stay outside Git.
