# Time-preserving K1 clock check

Read the [registered hypothesis and controls](protocol.md). This study restores
the native 20k K1 and tests exactly one clock calendar plus its same-calibration
pooled control. It does not train a new model, search a calibration or reopen
K2. Private runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v2`.

## Reproduction

Mac mini, existing image-training@475428fb and image-metrics@dc729a50 from the
local ledger `.aris/compute/macmini.md`. The [training specification](../cycle17-cifar-baselines/training-env-spec.json)
and [metric specification](../cycle17-cifar-baselines/metrics-env-spec.json)
are unchanged; no installation or GPU reservation is required for sampling.

After a clean source commit, prepare a previously unused runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B docs/autoresearch/image-time-clocks/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v2
```

A fresh agent must run these documented commands verbatim, once each and in
order. If one fails, preserve its outputs and stop without a fix or retry.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v2/source/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v2
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v2/source/study.py metric-preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v2
```

Readiness requires native 1024 prediction/64-image endpoint bitwise replay,
six independently recaptured native clock arrays and 774/6 mean values,
constant-table injection identity, state/time derivative fixtures, ten
promoted literal/dense witnesses, measured 8192 FFN MACs, and 32 bitwise
historical MPS feature rows with unchanged native FID. No new quality result
is produced by these checks.

Retain exact commands, actual tool sessions and terminal exits in
`doc-validation/`. Only after both exit 0 and the receipts pass, write
`documented-pass.json` with `status: PASS`, command/session/exit records,
`preflight_sha256`, `metric_preflight_sha256` and
`doc_vs_reality_mismatches: []`. Import success alone is not readiness.

Launch once; the controller runs both new arms, independent sampling replay,
features and independent feature/FID audit in order:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v2/source/controller.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v2
```

At most two one-thread CPU sampling workers run together. The feature extractor
uses the existing local MPS environment. Monitor actual PID/create-time and
primed five-second CPU/RSS records in `orchestrator/resources.jsonl` and
`output/resources.jsonl`. Observe an existing live handle after a timeout;
do not launch a replacement merely because an observation expired. No result
is accepted until all four phases and both audits exit 0.
