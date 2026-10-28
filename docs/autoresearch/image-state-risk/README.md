# Shared-state image-risk diagnostic

This study tests one missing empirical premise before another ODELayer design.
It does not train a model, alter a sampler, calculate a new FID or establish a
population oracle. Read the fixed [protocol](protocol.md) and [source scope](notes.md).

Warm-reuse image-training@475428fb in the existing Mac mini environment. No
package installation, CUDA or GPU reservation is needed. Freeze committed
source and immutable parent/data/state bindings into a new exclusive runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/image-state-risk/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-state-risk-v1
```

A fresh agent must run the following exact preflight once, without fixing or
retrying, and retain the actual combined output and terminal exit metadata:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-state-risk-v1/source/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-state-risk-v1
```

Expected sentinel: `STATE_RISK_NATIVE_PREFLIGHT_PASS`. This checks actual
five-model native64-image predictions bitwise against the stored20k validation
prefixes, full45,000-image training split identity/normalization, fixed state
shapes, direct-likelihood conditional moments, complete reference derivatives
and the conditional-risk identity. It writes preflight.json after success.
The parent may then bind the actual exit0/no-mismatch record and preflight
SHA256 in documented-pass.json. No import-only readiness claim is sufficient.

Only after that pass, execute the registered read-only cohort and audit:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-state-risk-v1/source/study.py run --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-image-state-risk-v1
```

The controller saves actual child commands/identities/exits, CPU/RSS every
five seconds, and per-domain progress. At most one one-thread CPU child runs
at a time. A complete analysis requires both phase exits0, every independent
replay, and the actual consumed terminal controller exit. Preserve any failed
runtime and create a new version after correcting its cause.

Raw predictions, means, support statistics and logs remain outside Git. Publish
all100model/time/domain rows, their paired-reference validity checks, figures
and exact decision. No positive architecture conclusion follows from this
diagnostic alone; the original research objective remains open.
