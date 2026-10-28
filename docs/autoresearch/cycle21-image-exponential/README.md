# Frozen polynomial/exponential image diagnostic

Read the [question](protocol.md), [mathematics](theory.md) and
[evidence/literature review](review.md). No model is trained and no ODE
successor is selected by this diagnostic.

Reuse the existing, witnessed training/check/metrics environments and source
parents. Preserve every previous runtime. From the repository root:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python docs/autoresearch/cycle21-image-exponential/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle21-v2
```

After registration and preparation, execute this exact preflight once with
the existing image interpreter. It invokes the separate SciPy check interpreter
for matrix algebra, without importing the image model there. It verifies native
predictions and matrix fixtures, not new generation quality:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle21-v2/source/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle21-v2
```

Only after preflight passes and its receipt is committed, run once:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle21-v2/source/controller.py
```

The controller limits CPU sampling to two one-thread children, records
PIDs/create-times/exits/resources, and runs the independent audit and MPS
feature production after sampling. Follow the existing tool handle on
observation timeouts. The feature audit starts after both terminal exits.
Raw arrays, checkpoints and environments stay outside Git; Report1 and all
sealed parents remain untouched. Publish only audited, curated results.
