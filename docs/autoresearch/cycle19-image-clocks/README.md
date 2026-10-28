# Frozen image clock attribution

See the [registered protocol](protocol.md) and [exact map](theory.md).
This compares two constant-clock interventions with all four sealed native
controls. It is not retraining, a new ODE model or a rescue of prior gates.

Reuse the unchanged cycle17 training and metrics environments, already
validated by seeded kernels and independent documented-command passes. No
new package or environment invocation is introduced. Prepare a new runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python docs/autoresearch/cycle19-image-clocks/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle19-v2 --image-parent /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2 --resolution-parent /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle19-v2/source/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle19-v2
```

After committing the source and passing preflight, launch the frozen controller
once, capturing its terminal tool handle and log:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle19-v2/source/controller.py
```

It runs two one-thread CPU children, exact independent sampling replay,
unchanged MPS features and independent feature/FID checks. Commands, PIDs,
creation times, exits and five-second CPU/RSS telemetry remain in the runtime.
Observe the existing handle after timeouts. Never restart solely because a
monitor or SSH observation failed. Data, checkpoints, environments and logs
remain outside Git; submitted Report1 and sealed parents remain untouched.
