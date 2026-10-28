# Ordinary image SiLU controls

Completed: [analysis and decision](analysis-report.md), [statistics](stats-appendix.md),
and [figure catalog](figure-catalog.md). SiLU193 has lower fitting error and FID
than native K2 at equal FFN matrix work, with more parameters. No new ODE
architecture is nominated. The other fixed comparisons remain visible.

The [protocol](protocol.md) defines three cold ordinary baselines; the
[derivation](theory.md) fixes their widths from function shape and budgets.
No ODE, dataset, target, optimizer, seed or training length is changed.

Reuse the witnessed cycle17 training and metrics environments. No package,
environment spec or environment invocation changes. The new trainer entry
point gets an independent documented-command preflight before quality runs.
Prepare a new runtime, preserving all sealed parents:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python docs/autoresearch/cycle20-image-silu/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle20 --image-parent /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2 --resolution-parent /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18 --clock-parent /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle19-v2
```

Run the frozen preflight once. It writes software-check artifacts, not trained
quality evidence. Both the saved optimizer state and physical disk resume
must pass, along with shared backbone buffers and measured matrix costs:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle20/source/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle20
```

After source registration and preflight pass, run the controller once:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle20/source/controller.py
```

The queue allows at most two one-thread CPU training/sampling children. After
all three cells, independent CPU audit and MPS feature production can overlap;
feature audit starts only after both exit successfully. Commands, creation
times, PIDs, terminal exits and five-second CPU/RSS telemetry are saved.
Record the live tool handle and poll that handle on observation timeouts.
Do not relaunch merely because SSH or a monitor timed out.

All raw data, environments and checkpoints stay outside Git. Keep submitted
Report1 and previous negative studies untouched. Reused native and frozen-
clock endpoints are labeled explicitly; they are not new training runs.
