# Frozen image sampler resolution

Completed: all four arms, eight new sample banks and all 12 feature/FID
checks passed independent replay. The ODE FID lead persists but shrinks;
cycle17 fitting/order failures remain unchanged. See the
[analysis and figures](analysis-report.md), [statistical scope](stats-appendix.md)
and [decision](results/decision.json). Commands below document this completed
study; do not rerun them into its runtime.

The [protocol](protocol.md) and [derivation](theory.md) test whether the cycle17
FID ordering depends on its finite Heun mesh. This is read-only with respect
to trained models. It cannot rescue the failed fitting/order gates or nominate
a new method. All four models, all 1,024 starts and all three grids are required.

Environment reuse: unchanged cycle17 training and metrics specs, witnessed
kernels and fresh-agent doc receipts in the sealed parent. No installation,
new model or environment-document invocation is introduced.

Prepare a **new, absent** runtime, then execute its frozen source:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python docs/autoresearch/cycle18-image-resolution/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18 --parent /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18/source/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18
```

Commit the protocol/source and pass native replay before finer-grid sampling.
The corrected audit copies below are separately bound after the documented
[pre-numerical import failure](deviations.md). Original failed source/status
remain intact. The one-shot recovery uses [recover_audit.py](recover_audit.py)
and never relaunches sampling:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18/source/study.py run --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18/audit-v2-source/audit.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18 --phase sampling
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18/source/study.py evaluate --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18/audit-v2-source/audit.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18 --phase features
```

Only run each phase once into its designated output. Record parent handles,
commands, PIDs and terminal exits. Observe the existing handle on timeouts;
never start duplicate sampling. Private data, environments and raw logs stay
outside Git. Preserve any failed preparation and use an explicitly new runtime
if a correction is needed before quality outcomes.

The [mathematical interpretation](interpretation.md) separates regression risk,
finite-grid generation, endpoint capping and actual trajectory derivatives.
Its [two scalar fixtures](interpretation-fixtures/cap-identity.json) are analytic
checks, not fitted image results; [source coverage](primary-source-check.json)
records the two primary papers and access limits.
