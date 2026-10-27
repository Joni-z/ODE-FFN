# Unconditional image baseline bridge

See [protocol](protocol.md), [mathematics](theory.md) and
[environment validation](environment.md). This is a baseline study, not a new
architecture candidate. The data/sources from cycle16are immutable inputs.

The producer is study.py; image_metrics.py uses a separate metrics environment.
Private runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2`.
Data: `/Users/mr.z/ode-workspace-prep/image-bridge-data/cifar10-v2`.

Prepare only after the two environment witnesses, metric fixture and fresh-agent
doc pass succeed:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python docs/autoresearch/cycle17-cifar-baselines/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2 --parent /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle16-feasibility-v2 --data /Users/mr.z/ode-workspace-prep/image-bridge-data/cifar10-v2 --env-pass /Users/mr.z/ode-workspace-prep/image-environment-doc-pass.json
```

Compute frozen validation-reference Inception features, then native trainer
preflight. Run the **frozen copies** from the private runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2/source/image_metrics.py reference --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2/source/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2
```

Training launches at most two one-thread CPU children and records their PIDs,
terminal exits, checkpoints, RNG states and five-second resource telemetry:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2/source/study.py run --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2
```

Do not restart if an observation times out; inspect the same live handle/PID.
The parent's TRAINING_COMPLETE state is not a full audited-study completion.
Run the frozen image_metrics.py evaluate action only after all four cells finish,
then independently audit every endpoint before a quality conclusion.

Weights and raw data/results remain outside Git. The committed fixture uses
random synthetic features/images and is not a trained image result. The FID
computation uses the stock compatibility extractor; the covariance reference
fixture calls current SciPy sqrtm directly, avoiding the stock wrapper's obsolete
disp argument without modifying torch-fidelity or its extractor.

The initial oct28-cycle17 preparation/reference-only snapshot is preserved.
Active v2adds physical checkpoint-roundtrip and cold-Heun checks plus literal
quadratic-term telemetry, before any quality run. Its reference bank is reused
only after byte-identical pixels/extractor/weights/source checks.
