# Gaussian image reference: implementation in progress

The [protocol](protocol.md) selects a train-only PPCA59 reference and an
isotropic attribution model. This is an ordinary statistical control, not a
new ODE architecture. The [theory](theory.md) distinguishes the exact Gaussian
conditional mean from the existing capped/discrete sampler.

## Current scope

The reusable Gaussian/MLE primitives are implemented. Formal fitting,
sampling, independent audit and metric/controller entrypoints are still being
prepared. No real-data statistics, new image FID or neural training outcome
exists for this reference. A mathematical fixture pass is not experiment
readiness or evidence of model quality.

## Documented data-free eligibility command

Use the existing Mac mini image-metrics@dc729a50 environment from
`.aris/compute/macmini.md`, with the unchanged
[metric specification](../cycle17-cifar-baselines/metrics-env-spec.json).
No package installation. The prepared source snapshot is outside Git.

A fresh agent must execute this command exactly once, without edits, fixes or
retry. Preserve stdout/stderr, actual tool session and terminal exit. Stop on
failure; do not fit real statistics or create a replacement output. The output
must report PASS, 21 fixed state/time cases, MLE identities, physical prior
save/load, rank 59/184380 stored scalars and real_training_statistics_fitted=false.
The three exact-denoiser gain witnesses may exceed 1; this is intentional.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-gaussian-eligibility/source/gaussian_math.py --output /Users/mr.z/ode-workspace-prep/autoresearch/oct28-gaussian-eligibility/fixtures.json
```

The parent preserves an exact source/commit binding before this command.
A later full entrypoint preflight must additionally verify actual-shape native
Heun, kernel cost, fit normalization and historical MPS/FID before any formal
reference fit. The current fixture command does not claim those checks.
