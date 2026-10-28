# Gaussian image reference

The [protocol](protocol.md) selects a train-only PPCA59 reference and an
isotropic attribution model. This is an ordinary statistical control, not a
new ODE architecture. The [theory](theory.md) distinguishes the exact Gaussian
conditional mean from the existing capped/discrete sampler.

## Current scope

The complete source-bound reference cohort has finished: statistical fitting,
independent fit checks, generation, posterior/sample replay and feature/FID
audits all pass. PPCA59 gives FID 293.73 and isotropic 405.30, versus 174.71 for
the neural fixed-P control. Both fail the registered reference-quality gates.
The [complete analysis](analysis-report.md) retains every endpoint, fixed
example and gate. Its reporting-only PCA decomposition finds that similar
complement RMS can coexist with much worse FID; this does not identify a
neural mechanism or justify tuning residual variance. The
[methods](stats-appendix.md) and [figure catalog](figure-catalog.md) state
comparison and verification scopes. [Closure verification](results/artifact-verification.json) passes, including
all 17 independently recomputed reporting rows. The
[consumed archive exit](results/archive-tool-exit.json) records the seal and
the one unavailable original reporting-tool receipt.
This result does not prove every Gaussian predictor inadequate.

The original decomposition producer's terminal tool receipt was unavailable
after a truncated observation. Its complete outputs are preserved without
relaunch; closure independently checks every original-array decomposition
value. The six-stage formal controller and core curator have consumed exit0.

## Documented data-free eligibility command

Use the existing Mac mini image-metrics-v2@dc729a50 environment from
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


## Primitive result and full entrypoint preflight

The fresh [primitive doc pass](readiness/primitive-doc-pass.json) executed once
and terminated 0. All 21 state/time cases pass, with maximum discrepancy 1.78e-15;
physical prior save/load is bitwise. It reported one minor environment-label
mismatch: the ledger calls this image-metrics-v2@dc729a50, which is now named
correctly here. The passed original source/doc/receipt remains sealed under
[the primitive archive](readiness/primitive-archive.json); it is not rerun to
erase that mismatch. No actual image-model statistic or new FID is fitted.

After committing clean source, prepare a previously unused runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python -B docs/autoresearch/image-gaussian-reference/gaussian_reference.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gaussian-reference
```

A NEW fresh agent must execute this exact command once, without corrections,
retries or fitting, preserving the actual tool session/terminal exit and logs:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gaussian-reference/source/gaussian_reference.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gaussian-reference
```

PASS requires the 21 data-free cases, physical prior/native-Heun resume at the
actual 64-image/3072-dimensional/rank59 shape, independent Woodbury/manual
Heun within the fixed tolerances, actual 362496 matrix MACs per image,
normalization/MLE checks, actual-shape fake covariance/eigensolve timing,
32 historical MPS feature rows bitwise and all 32 cached FIDs. No real training
statistics are fitted. A mathematical/import-only pass is insufficient.

On both terminal 0 and no doc-vs-reality mismatches, retain the full receipts
under doc-validation and write documented-pass.json with status PASS, records
of command/exec_session/exit_code, preflight_sha256, environment_spec dc729a50
and doc_vs_reality_mismatches=[]. Stop/preserve any failed readiness runtime.
Only after the complete pass may the parent launch:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gaussian-reference/source/gaussian_reference.py run --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gaussian-reference
```

Six monitored stages: train-only statistical fit, independent raw-moment/eigen
checks, two-bank native Heun generation, independent posterior/sample replay,
pinned MPS features/FIDs and independent feature/FID replay. Save the actual
controller tool exit. Never relaunch because an observation wait expired.
Full reporting/curation follows those checks; the reference is not a successor.
