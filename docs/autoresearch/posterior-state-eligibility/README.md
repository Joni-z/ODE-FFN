# Finite posterior-state eligibility

This is a data-free mathematical gate for the selected
[question](../post-cold-control-review/research-question-card.md), not an image
Gaussian-mixture fit, a trained FFN or a quality improvement. The
[derivation](derivation.md) keeps the third-moment term and separates fixed-y
precision evolution from fixed-z FM evolution. The
[protocol](protocol.md) fixes all cases, tolerances and the equivalence decision.

## Documented identity command

After committing clean source, prepare a new private runtime with a copy of
this folder, settings/source-commit and SHA256 bindings. Preserve the cold
and Gaussian reference seals. Use the unchanged warm Mac mini .venv-test
environment; no package installation or GPU allocation.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-posterior-state-eligibility/source/check_identities.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-posterior-state-eligibility
```

The output must retain all 72 fixed-observation cases, 72 FM-coordinate
cases and 54 integrated endpoints. An absent/truncated observation is not
permission to restart an existing process. Retain source, logs, actual tool
exit and every failed case. This command supplies no training readiness or
image-performance evidence for a later model.

## Current status

Derivation and checker are prepared. No numerical outputs or new architecture
are claimed. A positive identity check would still reject solver repackaging
alone as an ODELayer nomination under the predeclared decision.
