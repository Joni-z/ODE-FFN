# Invariant-clock feature flow

This is one prospective FFN constraint, not a validated improvement. The
[question](research-question-card.md), [primary prior art](literature-review.md)
and [derivation](theory.md) distinguish an exact hidden feature flow from
outer FM and from a Gaussian posterior. Existing action-angle methods already
contain the rotation construction; no mathematical novelty is claimed.

## Operator eligibility

After freezing this committed source outside Git, run the data-free command
in the unchanged warm Mac mini .venv-test environment, with one CPU thread:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-invariant-clock-operator/source/verify_operator.py --output /Users/mr.z/ode-workspace-prep/autoresearch/oct28-invariant-clock-operator/fixtures.json
```

This checks literal input/parameter derivatives, full feedback gain,
independent integration, active routes, counterexamples, initial identity
pairing and actual width64 matrix work. Preserve actual exit/log/source and
every failure. Passing it does not establish native model integration,
checkpoint/resume, full experiment readiness or image-quality improvement.

Only after a separately registered complete native preflight and fresh
agent-follows-doc pass may the finite old-law training screen start. No
training command is selected in this initial operator document.
