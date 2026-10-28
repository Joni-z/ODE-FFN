# Actual P-coordinate update check

A read-only extension of the completed constant-clock derivation. It checks
six real, nonzero Adam states in the full image backbone. It is not a new
training run, an optimizer proposal or a FID experiment. The
[protocol](protocol.md) freezes the cohort, tolerance and three representations;
the [derivation](DERIVATION_PACKAGE.md) states the invariant and assumptions.

Outcome: all six real states pass. Transported parameter error is at most
2.22e-16 and post-update prediction error 1.55e-15. Ordinary native-rule matrix
steps are 1.66655–1.66981 times larger in these cases. Read the
[complete analysis](analysis.md) and [all measured rows](results/). No new
training, FID or ODE method is claimed.

Warm-reuse Mac mini image-training@475428 fb, one CPU thread. No installation
or GPU reservation. Commit source first, then prepare a new exclusive runtime:

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/p-coordinate-steps/study.py prepare --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-p-coordinate-steps-v1
```

A fresh agent executes the following documented command exactly once,
verbatim. No source inspection, edits, retries, prepare command or alternative
invocation. Preserve combined stdout/stderr and actual terminal tool metadata.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-p-coordinate-steps-v1/source/study.py run --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-p-coordinate-steps-v1
```

Expected sentinel: `P_COORDINATE_REAL_STATES_PASS`. The command replays six
native 64-image post-state prefixes bitwise and their literal float32 paths;
checks the promoted precision bridge; reconstructs six original next batches;
then checks gradients, weighted clipping, transported moments, all mapped
parameters and post-update predictions. It saves all six state summaries and
36 effective-matrix increment rows to output/result.json. The ordinary
native-W one-step counterfactual is retained without a performance gate.

The source enforces the registered 300-second ceiling and reports progress,
primed CPU use and RSS at each state. Poll the same actual tool handle until
terminal; never restart merely because an observation expires. A failed
invocation preserves source/output and its discrepancy. Actual exit 0 with all
checks is required before interpreting the result. Original checkpoints,
datasets, source snapshots, prior stop rules and submitted Report 1 remain
unchanged. Raw output and full execution logs stay outside Git.
