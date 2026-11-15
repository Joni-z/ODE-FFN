# Authorized coefficient experiment

The user activated the full mechanism investigation. Mac mini is freely
available. On145 use only genuinely idle physical GPUs4-7;0-3 are prohibited.
The live inventory found resident compute processes on all4-7, so this stage
uses Mac mini CPU, two workers and the existing unchanged environment.

Two new runs: Ring8 and Product16 free-coefficient ISRU K8. The pinned native
trainer, data, sampling and metrics remain unchanged. Seed0,20k updates,
batch512, AdamW0.0003, weight decay0, gradient clipping1. There are42 extra
scalars per model, all initialized1. Existing fixed ISRU K8/K1, ordinary
Free ISRU and the previous SwiGLU references are reevaluated on new arrays.

Raw root: `/Users/mr.z/ode-workspace-prep/coefficient-20261116/`.
Data: `/Users/mr.z/ode-workspace-prep/data/synthetic/`.
Environment: `/Users/mr.z/ode-workspace-prep/.venv-test/` (warm reuse).
Evaluation streams2026111700/2026111800 are array RNGs, not training seeds.
Both streams are hashed before training. Selection measures15k/20k and
confirmation measures20k for all ten model/task combinations.

## Reproducible readiness invocation

Run from `/Users/mr.z/ODE-FFN-upstream`. Output directories must not exist:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python \
studies/muon_pd_20261102/coefficient_20261116/check.py \
--data-root /Users/mr.z/ode-workspace-prep/data/synthetic \
--assets /Users/mr.z/ode-workspace-prep/coefficient-20261116/assets \
--output /Users/mr.z/ode-workspace-prep/coefficient-20261116/doc-witness
```

This checks exact initial functions, input and original-parameter gradients,
nonzero new-coefficient gradients, actual matrix-polynomial algebra, native
8->16 save/resume versus uninterrupted16, and real evaluation/restoration
on both tasks. The final PASS.json is the readiness gate. These short16step
witnesses are excluded from research results.

Formal launch uses the existing `run_queue.py`, the new two-cell grid and
`coefficient_trial.py`. It runs from a clean frozen Git worktree outside
the development checkout. The launch receipt records the commit and actual
commands/PIDs. A separate watcher waits for both training exits and final
native test artifacts, then evaluates all registered references with the
existing evaluator and independently recomputes scores in NumPy. Persistent
status/history, queue terminal receipts and analysis COMPLETE.json are required.

## Implementation notes

The native adapter import chain already uses generic names `trial` and
`evaluate`. Initial local preflight discovered collisions with those names;
the new scripts use distinct `coefficient_trial`/`coefficient_evaluate` names.
Failed preflight records are preserved. No formal run was affected.

The tiny additional parameters enter the existing auxiliary AdamW group.
The original global clipping policy is preserved; once coefficients learn,
changed gradients/clipping are part of the parameterization intervention.
The test does not isolate an optimizer-independent theorem.
