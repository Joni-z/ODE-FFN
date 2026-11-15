# Conditional attribution control if state evolution helps

Do not interpret a state-flow gain over frozen K8 as a uniquely ODE-specific
contribution. That comparison also increases the number of nonlinear
response evaluations. A useful repeated nonlinear computation may explain
the gain without a benefit from prescribed ODE coefficients.

If the completed state-flow comparison supports continuing, release only
the midpoint-stage coefficient in each of its four substeps:

```text
u = P ISRU(y)
mid = y + gamma_i * (dt/2) * u
y_next = y + dt * P ISRU(mid),    dt=1/4.
```

Initialize each gamma_i=1. Original weights, hidden/full outputs, input and
common-parameter gradients must match exactly before training. Both arms
retain eight P and eight response evaluations per local solve. There are
four additional scalars per solve,24 per small model, in the existing
auxiliary AdamW group. The first-order displacement is unchanged.

This releases the second-order midpoint condition at fixed computation.
It is a learned discrete-stage control, not an ODE-free network. Euler
iteration is also an ODE discretization and must not be called an ODE-free
control. Compare quality after training and before/after refinement toward
the same underlying field. The exact continuous field does not depend on
the learned finite-stage gamma values.

For four steps over small total duration s, expansion of the complete map
has second-order coefficient

```text
(s^2/2) * [3/4 + sum_i(gamma_i)/16] * J_f(h)f(h).
```

At gamma_i=1 it equals the true flow's coefficient. Generic gamma values
release this restriction while preserving the initial first-order drift.
The same initial-function tests and source freeze are mandatory before any
conditional training decision. Do not add normalization, a different q,
optimizer changes or an extra objective.

Decisions: if free stages match/improve the fixed solver, describe any
state-flow gain as useful state feedback/repeated nonlinearity pending
further attribution. If fixed stages improve held-out generation and the
benefit survives solver refinement and full-field interventions, that
supports a more specific continuous-flow-bias hypothesis. If state flow
does not beat useful ordinary controls, this control is for mechanism
attribution only and does not justify an image-training campaign.

## Execution gate and documented readiness

The completed state-flow comparison passes its within-family gate: Ring8
confirmation SW2 is0.071 versus fixed-K8's0.088 (eight of eight banks better),
and Product16 is0.052 versus0.056 (six of eight banks better). Product16
still loses to ordinary SwiGLU0.034 and Free ISRU0.051. This licenses the
attribution control, not an image-performance claim.

Run once from `/Users/mr.z/ODE-FFN-upstream`:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python \
studies/muon_pd_20261102/coefficient_20261116/stage_check.py \
--data-root /Users/mr.z/ode-workspace-prep/data/synthetic \
--assets /Users/mr.z/ode-workspace-prep/coefficient-20261116/assets \
--output /Users/mr.z/ode-workspace-prep/coefficient-20261116/stage-doc-witness
```

Require native continuation/evaluator witnesses plus exact initial functions
and common gradients in PASS.json. Then run the two-cell stage grid with the
existing native20k recipe, frozen source, persistent queue and refinement
watcher. All previous arrays, controls, checkpoints and reported scores stay
bound to their original hashes. No response or optimizer adjustment.

Metadata note: the completed state-flow grid inherited `order=8` from the
Taylor grid. This was a label error; its source and witnesses implement four
explicit midpoint steps, with accuracy order2 and eight P applications. The
old raw records are preserved. Current stage metadata uses order2 and
explicitly records substeps4/P_applications8. Family/operator definitions,
not that old column, determine the computation. No old score is changed.
