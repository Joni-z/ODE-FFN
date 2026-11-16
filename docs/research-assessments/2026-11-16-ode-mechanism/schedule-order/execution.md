# Matched stage-order control: native CPU invocation

## Scope and unchanged environment

The two models differ only in chronological signs: (+,-,-,+) versus (+,-,+,-).
Both learn P and B; B starts at zero without consuming initialization RNG.
Four midpoint stages use eight matrix applications per local solve. Midpoint
is order two, not Taylor K8. The registered question and rejection rules are
in [the proposal](../schedule-order-control.md).

Use the existing response-screen-cpu@04f11f75 environment in
`.aris/compute/macmini.md`; no installation or rebuild. Python3.12.4,
Torch2.10.0 and NumPy2.2.6 remain unchanged. SciPy provides the independent
exact-linear-exponential witness. Each process uses one CPU thread, no GPU.
The following output paths must not exist before their invocation. Existing
outputs are evidence, not permission to delete or silently overwrite them.

## Fresh agent-follows-doc validation

From `/Users/mr.z/ODE-FFN-upstream`, execute these commands verbatim, in order:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/schedule_check.py --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --assets /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-assets --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/doc-witness
```

This must exit zero and write `doc-witness/PASS.json`: both full-model
initialization/gradient checks, exact linear exponential and native nonlinear
bracket checks, B=0 and B proportional to P nulls, actual matrix-call counts,
four native8->16 versus continuous16 checkpoint comparisons, and four actual
checkpoint-evaluator witnesses. A nonzero readout prevents the original
zero-initialized head from masking input/FFN gradients. New B gradients are
measured without normalization. Both B tensors use the native auxiliary
AdamW group, learning rate0.0003 and weight decay0; both P tensors use the
native projection AdamW group with the same settings. Global clipping stays1.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/coefficient_20261116/schedule_verify.py --witness /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/doc-witness --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/doc-independent.json
```

This separate NumPy process imports no Torch and reconstructs all ten
captured nonlinear maps. It must exit zero with map error below1e-12.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/contrast_prepare.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116 --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --witness /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/doc-witness --schedule-order
```

This must bind eight existing completed reference trajectories and four new
cases in `schedule-order/anchors.json`. Every new zero-step checkpoint must
match the shared-P reference in all common tensors and data RNG. Only the six
zero B matrices may be additional. Existing common evaluation arrays stay
unchanged; their manifest hash is
`4ae16695aab2ec596aea36b048184d53ea004c9702831b56bf168a8015b71e92`.

Do not run a full trajectory during this validation. Report command exits,
actual output paths and any documentation mismatch without repairing it.

## Formal execution after validation and source publication

Freeze the published commit in the external `schedule-order/repo` worktree.
Use the existing monitored `studies/muon_pd_20261102/run_queue.py`, two CPU
workers, `schedule_trial.py`, `schedule-grid.json`, and experiment id
`schedule_order_20261116`. Four seed0 trajectories each receive20,000updates,
batch512 and the unchanged native AdamW recipe. Record the exact invocation,
commit, PIDs and resource observation in `launch.json`; verify live workers
and advancing metrics. No automatic restart or checkpoint selection.

The existing `contrast_finish.py` reads the registered schedule protocol from
anchors, audits native recipe/data-RNG pairing, evaluates15k/20k selection
and20k confirmation with `schedule_evaluate.py`, and independently recomputes
all twelve model/task rows, including eight unchanged references. It checks
native final-test replay and all eight common4096-source sampling banks.
Training completion alone is not the scientific outcome. Frozen inner4/8/16
and outer128/256/512 follow-up is required before a composition-benefit claim.

Native image training, both finalized reports and all previous frozen
runtimes remain unchanged. Physical145GPUs0-3 are prohibited;4-7 require
fresh idle-memory and compute-PID checks before any later GPU observation.
