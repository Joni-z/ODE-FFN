# Frozen hidden-fidelity observer readiness

Use warm response-screen-cpu@04f11f75 and no package changes. A fresh agent reads
the compute contract/provider ledger and runs the following command verbatim
from /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/observer-repo.
First confirm fidelity-readiness/check does not exist. No source/environment/
argument fixes, retries or formal dispatch. Poll the same handle to actual exit.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/rk4_fidelity_check.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/fidelity-readiness/check
```

Require PASS.json passed=true, four cases, six layers/case, eight inputs/layer,
common_inputs_exact=true, and eight actual child exits0. Both numerical schemes
and tasks must qualify all points in both the original two-tolerance/work
reference and the independent per-input RK45 reference. Report reconstruction/
reference errors and every document/reality divergence. This is a numerical
observer check, not new training or a generation-quality result.

Own only fidelity-readiness/command-exits.json and an optional witness note.
Record the parent invocation's actual exit and observer-source-freeze.json.
The checker already preserves its children's exits in check/command-exits.json.
The source archive has no .git. Do not alter the live training runtime repo-v2.

After this pass and native20k/selection completion, formal observer commands
use rk4_fidelity.py with --folder/--evaluation pointing to each of the two new
RK4runs and the two prebound midpoint runs; omit --witness and --step. All use
the prebound data/assets roots above and new fidelity-formal/<task>-<solver>
outputs. Then invoke verify_rk4_fidelity.py on each retained output. No observer
result can replace targetSW2or the ordinary controls in the registered decision.
