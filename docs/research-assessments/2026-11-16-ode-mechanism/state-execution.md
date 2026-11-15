# Conditional state-flow stage: gate and invocation

The active user goal authorizes the mechanism investigation. The registered
second stage requires a resolved trained-state restart defect and a measurable
complete-field correction, not an attractive equation alone.

The existing fixed-K8 probe measured the following medians over six layers,
five physical times and eight fixed states per time:

| Task | Restart defect / layer increment | Refreshed-flow correction / increment |
|---|---:|---:|
| Ring8 | 0.139 | 0.247 |
| Product16 | 0.131 | 0.207 |

Exact exponential replacement changes the complete field by at most0.000053
RMS on these probes; replacing it with state-evolving midpoint changes it
by0.077 to0.931. The gate for a nontrivial operator comparison is met.
Fixed-weight feedback worsens oracle fit here; this is not a prediction that
the retrained operator wins. Its ability to train usefully is the open test.
This stage does not depend on selecting the best coefficient-screen score.

Train Ring8/Product16 `y'=P ISRU(y)` using four midpoint substeps, eight P
applications per local solve. Same common tensors, model parameters,20k
updates, batch512, seed0, AdamW0.0003 and native training loop. Initial hidden
functions differ from fixed K8. Compare both final training outcomes and
post-training internal refinement4/8/16 before claiming continuous dynamics.
No response/optimizer/width adjustment is included.

The same prebound new arrays are used; all candidate families and orders
are specified before new confirmation is read. The coefficient-stage watcher
is a separate frozen process and continues unchanged.

Run the following readiness invocation verbatim from
`/Users/mr.z/ODE-FFN-upstream`:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python \
studies/muon_pd_20261102/coefficient_20261116/state_check.py \
--data-root /Users/mr.z/ode-workspace-prep/data/synthetic \
--assets /Users/mr.z/ode-workspace-prep/coefficient-20261116/assets \
--output /Users/mr.z/ode-workspace-prep/coefficient-20261116/state-doc-witness
```

This checks original shared tensors/RNG and parameter count, midpoint
algebra, full input gradients, solver refinement, two native8->16
continuations and two real evaluator witnesses. Require PASS.json and exit0.
Then freeze source and launch through the unchanged persistent native queue.
CPU concurrency remains two; no occupied145 GPU is shared or stopped.
