# Adaptive density refinement: real invocations

Read [the numerical/statistical registration](density-adaptive-followup.md).
The eight trained models, source banks and observer are unchanged. The new
entry only replaces numerically unresolved source observations with finer
native-sampler observations. Warm environment specifications are unchanged.

## Fresh Mac CPU witness

Run the following once verbatim. It checks the completed parent against the
frozen source, then exercises both fields, native Heun16/32, and the full merge
on four sources. This is a numerical-entry witness, not population evidence.

```sh
cd /Users/mr.z/ODE-FFN-upstream && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/density_transport_20261116/adaptive_dispatch.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry/portable-v2 --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/portable --runtimes /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/formal-runs-mac-ring-controls --indices 2 --device cpu --derivative-backend coordinate --witness --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/adaptive-doc-cpu
```

Require exit 0, `COMPLETE.json` passed/witness, four subset rows with exact
native endpoints, two merged four-source fields, and unchanged parameters.
Independently verify the witness with `verify_adaptive.py` and the published
source commit before declaring the new entry ready.

## Full Mac refinement

After the fresh witness and independent check, run the following once. It
retains 1024 sources in each field for each of the two completed ordinary
Ring8 controls. Each accepted source stops individually; every unresolved
source is retained through the registered upper bound.

```sh
cd /Users/mr.z/ODE-FFN-upstream && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/density_transport_20261116/adaptive_dispatch.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry/portable-v2 --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/portable --runtimes /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/formal-runs-mac-ring-controls --indices 2,3 --device cpu --derivative-backend coordinate --workers 2 --readiness /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/adaptive-doc-cpu/COMPLETE.json --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/adaptive-runs-mac-ring-controls --detach
```

Record the queue PID, both worker PIDs, advancing subset step counts and
terminal exit codes. Execution success and numerical qualification are separate
fields. Do not restart a live worker after an observation timeout.

## Remaining six cases

Wait for successful full-parent worker completion before refining its sources.
Use the unchanged GPU4-7 selector and exclusive UUID guard if a card becomes
idle; otherwise use Mac CPU with its matching witness. Never modify the running
formal producer. The separate frozen refinement worktree is
`/data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/adaptive-repo`, source
405a201. The original 3a7d5fe worktree remains unchanged.

After Product16 shared-generator parent index 1 has exited successfully and
a permitted GPU is idle, a fresh agent runs this witness once verbatim:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 145 'cd /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/adaptive-repo && /data2/zz5070/miniconda3/envs/py312/bin/python -u studies/muon_pd_20261102/density_transport_20261116/adaptive_dispatch.py --root /data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116/portable --data /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/portable --runtimes /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/formal-runs-cuda-batched --indices 1 --device cuda --derivative-backend batched --witness --output /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/adaptive-doc-cuda'
```

Exit 75 means no permitted idle card. Do not reclaim or share a card. Require
the same four-subset/two-merge/unchanged-parameter checks and independent
native-law reconstruction as on CPU. This exercises the real 16-dimensional
entry and both fields on the validated batched Jacobian backend.

When all six GPU parent workers have exited successfully, launch the bounded
refinement once using the successful CUDA witness. The stdlib dispatcher may
wait for newly idle cards, using one worker per permitted GPU.

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 145 'cd /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/adaptive-repo && /data2/zz5070/miniconda3/envs/py312/bin/python -u studies/muon_pd_20261102/density_transport_20261116/adaptive_dispatch.py --root /data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116/portable --data /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/portable --runtimes /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/formal-runs-cuda-batched --indices 1,7,0,6,5,4 --device cuda --derivative-backend batched --workers 4 --readiness /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/adaptive-doc-cuda/COMPLETE.json --output /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/adaptive-runs-cuda --detach'
```

## Independent audit

`verify_adaptive.py --root ... --data ... --source-commit <published commit>
--cases <adaptive case folders> --parents <matching formal case folders>
--output <new audit directory>` replays the initial failures, every selected
subset, all native-law density identities and the exact all-source merge.
Full statistical analysis requires eight unique model IDs and all sixteen
model/field qualification decisions; missing or unresolved cells remain visible.
