# Full density observer: invocation and resource selection

The comparison is fixed in [density-formal-next.md](density-formal-next.md):
eight checkpoints, original/projected fields, four common256-source banks,
native Heun1024/2048,128formal rows. All numerical gates remain unchanged.
The existing density_ops.pyand pinned sampler are unchanged.

## Source and environments

Publish the source before deployment. Mac mini reuses
`/Users/mr.z/ode-workspace-prep/.venv-test/bin/python` with one CPUthread per
worker.145reuses `/data2/zz5070/miniconda3/envs/py312/bin/python`,
Torch2.8.0+cu128. No package/spec changes. The CUDA controller uses no Torch
context; it binds each child to one freshly idle physical4-7UUID. A foreign
resident process or500MiB-or-more memory makes a card ineligible.0-3are
prohibited. Children check occupancy at least between rows and periodically
while recording the density path. Never stop/share another user's task.

Remote frozen worktree:
`/data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/repo`.
The immutable eight-checkpoint bundle is reused at
`/data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116/portable`.
Only the712600-byte P2data archive needs copying to the new runtime; verify
its hash after transfer completes, then extract it as portable/.

## Fresh CPU invocation

The author CPUcheck already passes eight derivative/reference cases and two
actual formal-entry witnesses. A fresh agent runs the following once verbatim:

```sh
cd /Users/mr.z/ODE-FFN-upstream && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/density_transport_20261116/formal_dispatch.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry/portable-v2 --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/portable --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/formal-doc-cpu --device cpu --phase witness
```

Require exit0, COMPLETE.jsonpassed, model_cases=8, two formal_witnesses,
unchanged parameters and exact native endpoints. All underlying derivative,
oracle, Gaussian and actual density-probe checks are repeated by this entry.

## Fresh CUDA invocation

After staging the clean worktree and the hash-verified P2data, a fresh agent
runs this once verbatim. The dispatcher selects the idle permitted card;
exit75means none is available, not a failed model or a reason to reclaim one.

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 145 'cd /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/repo && /data2/zz5070/miniconda3/envs/py312/bin/python -u studies/muon_pd_20261102/density_transport_20261116/formal_dispatch.py --root /data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116/portable --data /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/portable --output /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/formal-doc-cuda --device cuda --phase witness'
```

Require the same eight-case/two-entry receipt, CPU/CUDAderivative and native
trajectory agreement, exit0and no foreign job touched. Do not change or
rebuild the environment if only resource eligibility fails.

## Timing-only routing test

On each ready host run the same formal_dispatch.pywith phasebenchmark,
indices0,1,5, workers1and a new outputdirectory; pass the host's successful
formal-docreadiness file. This sequentially times Ring8shared P,
Product16shared P and Product16SwiGLU at256sources/32steps, both fields.
Benchmark arrays and timings are not density-performance evidence. Keep them
distinct from phaseformal. Compare outputs across CPU/CUDAbefore mixing hosts.

Choose routing from measured wall time at the actual batch size. Ordinary
models remain in the full comparison. Use --indicesonly to allocate distinct
cases to hosts; the final independent collector must see all eight IDs once
and all128formal rows. Save the routing decision before full launch.

## Full deployment and completion

Use phaseformal, the successful readiness, new outputdirectories,
workers1-4and detach. Each case has16rows: two fields x four banks x two
resolutions. Watch status.json, live PIDs, advancing row/step counts and
resource-history.json. Re-poll existing handles after observation timeouts.

Execution completion requires every selected process to exit0and every
formal COMPLETE to have16rows,256sources, four banks,1024/2048steps,
unchanged parameters and correct source/data/checkpoint digests. Scientific
completion additionally requires independent NumPy reconstruction of all
per-point identities, all numerical gates and the preregistered paired
density/time-bin analysis. Preserve unresolved cells; no automated rescue,
sample removal, architecture selection or new training is part of this run.
