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
cd /Users/mr.z/ODE-FFN-upstream && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/density_transport_20261116/formal_dispatch.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry/portable-v2 --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/portable --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/formal-doc-cpu-batched --device cpu --phase witness --derivative-backend batched
```

Require exit0, COMPLETE.jsonpassed, model_cases=8, two formal_witnesses,
unchanged parameters and exact native endpoints. All underlying derivative,
oracle, Gaussian and actual density-probe checks are repeated by this entry.

## Fresh CUDA invocation

After staging the clean worktree and the hash-verified P2data, a fresh agent
runs this once verbatim. The dispatcher selects the idle permitted card;
exit75means none is available, not a failed model or a reason to reclaim one.

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 145 'cd /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/repo && /data2/zz5070/miniconda3/envs/py312/bin/python -u studies/muon_pd_20261102/density_transport_20261116/formal_dispatch.py --root /data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116/portable --data /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/portable --output /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/formal-doc-cuda-batched --device cuda --phase witness --derivative-backend batched'
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

## Execution-only Jacobian vectorization amendment

The initial256-source coordinate benchmark extrapolates to220minutes for the
full Product16state case. The frozen density_ops.pyand every model/sampler
remain unchanged. A process-local adapter now computes all coordinate VJPs
in one exact batched autograd call. For sample-separable outputs, applying
all identity basis vectors and permuting the returned axes is the same full
Jacobian; no random trace, low-rank approximation or sample reduction is used.

[PyTorch2.8autograd documentation](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.grad.html)
describes the vectorized VJPflag and warns that speed must be measured. The
author CPUcheck covers all sixteen model/field cells: full Jacobian errors0,
complete-observer array errors<=3.56e-15, native endpoints exact. This checks
whole trajectories as well as derivatives at multiple times and states.
The two new documented commands above rerun those checks and two actual
formal entries with --derivative-backend batchedbefore any full deployment.
Original coordinate receipts and timings remain preserved.

Timing-only queues must also pass --derivative-backend batchedwhen using the
new batched-readiness file. Measure actual256-source speed; if it is worse,
retain the coordinate backend with a separately matching readiness. Hardware
and derivative routing may differ by task only after documented equivalence.
No formal population-density row has started yet.

## Measured CPU decision and CPU fallback

The batched adapter is numerically equivalent, but at the actual256-source
batch it is slower on Mac mini: projected Product16state timing is62.291s
versus29.520sfor32steps. Therefore do not use it for CPUformal evaluation.
Keep the coordinate implementation for CPU; CUDA's batched speed remains to
be measured.145has64logical CPUs, load around22and about239GiBavailable
memory at this read-only inventory; at most four one-thread CPUworkers is a
bounded fallback that allocates no GPU. This is not a claim that exactly
42individual cores are idle. The idle-GPU policy remains unchanged.

A fresh agent runs each applicable CPUcoordinate invocation below once
verbatim before its host is declared ready for the full measurement.

Mac mini:

```sh
cd /Users/mr.z/ODE-FFN-upstream && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/density_transport_20261116/formal_dispatch.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry/portable-v2 --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/portable --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/formal-doc-cpu-coordinate --device cpu --phase witness --derivative-backend coordinate
```

145 CPU, with CUDAdevices disabled by the dispatcher:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 145 'cd /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/repo && /data2/zz5070/miniconda3/envs/py312/bin/python -u studies/muon_pd_20261102/density_transport_20261116/formal_dispatch.py --root /data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116/portable --data /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/portable --output /data2/zz5070/ODE_FFN/mechanism-density-formal-20261116/formal-doc-host-cpu-coordinate --device cpu --phase witness --derivative-backend coordinate'
```

Then benchmark indices0,1,5on145CPUsequentially with coordinate readiness,
and choose disjoint full-run case allocations from measured throughput.
No source, bank, step count or model may be omitted because a backend is slow.
