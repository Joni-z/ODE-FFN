# All-source trajectory reference: exact CPU execution witness

Reuse response-screen-cpu@04f11f75 on Mac mini: the existing .venv-test,
Python3.12.4/Torch2.10.0/NumPy2.2.6, one thread per process. No package changes.
The new immutable source archive is
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/runtime/`.
Its manifest binds regular source files at archive extraction; preserve runtime
bytecode outside that manifest. Audit all manifested hashes before and after.

The helper reconstructs both Heun stages from the actual saved native path,
then checks the same 128/256-step analytic observer on every retained input.
CPU equations use the separately verified power-reuse algebra. The two-source
GPU readiness banks already exist; their arrays are immutable input anchors,
not evidence of full population quality. Both authored CPU checks and NumPy
reconstructions pass. The first complete 4,096-input timing takes 46.313/95.056
seconds at the two references, with 584.828 MiB peak RSS. The eventual collector
is capped at four one-thread CPU workers; it never allocates a GPU or starts
training. The four-card GPU sampling queue remains independent and unchanged.

Fresh reader receives this document and the compute contract/Mac ledger only.
Run the five commands verbatim, sequentially, once each. Await each original
handle to ACTUAL terminal exit. No edit, installation, retry, data regeneration
or formal follow-up queue. All commands explicitly disable CUDA/MPS use.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/runtime/repo/studies/muon_pd_20261102/target_transport_20261117/trajectory_reference.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/population-resolution/portable --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/analytic-target/fresh-product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/fresh-product --execution-proof /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/factored-analytic-observer/formal-power/COMPLETE.json
```

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/runtime/repo/studies/muon_pd_20261102/target_transport_20261117/verify_trajectory_reference.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/fresh-product --main /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/analytic-target/fresh-product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/fresh-product/independent.json
```

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/runtime/repo/studies/muon_pd_20261102/target_transport_20261117/trajectory_reference.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/population-resolution/portable --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/analytic-target/fresh-ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/fresh-ring --execution-proof /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/factored-analytic-observer/formal-power/COMPLETE.json
```

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/runtime/repo/studies/muon_pd_20261102/target_transport_20261117/verify_trajectory_reference.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/fresh-ring --main /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/analytic-target/fresh-ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/fresh-ring/independent.json
```

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/runtime/repo/studies/muon_pd_20261102/target_transport_20261117/collect_trajectory_reference.py --repo /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/runtime/repo --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/population-resolution/portable --local-main /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/analytic-target/formal --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/fresh-inventory --execution-proof /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/factored-analytic-observer/formal-power/COMPLETE.json --python /Users/mr.z/ode-workspace-prep/.venv-test/bin/python --inventory-only
```

Each producer must retain all 16 calls x two sources, exact times, reconstructed
path inputs, both reference arrays and saved-GPU parity, unchanged tensors/
methods/RNG, and numerically_qualified=true. Each independent verifier must
reconstruct all seven array-derived gates and bind the exact COMPLETE/array
hashes. The final command is read-only remote discovery, recording eligible
completed-bank metadata and the actual controller state. It neither copies
data nor starts model evaluation. Discovery may observe different progress
as the main queue advances; that is not invocation divergence.

After all ACTUAL exits, write a SEPARATE receipt to
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/trajectory-reference/fresh-readiness.json`.
Include exact commands/exits, output hashes, before/after source audits,
invocation/document divergences, source_sha256 of trajectory_reference.py,
portable_sha256, and passed. A failed command remains failed; preserve its
outputs and do not repair or retry. Parent-owned formal execution may start
only after this receipt is independently checked. Full scientific scope remains
128 bank/mesh tasks x 16 calls x 4,096 sources, including every failed gate.
