# Exact factored observer: documented numerical witness

Reuse hidden-defect-cuda@41450454, the existing py312 environment on145.
No installation or environment rebuild. The immutable runtime is
`/data2/zz5070/ODE_FFN/factored-analytic-observer-20261117/`; its sibling
`source-freeze.json` binds every regular file under `repo/`.

Two separately recorded variants evaluate the same coupled equations:
`linear-factorization` groups the shared linear operator; `power-reuse`
also reuses half-integer powers. The derivation and unchanged numerical gates
are in analytic-drift-factorization.md. CPU comparisons cover768learned states,
48layer cells, both solver directions and both zero-forcing directions.
These are observer execution variants, not new learned architectures.

The fresh reader receives this document and the compute contract/145ledger.
Verify all frozen source hashes before and after. Run the following commands
verbatim, once each, sequentially. Await each tool handle to its ACTUAL exit.
Do not install, edit, retry, change flags/tolerances or launch formal work.
The selector requires physicalGPU4-7below500MiBand no compute PID;0-3are
prohibited. Exit75means no eligible card and is not permission to share one.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 145 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/factored-analytic-observer-20261117/repo/studies/muon_pd_20261102/target_transport_20261117/check_compiled_defect_observer.py --root /data2/zz5070/ODE_FFN/learned-defect-fields-20261117/portable --reference /data2/zz5070/ODE_FFN/learned-defect-fields-20261117/formal --output /data2/zz5070/ODE_FFN/factored-analytic-observer-20261117/fresh-linear --cache /data2/zz5070/ODE_FFN/factored-analytic-observer-20261117/cache --variant linear-factorization --witness'
```

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 145 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/factored-analytic-observer-20261117/repo/studies/muon_pd_20261102/target_transport_20261117/check_compiled_defect_observer.py --root /data2/zz5070/ODE_FFN/learned-defect-fields-20261117/portable --reference /data2/zz5070/ODE_FFN/learned-defect-fields-20261117/formal --output /data2/zz5070/ODE_FFN/factored-analytic-observer-20261117/fresh-power --cache /data2/zz5070/ODE_FFN/factored-analytic-observer-20261117/cache --variant power-reuse --witness'
```

Each invocation must produce eight cells with the original first8inputs.
It checks compiled256-step FP64primal/tangent against the original eager
arrays, independent whole-network JVP/finite differences, zero forcing,
sign reversal, native replay and unchanged tensors/methods/RNG. Compiler
counters must show actual graphs and no graph breaks/unimplemented fallback.
Small-prefix times include compilation and are not full-batch speed evidence.

After each ACTUAL exit, write a SEPARATE receipt to `fresh-linear-receipt.json`
or `fresh-power-receipt.json` under the runtime root. Include exact command,
actual exit, before/after source audit, invocation divergence, GPUidentity,
output hashes, `fields_sha256`, `step_source_sha256`, selected `variant`,
and `passed`. A failed invocation remains failed; preserve its output/logs.
No repair or replacement launch. Parent-owned formal work later checks all
8x4096original inputs and three paired timings on ProductRK4confirmation.
