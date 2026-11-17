# Analytic target transport: exact entrypoint witness

Reuse145hidden-defect-cuda@41450454and the existing py312 stack. No packages
change. The new immutable runtime is `/data2/zz5070/ODE_FFN/analytic-target-20261117/`.
Its source-freeze manifest binds every regular source file present at archive
extraction. Runtime bytecode/compiler caches are not additional frozen source.
Audit all manifested hashes before/after; preserve runtime caches.

The selected power-reuse observer already passes all8x4096field comparisons,
independent NumPy reconstruction and the fresh documented derivative controls.
Three paired timings give4.475xresident speedup. Its immutable execution proof
is the formal-power/COMPLETE.json in the factored observer runtime. Main entry
checks that proof and its independent array reconstruction before execution.

Fresh reader: receive this document and the compute contract/145ledger only.
Run the four commands below verbatim, once each, sequentially, awaiting each
original handle to ACTUAL exit. Do not edit/install/retry/change flags or launch
formal work. Exit75means no permitted idle GPU. Physical0-3are prohibited;
4-7require<500MiBand no compute PID. Never share/stop another user's task.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 145 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/analytic-target-20261117/repo/studies/muon_pd_20261102/target_transport_20261117/evaluate_analytic_target.py --root /data2/zz5070/ODE_FFN/population-resolution-20261117/portable --reference /data2/zz5070/ODE_FFN/population-resolution-20261117/formal --output /data2/zz5070/ODE_FFN/analytic-target-20261117/fresh-product --cache /data2/zz5070/ODE_FFN/analytic-target-20261117/cache --execution-proof /data2/zz5070/ODE_FFN/factored-analytic-observer-20261117/formal-power/COMPLETE.json --case product_rk4_trained --split confirmation --bank 0 --steps 128 --witness'
```

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 145 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/analytic-target-20261117/repo/studies/muon_pd_20261102/target_transport_20261117/verify_analytic_target.py --input /data2/zz5070/ODE_FFN/analytic-target-20261117/fresh-product --output /data2/zz5070/ODE_FFN/analytic-target-20261117/fresh-product/independent.json'
```

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 145 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/analytic-target-20261117/repo/studies/muon_pd_20261102/target_transport_20261117/evaluate_analytic_target.py --root /data2/zz5070/ODE_FFN/population-resolution-20261117/portable --reference /data2/zz5070/ODE_FFN/population-resolution-20261117/formal --output /data2/zz5070/ODE_FFN/analytic-target-20261117/fresh-ring --cache /data2/zz5070/ODE_FFN/analytic-target-20261117/cache --execution-proof /data2/zz5070/ODE_FFN/factored-analytic-observer-20261117/formal-power/COMPLETE.json --case ring_midpoint_trained --split selection --bank 0 --steps 128 --witness'
```

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 145 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/analytic-target-20261117/repo/studies/muon_pd_20261102/target_transport_20261117/verify_analytic_target.py --input /data2/zz5070/ODE_FFN/analytic-target-20261117/fresh-ring --output /data2/zz5070/ODE_FFN/analytic-target-20261117/fresh-ring/independent.json'
```

The explicit witness flag selects two original sources, Heun8and hidden
reference16for derivative-composition readiness. It is not a quality result
or a formal reference reduction. Formal work retains4,096sources,128/2048
outer steps and256hidden-reference steps. Both forcings travel through the
actual sampler; independent epsilon finite differences check the learned
analytic forcing, while the measured forcing also has a whole-sampler JVP.
The earlier unsupported nested-forward-ADattempt is retained separately.

Each main invocation must yield passed=true and numerically_qualified=true,
exact/native replay within its prescribed gate,16prebound trajectory probes,
128/256observer checks, target derivative checks and unchanged tensors/methods/
RNG. Each independent command must yield passed=true and bind its COMPLETE
and bank-array hashes. No compiler graph breaks/unimplemented fallback.

After all ACTUAL exits, write a SEPARATE receipt to the runtime's
fresh-readiness.json. Include exact commands/exits, both output/independent
hashes, source audits, invocation divergences, source_sha256 of the main
entrypoint, kernel_sha256, portable_sha256, GPUidentity and passed Boolean.
A failure remains failed; preserve outputs and do not repair or restart.
Parent later validates the receipt before any monitored formal bank queue.
