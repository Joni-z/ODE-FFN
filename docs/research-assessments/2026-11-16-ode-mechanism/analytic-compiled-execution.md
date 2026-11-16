# Exact compiled-step numerical witness

Use the unchanged145hidden-defect-cuda@41450454stack: Python3.12.11,
Torch2.8.0+cu128,NumPy1.26.4,Triton3.4.0. gcc/g++are present;nvccis absent.
No packages are installed or rebuilt. The actual seeded CUDAcompilation below,
not the inventory alone, determines compiler readiness.

The canonical eager observer and learned tensors remain unchanged. The adapter
factors one autonomous coupled RK4step and calls torch.compile with
fullgraph=True,dynamic=False,default mode. Fixed CPUchecks66114exit0: all eight
whole-network prefix outputs and48nonzero-incoming-variation layer checks match
the original bit for bit. All original4,096-input reference arrays are already
verified. The fresh reader needs only this document, the compute contract and
the relevant provider ledger; no further protocol text is needed to execute.

Check all adjacent source-freeze.json hashes before/after. The archive has no
Git metadata. Do not edit, install, change flags, alter tolerances or run formal
full-field/timing tests. The script selects a freshly empty physicalGPU4-7and
binds its UUIDbefore Torch import; never0-3or another user's task. Exit75means
no eligible GPU. Keep that outcome, with no replacement launch.

Execute this command verbatim once. Await the same tool session to ACTUAL
terminal exit; preserve failures and any compiler fallback/graph-break report.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 145 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/compiled-analytic-observer-20261117/repo/studies/muon_pd_20261102/target_transport_20261117/check_compiled_defect_observer.py --root /data2/zz5070/ODE_FFN/learned-defect-fields-20261117/portable --reference /data2/zz5070/ODE_FFN/learned-defect-fields-20261117/formal --output /data2/zz5070/ODE_FFN/compiled-analytic-observer-20261117/fresh-witness --cache /data2/zz5070/ODE_FFN/compiled-analytic-observer-20261117/cache --witness'
```

Expected eight cells, each the original first8inputs. Compare the compiled256
primal/tangent with the saved original values, FP64unchanged. The script also
executes independent whole-network epsilon-JVP/FD,zero/sign reversal, exact
native replay and unchanged tensors/methods/RNGchecks. It retains compiler
counters, compilation timing diagnostics and actual output arrays. No graph
break, unimplemented fallback or zero compiled-graph count is a pass.

After ACTUAL exit, write the receipt in a SEPARATE call to
/data2/zz5070/ODE_FFN/compiled-analytic-observer-20261117/fresh-receipt.json.
Include command/actual exit, before/after source hashes, invocation divergence,
GPUidentity, output hashes, fields_sha256,step_source_sha256 and passed Boolean.
Require zero actual exit, unchanged source and the complete eight-cell result.
If it fails, preserve FAILED.json/logs and record passed=false. Do not repair
or retry. Expected bytecode files and compiler caches remain outside the
manifested source; preserve them as runtime artifacts.

Parent-owned formal validation later compares ALL4,096inputs per cell and runs
three synchronized paired timings on the original ProductRK4confirmation field.
Small-prefix checks establish numerical readiness, not full-batch speed or
generative quality. All reference counts, data and scientific gates stay fixed.

API references: [PyTorch2.8compile](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html),
[compiler diagnostics](https://docs.pytorch.org/docs/2.8/torch.compiler_troubleshooting.html).
