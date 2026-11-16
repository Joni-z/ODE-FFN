# CUDA readiness for pending new-model refinements

This document records CUDAreadiness obtained while the two Mac formal-density
workers were running. Both documented native entries below passed. Subsequent
measured routing moved Product16to a GPUwhile preserving its suspended CPU
fallback; see [the separate execution record](observations/resource-handoff.md).
The models, source points, numerical gates and observer stay fixed by
observation-protocol.md. Readiness alone is not a population observation.

## Source and transferred inputs

Host145, separate root:
`/data2/zz5070/ODE_FFN/mean-preserving-observations-20261116`.
Source is a clean detached fd7f09b0064ceab36a13f7757fd2716d002cbba6worktree,
fetched through the public canonical GitHub repository. Earlier worktrees
and the running Mac source are unchanged. Warm py312/Torch2.8.0+cu128 is
reused, with one CPU thread/process and no package changes.

The42,373,120-byte input archive has SHA256
`0aa3379abccc7a55eec2946394625cd090c4691856a003a91bccdc1505c9476b`.
All fourteen checkpoint/config/manifest/asset/reference files must match the
per-file binding after extraction. No new evaluation stream is generated.

## Fresh documented CUDA entry

A fresh pass executes this command verbatim once, with absent output paths:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 145 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/repo/studies/muon_pd_20261102/density_transport_20261116/formal_dispatch.py --root /data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/input/density-models --data /data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/density-data --output /data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/doc-cuda-2 --phase witness --device cuda --derivative-backend batched'
```

The standard dispatcher selects a freshly empty physical GPU4-7, requiring
memory below500MiB and no resident compute PID. Physical0-3are prohibited.
Exit75means no eligible card; it does not authorize stopping or sharing a
foreign task. Children recheck occupancy. Do not assign a remembered UUID
without a new resource check.

Require exit0, model_cases2, two formal-entry witnesses, four exact batched
Jacobian/backend cells, CPU-anchor agreement, native endpoint equality and
unchanged parameters. The driver has no CUDA allocation while its child
runs. Record the selected physical index/UUID and actual PID. Verify the
source/data digests and independently reconstruct both native-law witness
observations. A successful import alone is not readiness.

The GPU is released after the witness. Do not reserve it while waiting for
Mac parents. After a successful completed parent is available, copy only its
complete rows and verify them before any adaptive execution. Keep the
original Mac outputs and all source identifiers. Tiny Ring8refinement subsets
retain the previously measured CPU preference; future Product16CUDA work
requires its separately recorded adaptive-entry witness and a fresh idle card.


## First resource attempt

The first documented command exited75before launching a child or creating
its output. The archive and all14files were verified and the source stayed
clean. A later author audit calls the exact resource-selector function and
finds physical5/7eligible, matching raw driver memory/process output. No
reproducible parser fault was found; the instantaneous eligibility result
must not be weakened to force a launch. Record the first attempt separately.
The command above uses a new absent `doc-cuda-2`output for the next witnessed
attempt after that positive eligibility audit. No model, data, environment
or numerical gate changes.


## Product16 adaptive-entry witness

The second formal-entry witness passed on physicalGPU5and released its
context. Before the next command, independently check that GPU5again has
memory below500MiB and no resident compute PID. If that precondition fails,
stop without executing or substituting another card. The native guard also
checks UUIDbinding and foreign compute PIDs before any model allocation.
This uses the actual completed CUDA witness as parent, not a live Mac case.

A fresh documented pass executes this single command verbatim once:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 145 'env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=GPU-ce799976-7805-c539-9913-3bbadcdbadbf /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/repo/studies/muon_pd_20261102/density_transport_20261116/adaptive_density.py --root /data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/input/density-models --data /data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/density-data --case-output /data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/doc-cuda-2/formal-entry-1 --parent-commit fd7f09b0064ceab36a13f7757fd2716d002cbba6 --output /data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/adaptive-doc-cuda-product16 --device cuda --derivative-backend batched --gpu-uuid GPU-ce799976-7805-c539-9913-3bbadcdbadbf --witness'
```

Require exit0, witness=true, four native rows and two merged four-source
cells with numerical_qualified=false. This tests execution and native
identity; it is not a population result. Preserve all source/hash/gate
records. After independent NumPy reconstruction, this readiness may be used
for a future Product16refinement with a successful terminal formal parent.
The normal adaptive dispatcher must freshly select an empty permitted card;
this witness does not reserve GPU5. The witness itself leaves Mac workers intact.
