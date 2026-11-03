# Cycle12 native readiness

Unchanged synthetic environments, read-only checkpoints and native evaluator.
Recheck freeGPU0/UUID, disk and clean source HEAD before running either command.
Lenovo source identity uses D:/Research/ODE-FFN/tools/git/cmd/git.exe directly.
Each invocation runs once in a new directory; preserve failure logs/outputs.

The command checks all24cohort records against five hashed source CSVs, then
all twelve physical checkpoint/config/cache paths for that host. The GPU kernel
witness tests native scalar transfer values/derivatives and all four families
with a nonzero fixture readout. It then loads four fixed AdamW checkpoints,
one per family: swap/restore must preserve all tensors/buffers, exact native
probe predictions and exact native64-sample Heun32 rollouts.

These are readiness probes. They do not reproduce final4096-sample quality;
formal results must independently reproduce every native cohort endpoint.
No formal frozen cohort before both fresh documented commands pass.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle12-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle12/swap.py --source /data2/zz5070/ODE_FFN/professor-autoresearch-cycle12-20261103/repo/studies/muon_pd_20261102 --device cuda --host 145 --workspace /data2/zz5070/ODE_FFN --out /data2/zz5070/ODE_FFN/professor-autoresearch-cycle12-20261103/preflight/documented --witness'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle12-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle12/swap.py --source D:/Research/ODE-FFN/professor-autoresearch-cycle12-20261103/repo/studies/muon_pd_20261102 --device cuda --host lenovo --workspace D:/Research/ODE-FFN --out D:/Research/ODE-FFN/professor-autoresearch-cycle12-20261103/preflight/documented --witness'
```

Pass: exit0, finished.json present with matching results digest, passed=true,
complete=true, witness=true,24cohort records/five CSVs verified,12path bindings,
four kernel-family rows, four actual-checkpoint rows and bindings perhost. Every
round-trip probe/rollout and parameter/buffer check must pass. Record clean Git
revision and producer/cohort hashes. No source repairs or silent retries.
