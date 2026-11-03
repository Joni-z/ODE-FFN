# Cycle12 frozen execution

Record the immutable source revision after publication. Launch only after both
fresh native readiness passes. One read-only producer perhost, no optimizer or
training process. The native batch/sampler are retained. Each producer records
case-by-case GPU/PID progress in progress.json/history.jsonl and retains partial
results if interrupted. Detached-process handling follows the existing queue.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle12-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle12/swap.py --source /data2/zz5070/ODE_FFN/professor-autoresearch-cycle12-20261103/repo/studies/muon_pd_20261102 --device cuda --host 145 --workspace /data2/zz5070/ODE_FFN --out /data2/zz5070/ODE_FFN/professor-autoresearch-cycle12-20261103/audit/frozen --detach'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle12-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle12/swap.py --source D:/Research/ODE-FFN/professor-autoresearch-cycle12-20261103/repo/studies/muon_pd_20261102 --device cuda --host lenovo --workspace D:/Research/ODE-FFN --out D:/Research/ODE-FFN/professor-autoresearch-cycle12-20261103/audit/frozen --detach'
```

Each host evaluates twelve checkpoints/twenty-four native/swap outcomes.
Confirm completed progress or a live advancing PID after SSH exits; a fast
completed diagnostic does not need to stay running for a60s launch witness.
The final results/finished markers must agree with the actual process exit.

Native replay failures stop the producer. Nonfinite swapped fields/rollouts or
sampling metrics are retained with finite fractions/null scores, not scored zero
or repaired. Require all24native array anchors and exact round-trip predictions,
then independently recompute every finite saved-array sliced-W2 score. These
48evaluations do not add rows to the training TSV. Both GPUs are released when
the producers exit; no automatic new training follows a favorable frozen swap.
