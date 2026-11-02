# Wave2 native witness commands

A fresh agent must execute these exact commands once on each host. Recheck GPU5
on145 and GPU0 onLenovo first. Code uses a separate frozen worktree; first-wave
source and outputs remain untouched. These commands perform bounded readiness
checks only, not formal20k runs. Do not repeat a failed smoke into the same output
folder; preserve the failed directory and document an explicit new witness path.

All original-arm tensors and three updates must agree with the frozen first-wave
implementation. Norm matching, direction preservation, dense sandwich/Jacobian,
optimizer decay scope and physical resume must pass. CUDA additionally invokes
three8-update full-trainer smoke cases and resumes one from8 to16.

## 145

```bash
ssh 145 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/muon-pd-wave2-20261102/repo/studies/muon_pd_20261102/wave2/preflight_wave2.py --device cpu --frozen-source /data2/zz5070/ODE_FFN/muon-pd-20261102/repo/studies/muon_pd_20261102 --output /data2/zz5070/ODE_FFN/muon-pd-wave2-20261102/preflight/cpu.json'
```

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-ce799976-7805-c539-9913-3bbadcdbadbf OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/muon-pd-wave2-20261102/repo/studies/muon_pd_20261102/wave2/preflight_wave2.py --device cuda --frozen-source /data2/zz5070/ODE_FFN/muon-pd-20261102/repo/studies/muon_pd_20261102 --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --smoke-output /data2/zz5070/ODE_FFN/muon-pd-wave2-20261102/preflight/native-smoke --output /data2/zz5070/ODE_FFN/muon-pd-wave2-20261102/preflight/cuda.json'
```

## Lenovo

```bash
ssh lenovo 'set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:\Research\ODE-FFN\envs\synthetic\Scripts\python.exe D:\Research\ODE-FFN\muon-pd-wave2-20261102\repo\studies\muon_pd_20261102\wave2\preflight_wave2.py --device cpu --frozen-source D:/Research/ODE-FFN/muon-pd-20261102/repo/studies/muon_pd_20261102 --output D:/Research/ODE-FFN/muon-pd-wave2-20261102/preflight/cpu.json'
```

```bash
ssh lenovo 'set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:\Research\ODE-FFN\envs\synthetic\Scripts\python.exe D:\Research\ODE-FFN\muon-pd-wave2-20261102\repo\studies\muon_pd_20261102\wave2\preflight_wave2.py --device cuda --frozen-source D:/Research/ODE-FFN/muon-pd-20261102/repo/studies/muon_pd_20261102 --data-root D:/Research/ODE-FFN/datasets/synthetic --smoke-output D:/Research/ODE-FFN/muon-pd-wave2-20261102/preflight/native-smoke --output D:/Research/ODE-FFN/muon-pd-wave2-20261102/preflight/cuda.json'
```

## Formal queue after readiness

Use the shared `run_queue.py`, with absolute `--grid` pointing to wave2/grid.json,
`--runner` pointing to wave2/train_wave2.py, `--experiment-id muon_pd_wave2_20261102`,
and the correct host, CUDA device binding, data root and new output root. Keep
`--workers 2 --detach`. Windows includes CREATE_BREAKAWAY_FROM_JOB, previously
verified to survive OpenSSH exit. The queue now polls exits every2s and logs a
heartbeat every30s or on transitions, avoiding idle time between completed jobs.
A failure stops new dispatches; no automatic retries or extra families.
