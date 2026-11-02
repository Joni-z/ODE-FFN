# Native readiness and launch commands

Run these from Mac mini. They execute kernels on the named remote hosts. A fresh
agent must execute the four preflight commands exactly and report their exit
codes and JSON witnesses before formal launch. No formal training is authorized
by this document alone; the user separately authorized this study on 2 November.

## 145: warm torch 2.8.0+cu128

Source: `/data2/zz5070/ODE_FFN/muon-pd-20261102/repo`.
Output: `/data2/zz5070/ODE_FFN/muon-pd-20261102/output`.
Existing environment: `/data2/zz5070/miniconda3/envs/py312/bin/python`.
GPU5 was free at the initial probe. Recheck it before either CUDA witness or launch.
Other GPUs are occupied and must not be shared or stopped.

```bash
ssh 145 'cd /data2/zz5070/ODE_FFN/muon-pd-20261102/repo && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/preflight.py --device cpu --output /data2/zz5070/ODE_FFN/muon-pd-20261102/preflight/cpu.json'
```

```bash
ssh 145 'cd /data2/zz5070/ODE_FFN/muon-pd-20261102/repo && CUDA_VISIBLE_DEVICES=GPU-ce799976-7805-c539-9913-3bbadcdbadbf OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/preflight.py --device cuda --output /data2/zz5070/ODE_FFN/muon-pd-20261102/preflight/cuda.json'
```

## Lenovo: warm torch 2.8.0+cu126

Source: `D:/Research/ODE-FFN/muon-pd-20261102/repo`.
Output: `D:/Research/ODE-FFN/muon-pd-20261102/output`.
Existing Python: `D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe`.
Git: `D:/Research/ODE-FFN/tools/git/cmd/git.exe`. The old dirty `repo/` and
`upstream/` checkouts are preserved; this study uses a separate snapshot.
Windows application-control policy and driver installation are unchanged.

```bash
ssh lenovo 'set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:\Research\ODE-FFN\envs\synthetic\Scripts\python.exe D:\Research\ODE-FFN\muon-pd-20261102\repo\studies\muon_pd_20261102\preflight.py --device cpu --output D:\Research\ODE-FFN\muon-pd-20261102\preflight\cpu.json'
```

```bash
ssh lenovo 'set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:\Research\ODE-FFN\envs\synthetic\Scripts\python.exe D:\Research\ODE-FFN\muon-pd-20261102\repo\studies\muon_pd_20261102\preflight.py --device cuda --output D:\Research\ODE-FFN\muon-pd-20261102\preflight\cuda.json'
```

Expected: `passed: true`, exact official optimizer updates and serialized resume,
dense-polynomial error below 1e-10, full gate derivative and paired-initialization
checks. Benchmarks are readiness/placement evidence only. They are not measured
generation quality. Compare CPU/CUDA seconds per update before choosing devices.

## Training smoke and formal queue

On each selected device, first run `run.py --cell CELL --data-root DATA_ROOT
--output SEPARATE_SMOKE_ROOT --device DEVICE --steps 8`; require native metrics,
update spectra, step-8 checkpoint and `ADAPTER_COMPLETE`. Check scalar-Muon,
PD-AdamW and SwiGLU-AdamW cells, with no formal budget changes.

Then launch `queue.py --host HOST --device DEVICE --data-root DATA_ROOT
--output OUTPUT --workers 2 --detach`. Keep the chosen CUDA binding and one CPU
thread per worker. The queue lock prevents duplicate starts; failures remain
visible and require inspection. Verify increasing steps in `monitor/status.json`
and resource samples in `monitor/history.jsonl`. Do not restart a living job
after an SSH timeout. The queue does not send external notifications.

Formal source commits, selected devices, native witnesses and exact launch
invocations will be recorded in `execution.md` before dispatch.
