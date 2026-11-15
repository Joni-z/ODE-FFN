# 145 launch and monitoring

Source freeze: `da8b8a4` in
`/data2/zz5070/ODE_FFN/professor-followup-20261115/repo`.
Warm environment: `/data2/zz5070/miniconda3/envs/py312/bin/python`.
Data: runtime/data (byte-verified local Ring8 mirror; original Product16 symlink).

| Physical GPU | Work, in order |
|---|---|
|4|Ring8 scaled tanh20k, then15k/20k native/C8 evaluation|
|5|Product16 scaled tanh20k, then15k/20k native evaluation|
|6|Ring8 ISRU-D replay20k and evaluation; SwiGLU64/Muon replay20k and evaluation|
|7|Ring8 Free ISRU replay20k and evaluation|

Each process has one CPU thread. Use the existing persistent queue, one worker
per GPU. It saves GPU use, PID, step, validation metric and completion/failure
every30seconds. The pipeline adds its training/evaluation phase. Failures stop
that GPU's pending queue and are not retried automatically. A detached queue
continues after SSH disconnects; its lock prevents accidental duplicate launch.

Template on145, substitute GPU=4,5,6,7 only after all documented checks pass
and a fresh nvidia-smi shows the target idle:

```bash
cd /data2/zz5070/ODE_FFN/professor-followup-20261115/repo
GPU=4
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/professor-followup-20261115/data --output /data2/zz5070/ODE_FFN/professor-followup-20261115/gpu$GPU --workers 1 --grid studies/muon_pd_20261102/followup_20261115/gpu$GPU.json --runner studies/muon_pd_20261102/followup_20261115/pipeline.py --experiment-id professor_followup_20261115 --detach
```

Completion requires native ADAPTER_COMPLETE, frozen-evaluation COMPLETE.json,
phase=complete, and queue-finished.json failed=false. A training checkpoint or
an idle GPU alone is not evidence of successful completion.

The C8 projection uses eight rotations per velocity evaluation. It is an
intervention to diagnose symmetry, not an equal-compute architecture claim.
Training stays seed0.15k and20k are both registered; neither is selected using
the new sampling bank. Existing Product16 controls reuse cycle14's same banks.
