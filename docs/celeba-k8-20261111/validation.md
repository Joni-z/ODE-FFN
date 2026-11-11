# Torch validation commands

Code sync uses GitHub main. Formal jobs use a clean detached worktree of the
verified implementation commit, passed as ODE_FFN_CODE to sbatch; later
development cannot change a queued run. The CUDA12.8 venv is
`/scratch/zz5070/ODE_FFN/envs/jit-celeba-cu128`. Its Python base is the preserved
Python3.10 environment `envs/jit-celeba`; it has its own site-packages.

After connecting with `ssh torch`, run the software/data check:

```bash
cd /scratch/zz5070/ODE_FFN/ODE-FFN
source scripts/activate_torch.sh
python scripts/check_celeba.py --device cpu --output /scratch/zz5070/ODE_FFN/operations/celeba-k8-20261111/documented-cpu.json
```

Expected: `CELEBA_K8_CPU_PASS`. It checks K1/K8 against dense matrix
polynomials, derivatives through h and P, K8 against matrix exponentiation,
and the official 162,770-image CelebA training adapter. This does not allocate
a GPU.

Validate the resource request without allocating:

```bash
sbatch --test-only slurm/celeba_smoke.slurm
sbatch --test-only slurm/celeba.slurm isru_d
```

The GPU readiness job runs `slurm/celeba_smoke.slurm`: one GPU, four CPUs,
24GB RAM, fifteen-minute limit, priority account and no explicit partition.
It runs BF16 K8 dispatch and three actual optimizer/EMA updates in **both full
JiT-B/4 models at batch64**, recording memory and parameters. Expected:
`CELEBA_K8_CUDA_PASS` and `cuda-readiness.json` with status PASS.

Only one copy of that readiness job should be submitted. The two-hour K8
and SwiGLU jobs are submitted with `afterok:<readiness job id>` and
`--kill-on-invalid-dep=yes`. A queued readiness job has not passed its CUDA
check. Successful submission is not evidence that training has started.

Each training output contains `progress.json`, `epochs.jsonl`, a full atomic
`checkpoint-last.pth`, TensorBoard events and `gpu-<jobid>.csv`. `paused`
means the bounded allocation ended at an epoch boundary; `complete` means
all200epochs finished. A wall-time kill without a clean progress update is
an interruption; inspect Slurm state and the last complete checkpoint.
