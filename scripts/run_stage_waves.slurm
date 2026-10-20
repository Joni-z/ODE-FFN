#!/bin/bash
#SBATCH --job-name=ode-stage-waves
#SBATCH --array=0-2%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|h100|h200"
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/zz5070/ODE_FFN/runs/synthetic/slurm/%A_%a.out
#SBATCH --error=/scratch/zz5070/ODE_FFN/runs/synthetic/slurm/%A_%a.err

set -euo pipefail
cd /scratch/zz5070/ODE_FFN/experiments/stage_20261020/runtime
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python
printf 'job=%s task=%s nodes=%s CUDA_VISIBLE_DEVICES=%s\n' "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SLURM_JOB_NODELIST}" "${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "No CUDA device visible in Slurm allocation"; print({"torch": torch.__version__, "cuda": torch.version.cuda, "visible_gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, flush=True)'
"$PYTHON" scripts/verify_stage_sources.py
"$PYTHON" scripts/run_stage_waves.py --index "${SLURM_ARRAY_TASK_ID}" --device cuda \
    --output-root /scratch/zz5070/ODE_FFN/runs/synthetic --report-root /scratch/zz5070/ODE_FFN/jit-codebase/docs/research-2026-10-20
