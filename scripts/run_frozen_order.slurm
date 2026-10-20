#!/bin/bash
#SBATCH --job-name=ode-frozen-order
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|h100|h200|rtx6000"
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zz5070/ODE_FFN/runs/synthetic/slurm/%j-frozen-order.out
#SBATCH --error=/scratch/zz5070/ODE_FFN/runs/synthetic/slurm/%j-frozen-order.err
set -euo pipefail
cd /scratch/zz5070/ODE_FFN/experiments/stage_20261020_frozen_order/runtime
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export GIT_OPTIONAL_LOCKS=0
/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python scripts/run_frozen_order.py launch \
  --manifest manifest.json \
  --output /scratch/zz5070/ODE_FFN/runs/synthetic/stage_20261020_frozen_order
