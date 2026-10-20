#!/bin/bash
#SBATCH --job-name=ode-stage-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/zz5070/ODE_FFN/runs/synthetic/slurm/%j-report.out
#SBATCH --error=/scratch/zz5070/ODE_FFN/runs/synthetic/slurm/%j-report.err
set -euo pipefail
cd /scratch/zz5070/ODE_FFN/experiments/stage_20261020/runtime
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python scripts/verify_stage_sources.py
/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python scripts/build_stage_report.py \
    --report-root /scratch/zz5070/ODE_FFN/jit-codebase/docs/research-2026-10-20 --prior-root /scratch/zz5070/ODE_FFN/jit-codebase/docs/research-2026-10-19 \
    --output-root /scratch/zz5070/ODE_FFN/runs/synthetic
