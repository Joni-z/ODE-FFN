#!/bin/bash
#SBATCH --job-name=celeba-k8-readiness
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|l40s|rtx6000|b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/zz5070/ODE_FFN/operations/celeba-k8-20261111/cuda-%j.out

set -euo pipefail
cd "${ODE_FFN_CODE:-/scratch/zz5070/ODE_FFN/ODE-FFN}"
source scripts/activate_torch.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1
python scripts/check_celeba.py --device cuda --full-model --batch-size 64 \
    --output /scratch/zz5070/ODE_FFN/operations/celeba-k8-20261111/cuda-readiness.json
