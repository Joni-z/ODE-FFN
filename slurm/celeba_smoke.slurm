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
unset RANK WORLD_SIZE LOCAL_RANK SLURM_PROCID
IMAGE_SIZE=${1:-64}
if [[ "$IMAGE_SIZE" == 128-wide ]]; then
    OUTPUT=/scratch/zz5070/ODE_FFN/operations/celeba128-wide-20261114/cuda-readiness.json
    BATCH_SIZE=256
elif [[ "$IMAGE_SIZE" == 128 ]]; then
    OUTPUT=/scratch/zz5070/ODE_FFN/operations/celeba128-20261112/cuda-readiness.json
    BATCH_SIZE=${ODE_FFN_BATCH_SIZE:-256}
elif [[ "$IMAGE_SIZE" == 64 ]]; then
    OUTPUT=/scratch/zz5070/ODE_FFN/operations/celeba-k8-20261111/cuda-readiness.json
    BATCH_SIZE=64
else
    exit 2
fi
GPU_UUID=$(python -c 'import torch; print("GPU-" + str(torch.cuda.get_device_properties(0).uuid))')
echo "Monitoring CUDA device $GPU_UUID; Slurm global IDs ${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -i "$GPU_UUID" -l 5 > "${OUTPUT%.json}-${SLURM_JOB_ID}-gpu.csv" \
    2> "${OUTPUT%.json}-${SLURM_JOB_ID}-gpu.err" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$MONITOR_PID"
if [[ "$IMAGE_SIZE" == 128-wide ]]; then
    python tests/check_wide_ffn.py --device cuda --batch-size "$BATCH_SIZE" --output "$OUTPUT"
else
    python scripts/check_celeba.py --device cuda --full-model --batch-size "$BATCH_SIZE" \
        --image-size "$IMAGE_SIZE" --output "$OUTPUT"
fi
