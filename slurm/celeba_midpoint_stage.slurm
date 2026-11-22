#!/bin/bash
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|l40s|rtx6000|b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/zz5070/ODE_FFN/runs/celeba/slurm/%x-%j.out
set -euo pipefail
cd "${ODE_FFN_CODE:?missing frozen runtime}"
source scripts/activate_torch.sh
unset RANK WORLD_SIZE LOCAL_RANK SLURM_PROCID
export PYTHONUNBUFFERED=1
root=/scratch/zz5070/ODE_FFN
run="$root/runs/celeba/20261123-midpoint4-detached-s0"
[[ "$(git rev-parse HEAD)" == "$(cat "$run/source-commit.txt")" ]]
phase=${1:?train or fid}
epoch=${2:?milestone number}
stage_output="$run"
if [[ "$phase" == fid ]]; then
    stage_output=$(printf '%s/paired/epoch-%04d' "$run" "$epoch")
fi
mkdir -p "$stage_output"
gpu_uuid=$(python -c 'import torch; print("GPU-" + str(torch.cuda.get_device_properties(0).uuid))')
nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total --format=csv -i "$gpu_uuid" -l 30 > "$stage_output/gpu-$SLURM_JOB_ID.csv" &
gpu_monitor=$!
local_data=$(mktemp -d "/tmp/ode-detached-${SLURM_JOB_ID}.XXXXXX")
trap 'kill "$gpu_monitor" 2>/dev/null || true; rm -rf -- "$local_data"' EXIT
export TMPDIR="$local_data"
data_source="$root/datasets/celeba"
mkdir -p "$local_data/celeba"
cp "$data_source"/{list_attr_celeba.txt,identity_CelebA.txt,list_bbox_celeba.txt,list_landmarks_align_celeba.txt,list_eval_partition.txt} "$local_data/celeba/"
python -m zipfile -e "$data_source/img_align_celeba.zip" "$local_data/celeba"
python -u scripts/celeba_midpoint_stage.py "$phase" "$epoch" --data "$local_data" --run "$run" \
    --native "$root/runs/celeba/20261114-b16-swiglu-s0-to160" --validation "$root/operations/celeba-midpoint-detached/cuda-check.json"
