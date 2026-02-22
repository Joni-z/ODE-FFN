#! /bin/bash
#SBATCH --job-name=jit
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=30:00:00

#SBATCH --gres=gpu:1

#SBATCH --account=torch_pr_63_tandon_advanced

#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/ODE_FFN/jit-codebase/logs/jit-%j.out
#SBATCH --error=/scratch/zz5070/ODE_FFN/jit-codebase/logs/jit-%j.err

set -euo pipefail

# 使用 Slurm 分配的那块 GPU，否则监控会看到“分配到的 GPU”利用率为 0 而 cancel job
# 多卡节点上 Slurm 可能分配 GPU 1，但未设置 CUDA_VISIBLE_DEVICES 时进程会用 GPU 0
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
  else
    gpu_idx=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/.*IDX:\([0-9]*\).*/\1/p' | head -1)
    if [ -n "$gpu_idx" ]; then
      export CUDA_VISIBLE_DEVICES="$gpu_idx"
    fi
  fi
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

# Conda 激活
CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate VLM || { echo "ERROR: conda activate VLM 失败"; exit 1; }
else
  echo "ERROR: 未找到 conda.sh，请设置 CONDA_SH"
  exit 1
fi
export PATH="$CONDA_PREFIX/bin:$PATH"

cd /scratch/zz5070/ODE_FFN/jit-codebase

CONFIG_NAME=${1:-jit_b16_in256_imagenet}
CONFIG_PATH="./configs/${CONFIG_NAME}.yaml"
mkdir -p logs

PRECISION=${PRECISION:-bf16}

# 单机单卡：accelerate 最简配置，减少变量也更稳
echo "Config: $CONFIG_PATH | Log: logs/${CONFIG_NAME}.log"

accelerate launch \
  --num_processes 1 \
  --mixed_precision $PRECISION \
  src/main_jit.py --config "$CONFIG_PATH" 2>&1 | tee "logs/${CONFIG_NAME}.log"
