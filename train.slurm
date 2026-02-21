#! /bin/bash
#SBATCH --job-name=jit
#SBATCH --nodes=1
#SBATCH --ntasks=1

# 尽量少要资源：更容易被调度（Torch 指南强调只申请所需资源）
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00

# 你就是要 1 张卡
#SBATCH --gres=gpu:1

# Torch 要求必须带 account
#SBATCH --account=torch_pr_63_tandon_advanced

#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/ODE_FFN/jit-codebase/logs/jit-%j.out
#SBATCH --error=/scratch/zz5070/ODE_FFN/jit-codebase/logs/jit-%j.err

set -euo pipefail

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
