#!/bin/bash
# 在登录节点运行，用与 train.slurm 相同的环境做一次快速自检，再 sbatch 可减少排到卡后秒挂的概率。
# 用法: ./test_before_submit.sh [config名，默认 jit_b16_in256_imagenet]
set -e
cd "$(dirname "$0")"
CONFIG_NAME=${1:-jit_b16_in256_imagenet}
CONFIG_PATH="./configs/${CONFIG_NAME}.yaml"

echo "=== 使用与 sbatch 相同的 conda 与环境 ==="
CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_SH" ]; then
  echo "ERROR: 未找到 $CONDA_SH，请设置 CONDA_SH"
  exit 1
fi
source "$CONDA_SH"
conda activate VLM || { echo "ERROR: conda activate VLM 失败"; exit 1; }
export PATH="$CONDA_PREFIX/bin:$PATH"
export WANDB_MODE=offline

echo "=== 运行检查脚本 (config=$CONFIG_PATH，登录节点默认不跑前向) ==="
# 登录节点无 GPU/内存紧，默认 --no-forward；有卡时可用: python check_training_setup.py --config $CONFIG_PATH
python check_training_setup.py --config "$CONFIG_PATH" --no-forward
echo ""
echo "通过后可以: sbatch train.slurm $CONFIG_NAME"
