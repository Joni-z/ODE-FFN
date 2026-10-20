#! /bin/bash
#SBATCH --job-name=fid_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1

# 尽量小：更容易调度
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=04:00:00

# 需要 1 张 GPU
#SBATCH --gres=gpu:1

# 必须指定 account
#SBATCH --account=torch_pr_63_tandon_advanced

# 邮件别用 ALL（会很吵），一般 END/FAIL 就够
#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --output=/scratch/zz5070/ODE_FFN/jit-codebase/logs/fid_stats-%j.out
#SBATCH --error=/scratch/zz5070/ODE_FFN/jit-codebase/logs/fid_stats-%j.err

set -euo pipefail

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
mkdir -p logs fid_stats

# 让脚本参数跟资源一致（workers 不要超过 cpus-per-task）
# batch 过大易 OOM（GPU/内存）；50 为 FID 常用默认，若仍 OOM 可改为 32 或 24
BATCH_SIZE=${BATCH_SIZE:-50}
NUM_WORKERS=${NUM_WORKERS:-4}

python src/util/generate_fid_stats_imagenet.py \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"

echo "Done. Output: fid_stats/jit_in256_stats.npz"
