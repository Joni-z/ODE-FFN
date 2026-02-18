#! /bin/bash
#SBATCH --job-name=fid_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/ODE_FFN/jit-codebase/logs/fid_stats-%j.out
#SBATCH --error=/scratch/zz5070/ODE_FFN/jit-codebase/logs/fid_stats-%j.err

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

# 生成 fid_stats/jit_in256_stats.npz（默认用 imagenet/train）
# 若需覆盖已有文件加: --overwrite
python src/util/generate_fid_stats_imagenet.py \
    --batch-size 100 \
    --num-workers 8

echo "Done. Output: fid_stats/jit_in256_stats.npz"
