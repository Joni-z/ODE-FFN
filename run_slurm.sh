#!/bin/bash
# 用法: ./run_slurm.sh <config名> [l40s|h200]
# 不写第二项默认用 l40s（节点多，排队一般更短）；要 H200 就写 h200
CONFIG_NAME=${1:?用法: ./run_slurm.sh <config名> [l40s|h200]}
GPU=${2:-l40s}

case "$GPU" in
  l40s)
    PARTITION=l40s_public
    GRES=gpu:l40s:1
    ;;
  h200)
    PARTITION=h200_tandon
    GRES=gpu:h200:1
    ;;
  *)
    echo "未知 GPU 类型: $GPU，用 l40s 或 h200"
    exit 1
    ;;
esac

echo "提交: config=$CONFIG_NAME, partition=$PARTITION, gres=$GRES"
sbatch --partition="$PARTITION" --gres="$GRES" train.slurm "$CONFIG_NAME"
