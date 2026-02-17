#! /bin/bash
#SBATCH --job-name=jit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
# GPU 二选一：不传则用下面默认；传参覆盖，例如 sbatch --partition=l40s_public --gres=gpu:l40s:1 train.slurm jit_b16_in256_imagenet
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
# 要跑 H200 时提交加：--partition=h200_tandon --gres=gpu:h200:1
#SBATCH --account=torch_pr_63_tandon_advanced

# 激活 conda 环境 VLM（若 conda 不在默认路径，请修改下面两行）
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate VLM

cd /scratch/zz5070/ODE_FFN/jit-codebase

# 配置名：sbatch train.slurm jit_b16_in256_imagenet 则用 configs/jit_b16_in256_imagenet.yaml
CONFIG_NAME=${1:-jit_b16_in256_imagenet}
CONFIG_PATH="./configs/${CONFIG_NAME}.yaml"
mkdir -p logs

PRECISION=${PRECISION:-bf16}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12017}
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

echo "Config: $CONFIG_PATH | Log: logs/${CONFIG_NAME}.log"

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes $WORLD_SIZE \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    main_jit.py --config "$CONFIG_PATH" 2>&1 | tee "logs/${CONFIG_NAME}.log"