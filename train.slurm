#! /bin/bash
#SBATCH --job-name=jit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --mail-user=zz5070@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/zz5070/ODE_FFN/jit-codebase/logs/jit-%j.out
#SBATCH --error=/scratch/zz5070/ODE_FFN/jit-codebase/logs/jit-%j.err

# Conda 激活（本机 miniconda 需先 source conda.sh，再用 conda activate；HPC 文档的 source activate 针对 module anaconda）
CONDA_SH="${CONDA_SH:-/scratch/zz5070/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate VLM || { echo "ERROR: conda activate VLM 失败"; exit 1; }
else
  echo "ERROR: 未找到 conda.sh，请设置 CONDA_SH"
  exit 1
fi
export PATH="$CONDA_PREFIX/bin:$PATH"
# 计算节点上 wandb 常因未登录报 401，先离线记录，之后在登录节点或本机运行 wandb sync 上传
export WANDB_MODE=offline

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