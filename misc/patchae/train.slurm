#! /bin/bash
#SBATCH --job-name=patch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --account=pr_119_tandon_priority

source /vast/hl3797/miniconda/bin/activate
source activate ldit
cd /vast/hl3797/jit-codebase/misc/patchae

PRECISION=${PRECISION:-bf16}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12220}
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
CONFIG_PATH="./$1.yaml"

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes $WORLD_SIZE \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    train.py --config $CONFIG_PATH > logs/$1.log 2>&1