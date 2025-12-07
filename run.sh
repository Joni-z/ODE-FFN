
PRECISION=${PRECISION:-bf16}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-114514}
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
CONFIG_PATH=${CONFIG_PATH:-"./configs/jit_b16_in256.yaml"}

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes $WORLD_SIZE \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    main_jit.py --config $CONFIG_PATH --debug
