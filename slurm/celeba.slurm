#!/bin/bash
#SBATCH --job-name=celeba-isrud-k8-s0
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|l40s|rtx6000|b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/zz5070/ODE_FFN/runs/celeba/slurm/%x-%j.out

set -euo pipefail
cd "${ODE_FFN_CODE:-/scratch/zz5070/ODE_FFN/ODE-FFN}"
source scripts/activate_torch.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1
# This is one Python process, not a torchrun/srun DDP launch.
unset RANK WORLD_SIZE LOCAL_RANK SLURM_PROCID

FFN=${1:-isru_d}
case "$FFN" in isru_d|swiglu) ;; *) exit 2 ;; esac
RECIPE=${2:-celeba64}
case "$RECIPE" in
    celeba64)
        MODEL=JiT-B/4; IMAGE_SIZE=64; BATCH_SIZE=64; NOISE_SCALE=1
        LR_ARGS=(--lr 0.0001)
        OUT=/scratch/zz5070/ODE_FFN/runs/celeba/20261111-b4-${FFN}-s0
        ;;
    celeba128)
        MODEL=JiT-B/16; IMAGE_SIZE=128; BATCH_SIZE=${ODE_FFN_BATCH_SIZE:-256}; NOISE_SCALE=0.5
        LR_ARGS=(--blr 0.00005)
        OUT=/scratch/zz5070/ODE_FFN/runs/celeba/20261112-b16-${FFN}-s0
        python - "$BATCH_SIZE" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path
path = Path('/scratch/zz5070/ODE_FFN/operations/celeba128-20261112/cuda-readiness.json')
record = json.loads(path.read_text())
assert record['status'] == 'PASS' and record['image_size'] == 128
assert record['source_commit'] == sys.argv[2], 'Readiness must match the frozen training source'
for arm in ('swiglu', 'isru_d'):
    assert record['full_JiT_B16'][arm]['batch'] == int(sys.argv[1])
PY
        ;;
    *) echo "Unknown CelebA recipe" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
AUTO_RESUME=${ODE_FFN_AUTO_RESUME:-0}
FRESH_START=${ODE_FFN_FRESH_START:-0}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
MAX_REQUEUES=${ODE_FFN_MAX_REQUEUES:-20}
ATTEMPT=${SLURM_JOB_ID}-r${RESTART_COUNT}
if (( RESTART_COUNT > MAX_REQUEUES )); then
    echo "Requeue limit reached; checkpoint retained" >&2
    exit 2
fi
START_EPOCHS=0
if [[ "$AUTO_RESUME" == 1 && ! -f "$OUT/checkpoint-last.pth" ]]; then
    if [[ "$FRESH_START" != 1 || "$RESTART_COUNT" != 0 || -f "$OUT/source-commit.txt" ]]; then
        echo "Continuation requires a checkpoint; fresh start must be explicit and is allowed only once" >&2
        exit 2
    fi
fi
if [[ -f "$OUT/checkpoint-last.pth" ]]; then
    START_EPOCHS=$(python - "$OUT/checkpoint-last.pth" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)
print(state['epoch'] + 1)
PY
)
fi
echo "Allocation $ATTEMPT resumes after $START_EPOCHS completed epochs"

if [[ -f "$OUT/source-commit.txt" ]]; then
    [[ "$(cat "$OUT/source-commit.txt")" == "$(git rev-parse HEAD)" ]] || {
        echo "Source differs from this run's recorded commit; stop before resuming" >&2
        exit 2
    }
else
    git rev-parse HEAD > "$OUT/source-commit.txt"
fi
git diff --quiet
git diff --cached --quiet
python -m pip freeze --all > "$OUT/packages-${ATTEMPT}.txt"
# Slurm global GPU ordinals need not match this job's visible device ordinals.
GPU_UUID=$(python -c 'import torch; print("GPU-" + str(torch.cuda.get_device_properties(0).uuid))')
echo "Monitoring CUDA device $GPU_UUID; Slurm global IDs ${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -i "$GPU_UUID" -l 30 > "$OUT/gpu-${ATTEMPT}.csv" \
    2> "$OUT/gpu-${ATTEMPT}.err" &
MONITOR_PID=$!
sleep 1
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
    cat "$OUT/gpu-${ATTEMPT}.err" >&2
    echo "GPU monitoring failed to start; stop before training" >&2
    exit 1
fi
LOCAL_DATA=$(mktemp -d "/tmp/ode-celeba-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
    kill "$MONITOR_PID" 2>/dev/null || true
    if [[ "$LOCAL_DATA" == /tmp/ode-celeba-${SLURM_JOB_ID}.* ]]; then
        rm -rf -- "$LOCAL_DATA"
    fi
}
trap cleanup EXIT

# Read one archive from scratch; keep per-image reads on the compute node.
DATA_SOURCE=/scratch/zz5070/ODE_FFN/datasets/celeba
mkdir -p "$LOCAL_DATA/celeba"
cp "$DATA_SOURCE"/{list_attr_celeba.txt,identity_CelebA.txt,list_bbox_celeba.txt,list_landmarks_align_celeba.txt,list_eval_partition.txt} "$LOCAL_DATA/celeba/"
python -m zipfile -e "$DATA_SOURCE/img_align_celeba.zip" "$LOCAL_DATA/celeba"

RESUME=()
if [[ -f "$OUT/checkpoint-last.pth" ]]; then
    RESUME=(--resume "$OUT")
fi

python -u main_jit.py \
    --dataset celeba --data_path "$LOCAL_DATA" \
    --model "$MODEL" --img_size "$IMAGE_SIZE" --class_num 1 \
    --ffn "$FFN" --ode_order 8 --seed 0 \
    --label_drop_prob 1 --cfg 0 --noise_scale "$NOISE_SCALE" --P_mean -0.8 --P_std 0.8 \
    --batch_size "$BATCH_SIZE" --num_workers 2 --epochs 200 --warmup_epochs 5 \
    "${LR_ARGS[@]}" --lr_schedule constant --weight_decay 0 \
    --ema_decay1 0.9999 --ema_decay2 0.9996 \
    --max_train_seconds 6600 --save_last_freq 1 --log_freq 100 \
    --output_dir "$OUT" "${RESUME[@]}"

# Continue only after a clean budget pause; a failed Python command stops above.
if [[ "$AUTO_RESUME" == 1 ]]; then
    read -r STATE COMPLETED TARGET < <(python - "$OUT/progress.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    progress = json.load(stream)
print(progress['status'], progress['completed_epochs'], progress['target_epochs'])
PY
)
    if [[ "$STATE" == paused ]] && (( COMPLETED < TARGET )); then
        if (( COMPLETED <= START_EPOCHS )); then
            echo "No completed-epoch progress; stop instead of requeueing" >&2
            exit 2
        fi
        if (( RESTART_COUNT >= MAX_REQUEUES )); then
            echo "Requeue limit reached; checkpoint retained at epoch $COMPLETED" >&2
            exit 2
        fi
        echo "Saved epoch $COMPLETED/$TARGET; requeueing job $SLURM_JOB_ID"
        cleanup
        trap - EXIT
        scontrol requeue "$SLURM_JOB_ID"
    elif [[ "$STATE" != complete ]] || (( COMPLETED != TARGET )); then
        echo "Unexpected progress state; stop before requeueing" >&2
        exit 2
    fi
fi
