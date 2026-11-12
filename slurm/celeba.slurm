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
OUT=/scratch/zz5070/ODE_FFN/runs/celeba/20261111-b4-${FFN}-s0
mkdir -p "$OUT"

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
python -m pip freeze --all > "$OUT/packages-${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -i "${SLURM_JOB_GPUS:-0}" -l 30 > "$OUT/gpu-${SLURM_JOB_ID}.csv" \
    2> "$OUT/gpu-${SLURM_JOB_ID}.err" &
MONITOR_PID=$!
sleep 1
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
    cat "$OUT/gpu-${SLURM_JOB_ID}.err" >&2
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
    --model JiT-B/4 --img_size 64 --class_num 1 \
    --ffn "$FFN" --ode_order 8 --seed 0 \
    --label_drop_prob 1 --cfg 0 --noise_scale 1 --P_mean -0.8 --P_std 0.8 \
    --batch_size 64 --num_workers 2 --epochs 200 --warmup_epochs 5 \
    --lr 0.0001 --lr_schedule constant --weight_decay 0 \
    --ema_decay1 0.9999 --ema_decay2 0.9996 \
    --max_train_seconds 6600 --save_last_freq 1 --log_freq 100 \
    --output_dir "$OUT" "${RESUME[@]}"
