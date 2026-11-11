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
python -m pip freeze --all > "$OUT/packages-${SLURM_JOB_ID}.txt"+nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -i "${SLURM_JOB_GPUS:-0}" -l 30 > "$OUT/gpu-${SLURM_JOB_ID}.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

RESUME=()
if [[ -f "$OUT/checkpoint-last.pth" ]]; then
    RESUME=(--resume "$OUT")
fi

python -u main_jit.py \
    --dataset celeba --data_path /scratch/zz5070/ODE_FFN/datasets \
    --model JiT-B/4 --img_size 64 --class_num 1 \
    --ffn "$FFN" --ode_order 8 --seed 0 \
    --label_drop_prob 1 --cfg 0 --noise_scale 1 --P_mean -0.8 --P_std 0.8 \
    --batch_size 64 --num_workers 2 --epochs 200 --warmup_epochs 5 \
    --lr 0.0001 --lr_schedule constant --weight_decay 0 \
    --ema_decay1 0.9999 --ema_decay2 0.9996 \
    --max_train_seconds 6600 --save_last_freq 1 --log_freq 100 \
    --output_dir "$OUT" "${RESUME[@]}"
