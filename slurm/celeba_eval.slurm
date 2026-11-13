#!/bin/bash
#SBATCH --job-name=celeba128-fid50k
#SBATCH --account=torch_pr_637_tandon_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|l40s|rtx6000|b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/zz5070/ODE_FFN/runs/celeba/slurm/%x-%j.out

set -euo pipefail
cd "${ODE_FFN_CODE:-/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112}"
source scripts/activate_torch.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1
unset RANK WORLD_SIZE LOCAL_RANK SLURM_PROCID

FFN=${1:?Pass swiglu or isru_d}
case "$FFN" in swiglu|isru_d) ;; *) exit 2 ;; esac
TRAIN=/scratch/zz5070/ODE_FFN/runs/celeba/20261112-b16-${FFN}-s0
OUT=/scratch/zz5070/ODE_FFN/runs/celeba/evaluations/20261113-b16-${FFN}-s0-ckpt100-ema9999
mkdir -p "$OUT"
if [[ -f "$OUT/result.json" ]]; then
    echo "Evaluation already completed; preserve its result"
    exit 0
fi
[[ ! -f "$OUT/evaluation-start.json" ]] || {
    echo "Existing incomplete evaluation requires inspection before a restart" >&2
    exit 2
}
git diff --quiet
git diff --cached --quiet
python - "$TRAIN" "$OUT" "$FFN" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
train, out, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert (train/'source-commit.txt').read_text().strip() == source
evidence = json.loads(Path('/scratch/zz5070/ODE_FFN/operations/celeba128-20261112/training-results.json').read_text())
checkpoint = evidence['arms'][arm]['checkpoint']
assert checkpoint['completed_epochs'] == 100 and checkpoint['bytes'] == (train/'checkpoint-last.pth').stat().st_size
assert evidence['source_commit'] == source
digest = hashlib.sha256()
with (train/'checkpoint-last.pth').open('rb') as stream:
    for chunk in iter(lambda: stream.read(16*1024*1024), b''):
        digest.update(chunk)
assert digest.hexdigest() == checkpoint['sha256']
record = dict(ffn=arm, job_id=os.environ['SLURM_JOB_ID'], checkpoint=checkpoint['path'],
              checkpoint_sha256=checkpoint['sha256'], checkpoint_epochs=100, source_commit=source,
              checkpoint_mtime_ns=(train/'checkpoint-last.pth').stat().st_mtime_ns,
              ema_decay=.9999, image_count=50000, generation_batch=256, seed=0,
              sampler='author Heun50 with final Euler', cfg=0,
              reference='official CelebA train162770, author deterministic128px crop', formal_training=False)
(out/'evaluation-start.json').write_text(json.dumps(record, indent=2)+'\n')
(out/'progress.json').write_text(json.dumps(dict(status='running', phase='sampling_then_FID50k', **record), indent=2)+'\n')
PY

GPU_UUID=$(python -c 'import torch; print("GPU-" + str(torch.cuda.get_device_properties(0).uuid))')
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -i "$GPU_UUID" -l 30 > "$OUT/gpu-${SLURM_JOB_ID}.csv" \
    2> "$OUT/gpu-${SLURM_JOB_ID}.err" &
MONITOR_PID=$!
sleep 1
kill -0 "$MONITOR_PID"
LOCAL_DATA=$(mktemp -d "/tmp/ode-celeba-eval-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR="$LOCAL_DATA"
cleanup() {
    kill "$MONITOR_PID" 2>/dev/null || true
    if [[ "$LOCAL_DATA" == /tmp/ode-celeba-eval-${SLURM_JOB_ID}.* ]]; then
        rm -rf -- "$LOCAL_DATA"
    fi
}
trap cleanup EXIT
DATA_SOURCE=/scratch/zz5070/ODE_FFN/datasets/celeba
mkdir -p "$LOCAL_DATA/celeba"
cp "$DATA_SOURCE"/{list_attr_celeba.txt,identity_CelebA.txt,list_bbox_celeba.txt,list_landmarks_align_celeba.txt,list_eval_partition.txt} "$LOCAL_DATA/celeba/"
python -m zipfile -e "$DATA_SOURCE/img_align_celeba.zip" "$LOCAL_DATA/celeba"

# Same frozen author entrypoint/objective/sampler/metric, with generation only.
python -u main_jit.py \
    --dataset celeba --data_path "$LOCAL_DATA" --model JiT-B/16 --img_size 128 --class_num 1 \
    --ffn "$FFN" --ode_order 8 --seed 0 --label_drop_prob 1 --cfg 0 \
    --noise_scale 0.5 --P_mean -0.8 --P_std 0.8 --batch_size 256 --num_workers 4 \
    --epochs 100 --warmup_epochs 5 --lr 0.00005 --lr_schedule constant --weight_decay 0 \
    --ema_decay1 0.9999 --ema_decay2 0.9996 --max_train_seconds 0 \
    --resume "$TRAIN" --output_dir "$OUT" --evaluate_gen --sampling_method heun \
    --num_sampling_steps 50 --gen_bsz 256 --num_images 50000 --keep_samples \
    2>&1 | tee "$OUT/evaluation.log"

python - "$OUT" <<'PY'
import json, math, re, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out = Path(sys.argv[1])
record = json.loads((out/'evaluation-start.json').read_text())
assert Path(record['checkpoint']).stat().st_mtime_ns == record['checkpoint_mtime_ns']
events = EventAccumulator(str(out)).Reload()
printed = re.findall(r'FID: ([0-9.eE+-]+), Inception Score: ([0-9.eE+-]+)', (out/'evaluation.log').read_text())
assert printed, 'The author entrypoint did not report FID'
fid, isc = map(float, printed[-1])
metric_source = 'author torch-fidelity result printed to the original log'
tags = events.Tags()['scalars']
fid_tags, is_tags = [t for t in tags if t.startswith('fid')], [t for t in tags if t.startswith('is')]
if fid_tags and is_tags:
    fid_tb, isc_tb = events.Scalars(fid_tags[-1])[-1].value, events.Scalars(is_tags[-1])[-1].value
    assert abs(fid_tb-fid) < .0002 and abs(isc_tb-isc) < .0002
    fid, isc = fid_tb, isc_tb
    metric_source = 'author torch-fidelity result logged to TensorBoard, checked against original console output'
assert math.isfinite(fid) and math.isfinite(isc)
folders = [p for p in out.iterdir() if p.is_dir() and p.name.startswith('heun-')]
assert len(folders) == 1 and len(list(folders[0].glob('*.png'))) == 50000
record.update(status='complete', FID50k=fid, inception_score=isc,
              metric_source=metric_source,
              sample_directory=str(folders[0]), inner_epoch_label='upstream evaluate_gen uses epoch0000; loaded checkpoint is100')
(out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
(out/'progress.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
PY
