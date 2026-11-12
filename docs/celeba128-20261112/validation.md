# Torch invocation and validation

The software environment is unchanged: Python 3.10.22, PyTorch 2.7.1+cu128,
spec hash `d8a61f4e`. Reuse the scratch venv; do not reinstall packages.
Source sync uses GitHub main. Formal jobs use the clean detached runtime
`/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112`.
Submit the corrected Slurm wrappers from the active checkout. Slurm stores
their content at submission; `ODE_FFN_CODE` keeps training/model source
frozen at65864ad independently of the wrapper revision.

After connecting with `ssh torch`, the following invocation checks 128px data,
the K8 polynomial and derivatives, bitwise epoch resume through the real
entrypoint, and rejection of a changed noise recipe:

```bash
cd /scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112
source scripts/activate_torch.sh
python scripts/check_celeba.py --device cpu --image-size 128 --output /scratch/zz5070/ODE_FFN/operations/celeba128-20261112/documented-cpu.json
```

Expected sentinel: `CELEBA_K8_CPU_PASS`. The tiny CPU model is a correctness
fixture, not an image benchmark or throughput measurement.

The resource request can be checked without allocating:

```bash
sbatch --test-only /scratch/zz5070/ODE_FFN/ODE-FFN/slurm/celeba_smoke.slurm 128
sbatch --test-only /scratch/zz5070/ODE_FFN/ODE-FFN/slurm/celeba.slurm swiglu celeba128
```

Submit readiness once with the frozen runtime in `ODE_FFN_CODE`:

```bash
sbatch --parsable --job-name=celeba128-readiness --output=/scratch/zz5070/ODE_FFN/operations/celeba128-20261112/cuda-%j.out --export=ALL,ODE_FFN_CODE=/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112,ODE_FFN_BATCH_SIZE=256 /scratch/zz5070/ODE_FFN/ODE-FFN/slurm/celeba_smoke.slurm 128
```

Its GPU witness runs both full JiT-B/16 FFNs with batch 256, three BF16
optimizer/EMA updates each, finite loss/gradients and the exact noise scale.
It records total GPU memory, peak allocated memory, all three step times and
warm-step throughput, with a separate GPU CSV. It writes
`operations/celeba128-20261112/cuda-readiness.json` with status PASS and
`CELEBA_K8_CUDA_PASS`. This short profile includes initialization/compilation
cost separately from the two warmed steps; measured training throughput is
still collected from actual epochs.

After the passing profile fixes the shared batch, the two training commands
use `/scratch/zz5070/ODE_FFN/ODE-FFN/slurm/celeba.slurm` with arguments
`swiglu celeba128` and `isru_d celeba128`, an `afterok` dependency on readiness,
and `--kill-on-invalid-dep=yes`. Pass the same batch plus
`ODE_FFN_AUTO_RESUME=1,ODE_FFN_FRESH_START=1,ODE_FFN_MAX_REQUEUES=20`.
The launcher independently checks the receipt's source and both arm batches.
Do not call a pending job running or a budget pause finished training.

Run monitoring from the active checkout after a launch receipt exists:

```bash
python scripts/monitor_slurm.py --launch /scratch/zz5070/ODE_FFN/operations/celeba128-20261112/launch.json --interval 60
```

Each new allocation uses a distinct telemetry/package filename containing
jobID and restart count. Slurm stdout appends across requeues. Old 64px outputs
and finalized reports remain unchanged.
Telemetry selects the allocated CUDA GPU UUID. Slurm's global GPU ordinals
need not match visible NVML indices; passing them directly caused the first
readiness attempt to stop before training. The corrected wrapper prefixes
PyTorch's raw UUID with `GPU-` for nvidia-smi. The short diagnostic confirmed
CUDA works while global ID1 is not a visible NVML index. See the
[Slurm variable definition](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_JOB_GPUS).
