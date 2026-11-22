# CelebA128 midpoint transfer

One JiT-B/16 trajectory tests whether the midpoint FFN improves face generation at the same training exposure as native SwiGLU. The primary comparison is FID50k at epoch 160. Each numbered checkpoint also tests whether halving the internal steps preserves FID at higher throughput.

The FFN projects 768 channels to 1,200, integrates four explicit midpoint steps over unit internal time, applies SiLU, integrates a second independent four-step map, and projects back to 768. Each map shares its matrix across all stages and refreshes the ISRU response at each evaluation. This preserves the parameter layout of the previous wide image candidate. Training recomputes the local solve during backward to save GPU memory.

- Seed 0, official CelebA train split, 128px, batch 256, 635 updates per epoch, unconditional JiT. AdamW at 0.00005 with five warmup epochs, then constant LR. Other settings come from the original CelebA128 wrapper.
- Every 20 epochs, reopen the numbered checkpoint and use EMA 0.9999 for both inner solvers. Use seed 0, the original Heun50, CFG 0, 50,000 generated images and the same deterministic CelebA training reference. Save each FID, checkpoint hash and timing in the epoch comparison JSON.
- Measure sampling at batch 256 on the same GPU, with one warmup batch and the median of three synchronized batches. Time includes the full outer sampler and excludes PNG writing and Inception. Record GPU UUID and raw seconds.
- Native SwiGLU has full checkpoints and FID at epochs 100, 120, 140 and 160. At those epochs, reuse its FID and measure its EMA sampling throughput on the candidate's GPU. The native checkpoints for 20 through 80 were never retained. Those rows contain the two candidate results.
- Four 40-epoch Slurm allocations advance the same checkpoint to 160. The learning rate is constant after warmup. Each allocation requests one GPU on the priority account, six CPUs and 12 hours. The estimate uses the earlier L40S K8 continuation (11.1 hours for 60 epochs and three FIDs), allows for four FIDs per 40 epochs and recomputation in backward. GPU telemetry runs every 30 seconds and a login-node monitor records queue state, epochs, checkpoints and paired results every minute.

[FFN](../../../ffn.py) · [Paired evaluation](../../../util/midpoint_eval.py) · [Launch](../../../scripts/submit_celeba_midpoint.py) · [Historical baseline](../../celeba128-wide-20261114/continuation-160/matched-FID.csv)

## Torch launch

Training source `603c353` passed the CPU midpoint/EMA check in the Torch environment. Slurm accepted jobs 23992197, 23992198, 23992199 and 23992200 for epochs 1–40, 41–80, 81–120 and 121–160 respectively. One GPU is eligible from a100, h100, h200, l40s, rtx6000 and b200 under `torch_pr_637_tandon_priority`. The last three jobs have after-success dependencies.

The persistent monitor runs on Torch (PID 2427363). Live paths under `/scratch/zz5070/ODE_FFN/`:

- `operations/celeba128-midpoint-20261122/monitor.json` contains the latest scheduler and metric snapshot.
- `runs/celeba/20261122-midpoint4-w1200-s0/progress.json` tracks training and evaluation once allocated.
- `runs/celeba/20261122-midpoint4-w1200-s0/paired/epoch-NNNN/comparison.json` contains each finished FID and same-device speed comparison.

[Launch receipt](launch.json)

## Where the GPU time went

On the L40S, job 23992197 took 11 h 49 min for epochs 1–40. From the
`Evaluating checkpoint` line to `PAIRED_MILESTONE`, the two FID evaluations
at epoch 20 took 220.4 minutes together. At epoch 40 the pair took 213.8
minutes. These two checkpoints account for 61.2% of the allocation. The
remaining 4 h 35 min includes training, staging and checkpoint writes.
A steady training epoch took 407.8 s on the L40S and 257.5 s on the next
A100 (median, epochs 2–39 excluding 20, and 42–59 respectively).
The epoch timer includes online evaluation, so it understates training
throughput on milestone rows.

The original SwiGLU ran 100 epochs in 3 h 12 min on one L40S, with online
FID disabled. Its last five epochs averaged 113.7 s. Both logs show JiT-B/16,
128px, batch 256, 635 updates per epoch, AdamW at 0.00005, five warmup
epochs and seed 0. The midpoint has 130.353M parameters and SwiGLU has
130.406M. Each training FFN applies a 1,200 by 1,200 matrix 16 times across
its two four-step solves. Backpropagation recomputes the solves to save
memory. These differences matter even at matched parameter counts.
[Original native run](../../celeba128-20261112/results.md)

The unconditional sampler also does avoidable work. JiT's Heun50 grid
evaluates velocity 99 times per batch (49 Heun steps and one final Euler
step). The largest evaluation time is 0.98. With CFG 0 throughout that grid,
the conditional branch has weight zero on every call. The original sampler
nevertheless executes both branches, or 198 network forwards. A seed-0 CPU
replay with a small nonzero JiT and distinct conditional and unconditional
predictions produced exactly the same final tensors with 99 forwards for
SwiGLU and both midpoint settings. The L40S timing above is for the original
198-forward path. The CUDA replay below measures the single-branch path.
[Replay the Heun50 check](../../../tests/check_unconditional_sampler.py)

## Detached evaluation on Torch

On one dedicated A100, the original and single-branch Heun50 samplers generated
identical CUDA/BF16 outputs from the same seed-0 batch of 256 faces. The
midpoint modes used the same epoch-60 EMA. SwiGLU used its epoch-100 EMA.
Times are medians of two synchronized batches after one warmup per path.

| FFN | Original, s | Single branch, s | Speedup | Largest output difference |
|---|---:|---:|---:|---:|
| Two-step midpoint | 13.712 | 6.858 | 1.999× | 0 |
| Four-step midpoint | 22.319 | 11.158 | 2.000× | 0 |
| SwiGLU | 7.232 | 3.615 | 2.000× | 0 |

The continuation starts at epoch 95 after 60,325 AdamW
updates. A hard link preserves the complete checkpoint, including both EMA
copies and the RNG state. Each training job writes an immutable milestone.
The next training job and the paired FID50k job can start once that
milestone has been saved. Both midpoint modes and native timing use the
same evaluation GPU. Raw outputs, GPU telemetry and per-epoch comparison
records are under `runs/celeba/20261123-midpoint4-detached-s0` on Torch.
Previous milestones remain under `runs/celeba/20261122-midpoint4-w1200-s0`.

[CUDA timings and equality](cuda-check.json) · [Checkpoint handoff](handoff.json) · [Slurm dependencies](detached-launch.json)
