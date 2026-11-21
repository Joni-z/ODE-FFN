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
