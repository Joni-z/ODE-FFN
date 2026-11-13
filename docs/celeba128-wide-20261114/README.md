# Parameter-matched ISRU-D: CelebA128

Authorized 14 November 2026. Keep the completed native SwiGLU baseline fixed.
Train widened ISRU-D K8 and K1, seed 0, 100 epochs, FID50k at 20/40/60/80/100.

## Baseline decision

The native JiT-B/16 baseline qualifies for this project's CelebA128 comparison:
FID50k 35.9178543, below the registered 50 gate, with verified original FFN,
loss, sampling, dataset and checkpoint bindings. This is our CelebA recipe;
the author supplies no matching CelebA benchmark score. Retain its completed
checkpoint100, source65864ad and FID; no baseline retraining or narrowing.
See [results](../celeba128-20261112/results.md) and
[integrity checks](../celeba128-20261112/EXPERIMENT_AUDIT.md).

## Hypothesis and exact model

Test whether restoring FFN parameter capacity makes the selected K8 useful,
and whether high order helps relative to K1 at that same capacity. Both arms:

```text
h (768) -> bias-free Win (1200)
        -> T_K with independent P1 (1200 x 1200)
        -> SiLU
        -> T_K with independent P2 (1200 x 1200)
        -> bias-free Wout (768)
T_K(x) = sum_{j=0..K} [P diag((1+1.2*x^2)^(-1/2))]^j x / j!
```

Compute each entrance diagonal once, retain its input gradients, and reuse
it across all Taylor terms. K1 and K8 differ only in K. P uses the existing
Linear Kaiming-uniform initializer. New entrance/exit projections use Xavier
uniform; all four matrices are bias-free. Both arms have identical initial
weights. RNG isolation preserves shared backbone weights and training RNG
relative to native JiT. This is an expanded FFN, including new projections;
comparison with the old unexpanded K8 is not a pure width-only intervention.

| Model | Total parameters (includes fixed positional embeddings) | FFN dense forward MAC/token |
|---|---:|---:|
| Fixed native SwiGLU |130,405,632 |4,718,592 |
| Widened K1 |130,402,560 |4,723,200 |
| Widened K8 |130,402,560 |24,883,200 |

The parameter difference is 3,072 (0.00236%). K1 approximately matches native
FFN matrix work; K8 costs 5.27 times as much within the FFN. Measure full-model
wall time and quality together. These MACs are not whole-model training FLOPs.

## Fixed protocol and decision

Use the original author-derived JiT-B/16, 128px, 12 blocks, token width768,
patch16, input bottleneck128, unconditional labels, official CelebA train162770
and author crop/random flip. Batch256, 635 drop-last updates/epoch, 63,500
updates total. AdamW5e-5, betas(.9,.95), weight_decay0, warmup5 then constant;
noise.5, logit time mean-.8/std.8, t_eps.05. Keep per-update EMA.9999/.9996.

FID uses first EMA.9999, seed0, standard pinned torch-fidelity, 50k images,
native Heun50 including final Euler, CFG0 and the same deterministic training
reference. Retain full numbered checkpoints0020/0040/0060/0080/0100, last full
checkpoint each epoch and every P snapshot. Evaluation preserves training RNG.
The registered comparison is checkpoint100. Stop at100 or an execution failure;
do not select the best intermediate FID as the primary endpoint.

K8 beating K1 would support a contribution of high order in the expanded
parameterization. Beating native would establish a quality result for this
recipe; the extra compute still determines its cost. If K1 matches or beats
K8, evaluate whether K8's extra work is justified. No additional optimizer,
spectral penalty, activation change or ordinary-baseline change is selected.

## Deployment and monitoring

Environment: reuse `/scratch/zz5070/ODE_FFN/envs/jit-celeba-cu128`, specd8a61f4e.
New outputs: `/scratch/zz5070/ODE_FFN/runs/celeba/20261114-b16-isrud-w1200-k{1,8}-s0`.
Operations and readiness: `/scratch/zz5070/ODE_FFN/operations/celeba128-wide-20261114`.
The clean source runtime and actual job/time limits are recorded at launch.
GPU telemetry every30seconds; persistent queue/progress/FID/checkpoint monitor
every60seconds. No automatic restarts. All six registered GPU types remain
eligible under the user's priority account.

## Validation commands

From the clean runtime on Torch, after `source scripts/activate_torch.sh`:

```bash
python tests/check_wide_ffn.py --device cpu --output /scratch/zz5070/ODE_FFN/operations/celeba128-wide-20261114/cpu-check.json
```

Expected `WIDE_FFN_CPU_PASS`: dense polynomial and all input/parameter
derivatives, gradcheck, preserved original-width state/RNG/output, exact full
parameter counts, identical K1/K8 initialization and shared backbone/RNG.

Submit one short readiness allocation with the new runtime in `ODE_FFN_CODE`:

```bash
sbatch --parsable --export=ALL,ODE_FFN_CODE=/scratch/zz5070/ODE_FFN/runtimes/celeba128-wide-20261114 slurm/celeba_smoke.slurm 128-wide
```

It profiles both full models at batch256, including optimizer, two EMAs,
finite gradients and native Heun50 generation. Use measured train/sample
throughput plus existing feature-extraction/data-loading timings to set the
continuous training limits, including all five FIDs. Formal launch uses
`slurm/celeba.slurm isru_d celeba128-wide 1` or `8`, an explicit fresh-start
flag and the same frozen runtime. The matching readiness receipt is required.

## Submitted execution

CUDA readiness23013985 passed on H100 gh003 at batch256: K1/K8 warmed train
steps .05820/.14624seconds, Heun50 generation3.603/9.264seconds per256images,
training peaks13.07/29.90GiB. Weights, gradients, samples and both new
projections are finite and receive gradients. This is a5update readiness
profile, not formal convergence evidence.

Frozen source17b61a56af1174120e06cd01e25266625c277ce4. Formal jobs:
K8 **23015406**, K1 **23015408**. Both use the registered100epoch protocol.
The persistent monitor PID is460860; GPU telemetry is attached by each job.
[Launch commands](launch.json), [readiness](cuda-readiness.json),
[budget arithmetic](budget.json), [snapshot](monitor.json).

The H100 profile projects whole runs near8h for K8 and6h for K1, including
five50k FIDs, a measured data-throughput floor, feature extraction/IO, startup
and margin. Different GPU types remain eligible, so initial limits20h/9h
cover the slower-card projection. K8 obtained H100 gh011; its limit was then
kept at20h because the cluster rejected a requested reduction to10h. No fixed
runtime pause or automatic requeue is planned. These
are estimates and upper limits; monitor actual epochs and completed FIDs.

Both formal jobs are now running: K8 on H100 gh011, K1 on L40S gl068.
Use the paired H100 readiness numbers for same-hardware speed comparisons;
wall times from these distinct training GPUs describe execution separately.
First formal FIDs are due after20completed epochs.
