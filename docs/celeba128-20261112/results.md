# CelebA128: completed 100-epoch comparison

Both seed 0 runs completed 100 epochs/63,500 optimizer updates, with finite
model/two-EMA weights and complete checkpoints. Both used distinct L40S
GPUs on gl066. The frozen source and saved recipe match.

| Measurement | Native SwiGLU | ISRU-D K8 |
|---|---:|---:|
| Epoch 100 training velocity loss | .02993375 | .03155256 |
| Last 5 epoch mean time (s) |113.73 |199.06 |
| Last 5 mean images/s |1429.4 |816.6 |
| Slurm elapsed |3h12m30s |5h35m53s |
| GPU utilization mean after first 5 min |95.73% |98.05% |
| Peak NVML used memory (GiB) |14.81 |22.17 |
| Full-model parameters |130,405,632 |87,879,936 |
| FID50k | Not yet measured | Not yet measured |

At matched exposure, K8's final training loss is 5.41% higher and its steady
epoch time is 1.75 times native. Fewer parameters have not produced cheaper
training: the eighth-order action reuses weights but repeats matrix products.
These are training-cost/fitting observations; generation is a separate
comparison of the saved checkpoints.

![Training loss and epoch time](training-curves.png)

The left panel shows the native training-velocity objective, not a synthetic
oracle field error. The right includes epoch/checkpoint time and initial
compilation. These are within-run curves; no error bars or independent-run
significance test is inferred from epochs.

## Learned P

All 101 K8 snapshots exist, each with 24 operators. Final snapshot values exactly
match the trained checkpoint. Five raw-P milestones give:

| Mean across 24 operators | Initialization | Epoch 100 |
|---|---:|---:|
| Spectral norm |1.1486 |2.1631 |
| Frobenius norm |15.9961 |18.5501 |
| Stable rank |193.95 |76.65 |
| Spectral radius |.5941 |1.1297 |

![Raw P milestones](P-milestones.png)

P grows and its spectrum becomes more concentrated. These averages describe
learned operators, not training repetitions. They do not establish stability
or spectrum of input-dependent PD(h), the complete FFN Jacobian, or the cause
of a final image-quality difference.

## Generation evaluation

No samples or FID existed when training completed. The
[registered evaluation](evaluation.md) uses first EMA .9999, seed 0, author
Heun50 and 50,000 images against the same deterministic training-split crop
reference. It reuses frozen source and checks checkpoint SHA256 before
loading, without additional training, solver sweeps or EMA substitution.

Exact records: [training results/checkpoint identities](training-results.json),
[all epoch records](epochs-results.csv), [P summaries](P-summary.json),
[per-operator milestones](P-milestones.csv) and
[evaluation command check](eval-command-preflight.json).
