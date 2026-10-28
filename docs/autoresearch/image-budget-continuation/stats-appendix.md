# Comparison units, costs and verification

Each arm is one fixed native training trajectory continued from 5k to 20k with model seed 0. Checkpoints, images, pixels, panels and P matrices are not independent trained repetitions. The analysis reports paired descriptive differences and the registered practical allocation margin; it does not infer a population ordering, p-value or training confidence interval from those counts.

The official CIFAR training set retains its original 45,000/5,000 training/validation split. The existing 1,024 random-time validation pairs and 1,024 Gaussian sampling starts are unchanged. FID uses 2,048 validation references, the pinned Inception weights/MPS feature path and identical uint8 conversion. Raw integration states are not clipped; conversion clips for Inception and display.

Training retains batch 64, AdamW 1e-4, betas (.9,.99), zero weight decay, clipping 1 and constant LR. Time is sigmoid(.8z−.8); x_t=tX+(1−t)e; clean prediction uses mean(((D−X)/max(1−t,.05))²). Full Adam, RNG, nonpersistent buffers and consumed-stream state are restored. Both final regression metrics use the exact existing bank.

| Model | Whole trainable parameters | FFN forward matrix MACs/token/block |
|---|---:|---:|
| ODE K1 | 211,094 | 16,512 |
| ODE K2 | 211,094 | 24,704 |
| Swi86 | 211,412 | 16,512 |
| Swi129 | 236,438 | 24,768 |
| SiLU64 | 185,744 | 8,192 |
| SiLU131 | 211,472 | 16,768 |
| SiLU193 | 235,280 | 24,704 |

These are FFN dense-matrix costs, not whole-model wall-time or total FLOPs. K2 performs 49.61% more such work than K1. The fixed 20k margin requires at least 5% lower FID than both declared ordinary references and weighted/clean MSE each no more than 5% above the best corresponding fitting reference. References are Swi86/SiLU131 for K1 and Swi129/SiLU193 for K2. [decision.json](results/decision.json) retains every ratio and its interpretation. No best checkpoint is substituted. The margin passes for K1 and fails for K2; neither result accepts a new method.

The random-time bank has maximum t=.87198, only 25 examples at t≥.7 and none at t≥.9. Five fixed panels of 64 pairs complement it. All panel errors use stored float32 times/deltas, not rounded decimal labels. These panels cannot decompose FID or generated-path error.

Independent training verification ends with exit 0/PASS after 1,884.20 seconds: 112 checkpoints reproduce predictions bitwise; model/Adam/RNG/stream states and 15,000 regenerated continuation batches agree; all 14 full Heun128banks reproduce bitwise. Deep diagnostics cover all four 5k/10k/15k/20k snapshots, including P spectra, clocks and ordinary matrix histories. Public summaries contain 48 P records and 70 fixed-time scores at the two new generation endpoints.

Independent feature verification ends with exit 0/PASS after 215.09 seconds. All 14,336 generated and 2,048 reference feature rows replay bitwise. Separately computed SciPy gesvd FIDs differ from the producer by at most 5.97e-13, below the unchanged 1e-8 limit. Formal files and parent inputs remain unchanged. See [training](results/training-audit.json), [features](results/feature-audit.json) and [actual tool exits](results/tool-exits.json).

The seven continuations take 6,595.00 seconds of cohort wall time on Mac mini CPU, with at most two one-thread workers. This is an execution record, not a matched throughput benchmark. The [ledger](results/results.tsv) records observed per-process RSS in GiB from the primed five-second monitor, not GPU memory or a guaranteed peak. All rows are retained native references; `keep` does not mean candidate acceptance.

The first preparation's preflight failed before formal training because two fixture paths shared CPU Adam-moment storage. The preserved v1 and corrected v2 are separate. Deep-copying the supplied state fixed fixture isolation; actual 5k files and all scientific tolerances stayed unchanged. Fresh documented native/resume checks passed for all seven arms before this cohort began.

The original curation receipt records that reports and visual review were pending at figure-generation time. The completed reports, [visual review](results/visual-review.json) and final artifact-verification receipt resolve those checks without rewriting that historical receipt. Submitted Report1 and prior stopped studies remain unchanged.
