# ISRU transfer to unconditional CIFAR-10 — Report2

The user requested this transfer on October 29, 2026, and selected **Mac mini
only**. This study tests the existing free-matrix ISRU design; it introduces
no adaptive clock, spectral penalty, learned solver, or new activation.
Report1 is final and remains untouched.

## Question and controls

Does the synthetic Product16 result transfer to image denoising/generation
when ordinary FFNs receive a larger backbone and meaningful training?
The synthetic result was task-specific. ISRU is a candidate, not an established
best architecture. The previous 211k-parameter, 20k-step image screen did not
establish a competent ordinary baseline and cannot answer this question.

All arms share a pixel JiT body: 32x32 RGB, patch4 (64 tokens), width192,
six blocks, three attention heads, patch embedding dimension128, no class
information, dropout or context tokens. Model seed0; every non-FFN tensor is
identical at initialization. The model takes a noisy image and diffusion
time, and predicts the clean image. A constant label0 only satisfies the
existing interface; CIFAR labels are never read.

| FFN | Actual hidden size | Total trainable parameters | Dense FFN MACs/token/block |
|---|---:|---:|---:|
| ISRU | 192 | 2,870,064 | 73,728 |
| SwiGLU matched | 128 | 2,872,752 | 73,728 |
| SwiGLU wide | 512 | 4,204,464 | 294,912 |

MACs cover FFN matrix work, not pointwise operations, the whole backbone, or
wall-clock time. ISRU uses `W2 (u / sqrt(1+1.2 u^2))`, `u=W1 h`, with independent
bias-free square matrices. Each starts at `I + .3 P + .045 P^2`, using a fresh
native professor-layer P initializer, exactly as in the synthetic FreeISRU.
Both matrices then train freely: there is no uniquely recoverable trained P.
This transfers an architecture/initialization, not synthetic trained weights.
SwiGLU uses the native Xavier/zero-bias initialization; the comparison tests
the complete design and does not isolate activation from initialization.

The manual-attention implementation is replaced locally by ordinary PyTorch
SDPA for speed, identically in all arms. The preflight compares active full
forward, input and parameter gradients with the original attention. The
unused SwiGLU auxiliary-statistic calculation is omitted; its actual forward
and gradients are checked against the native module. Historical `src/` is
not edited.

## Fixed training protocol

[config.json](config.json) is the executable registration. One stream supplies
the same examples, flips, Gaussian noise and times to all arms, interleaving
their updates on one MPS process. AdamW, batch512, LR1e-4, betas(.9,.95),
weight decay0, linear500-update warmup then constant LR, clip norm1, one
EMA(.9999), float32. The cap is **50,000 updates per arm**:25.6 million image
presentations, approximately569 passes over the45,000-image training split.
This is a budget, not a convergence certificate. No best-checkpoint or
best-EMA selection. Stream seed100000, validation seed200000 and sampling
seed300000 identify shared data banks; they are not repeated model seeds.

For normalized clean image x and Gaussian e: `z=t*x+(1-t)*e`,
`t=sigmoid(.8*N(0,1)-.8)`. Train clean prediction D by
`mean(((D(z,t)-x)/max(1-t,.05))^2)`, the corresponding velocity loss.
One random horizontal flip is shared across arms. A frozen official-data
manifest/split defines45k train and5k validation; the official test set is
not used. Validation uses fixed1024 held-out noisy examples every500 updates.

The time distribution, loss, batch-scaled LR, betas, constant/warmup schedule,
horizontal flip, EMA and sampler follow the primary
[official JiT training source](https://github.com/LTH14/JiT/blob/main/main_jit.py),
[denoiser](https://github.com/LTH14/JiT/blob/main/denoiser.py), and
[LR scheduler](https://github.com/LTH14/JiT/blob/main/util/lr_sched.py).
The CIFAR backbone, budget and clipping are explicit project adaptations;
this is not an official CIFAR reproduction or a claim of ImageNet-scale
training adequacy.

## Evaluation and interpretation

At2k,5k,10k,25k,50k, preserve named full checkpoints, raw/EMA image grids and
all FFN matrix singular-value summaries. For ISRU these are W1 and W2
trajectories, not fictional trained-P eigenvalues. Sampling uses the same
Gaussian starts, linear50-step schedule,49 Heun intervals and final Euler,
and velocity `(D(z,t)-z)/max(1-t,.05)`. No CFG or class conditioning.

At5k,10k,25k,50k, compute standard torch-fidelity Inception2048 FID on fixed
EMA samples:1024 generated images at intermediate points;5000 at the final
point, all against the5000-image validation reference. Record counts beside
every score. The final FID has a different sample count from intermediate
FIDs; this difference must not be read as a training gain. Feature weights
are the existing cached standard Inception checkpoint; no downloads or new
feature training. Pixel conversion is deterministic clipping/rounding from
[-1,1] to uint8. Float64 covariance FID is checked against the exact low-rank
formula. Reference features and generated features are retained with hashes.

Interpret validation curves together with fixed-case grids and FID for both
ordinary controls. If the ordinary controls are still producing poor samples
or materially improving at the end, baseline adequacy is unresolved and no
architecture verdict follows. A low fixed-budget loss, percentage FID gap,
or implementation check alone cannot establish a useful image method. Both
ISRU and ordinary-control outcomes will be published; do not rescue a weak
arm by selecting a new recipe after inspecting these outcomes.

## Execution and resources

Runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1`.
All data, source snapshots, raw logs, features and checkpoints stay outside
Git; code, protocol and curated evidence are published. Warm environments
image-training@475428fb and image-metrics@dc729a50 are reused without installs.

M1/16GiB calibration with batch512, ordinary SDPA and foreach Adam measured
2.027s ISRU,1.980s matched,2.159s wide per update; about6.17s per paired cohort
and5.6GB peak MPS driver memory. The full budget is roughly86 hours of
training plus evaluation. These are hardware-only random-image timings,
not quality results. The earlier batch128/manual-attention calibration is
also retained. Wall time and memory will be measured again in the real run.

The supervisor logs process CPU/RSS, free disk, status age, MPS allocations
and actual update speed every30 seconds. Portable GPU utilization is not
available through PyTorch MPS and is explicitly left unknown. Training pauses
while the metric subprocess uses MPS. Latest full state is saved every500
updates; named milestones are kept. Free space below8GiB or nonfinite
gradients fails visibly; there is no automatic restart or quality-based
early stopping. SIGTERM/SIGINT requests a stop after a complete cohort.
`caffeinate -i -w PID` prevents idle sleep while the owned trainer is alive.

## Exact documented readiness commands

After freezing the exclusive runtime, a fresh agent reads this document and
the Mac mini environment ledger, then executes these commands verbatim once.
It must not inspect/repair source, change flags or retry; report actual exits
and any discrepancy. Neither command performs formal quality training.

```sh
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1/source/preflight.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1 --output /Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1/documented-training
```

Expected: terminal exit0, `TRAINING_WITNESS_PASS` and a PASS `result.json`.
It checks the original ISRU forward/parameter/input derivatives, analytic
Jacobian, original SwiGLU, active full-model SDPA, actual real-data MPS
optimizer/EMA, physical disk resume with new objects, generator state,
validation and sampler. Float32 resume uses atol2e-6/rtol2e-4, without an
unsupported bitwise GPU guarantee. The shared CPU data-generator state must
be bitwise equal.

```sh
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1/source/metrics.py fixture --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1 --output /Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1/documented-metrics.json
```

Expected: terminal exit0, `METRICS_WITNESS_PASS` and a PASS JSON. Check
FID numerical identity, exact pixel roundtrip, actual Inception CPU/MPS
features (max difference<1e-3 and RMS<1e-4), and pinned package versions.
Bind both actual receipts and source hashes in `documented-pass.json`.

Launch once after both pass:

```sh
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1/source/launch.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1
```

The command returns a detached controller PID. Verify `output/process.json`,
`output/status.json` and increasing steps in `output/train.jsonl`, rather than
equating a returned PID with a successful launch. `output/monitor-latest.json`
is the most recent process observation. A final `process-exit.json` records
the actual child exit. Resume only after confirming the old owned process
is gone, using the same launch command with explicit `--resume`; it restores
all three model/Adam/EMA states, stream and RNG from `output/latest.pt`.

Current progress is maintained in [task_plan.md](task_plan.md) and
[notes.md](notes.md), with conclusions in Report2 once evidence exists.
