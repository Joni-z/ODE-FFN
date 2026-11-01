# Ordinary baseline before FFN selection

## Question and boundary
Can the upstream JiT implementation, ported to unconditional 32x32 CIFAR-10,
reach FID below 50? This run establishes the comparison platform; it does not
test an ODE or ISRU effect. No architecture ranking is accepted at FID above 50.
The author's published ImageNet performance is not a guarantee for this port.

## Model, data and objective
- Native JiT-B body: 12 blocks, width 768, 12 heads, bottleneck 128 and 32 context
  tokens; only patch size is changed to 2 for 32x32 images.
- Default native SwiGLU; no spectral penalty, new loss, scheduler search or
  added ODE component. Keep copied clean prediction, velocity loss and noise law.
- Official torchvision CIFAR10 training split: 50,000 RGB images; original
  center crop and random horizontal flip; uint8 normalized to [-1,1].
- Unconditional: label dropout probability 1 and sampling CFG scale 0. All
  labels are replaced by the null class during training. Seed 0 only.

## Optimization and budget
Use [train.json](train.json). Physical batch is finalized from measured CUDA
memory/throughput before launch, not from a quality sweep. No accumulation or
optimizer rewrite is introduced. AdamW betas (.9,.95), weight decay 0,
absolute LR 1e-4, five warmup epochs, then constant. This LR is a declared
local CIFAR setting, not an author-validated recipe. Retain the two original
EMAs (.9999 and .9996); default evaluation uses EMA1.

Initial allocation: 200 epochs, about ten million image presentations. Log
finite loss, throughput and GPU telemetry. Save the latest full model/Adam/two
EMA checkpoint each epoch; retain upstream numbered checkpoints every 100
epochs. Resume semantics remain upstream epoch-based, without a claim of
exact mid-epoch/RNG continuation. No automatic restart or rescue tuning.

## Evaluation and stopping
- Original 50-interval sampler: 49 Heun intervals plus final Euler. No sampler
  search or best-checkpoint selection for reporting a method gain.
- Online FID uses 5,000 generated images and the standard 50,000-image
  `cifar10-train` reference through the author's pinned torch-fidelity fork.
  Keep generated PNGs per evaluation epoch, TensorBoard FID/IS and stdout.
- Evaluate after completed epochs20,40,...,200. The one-based cadence avoids
  an expensive first-epoch evaluation during warmup; objective/sampler equations
  are unchanged. These FID5k points cannot satisfy the final FID50k gate.
- Before admitting an FFN comparison, run the same standard evaluator with
  50,000 generated images on a clearly identified checkpoint; require FID50k
  below50 and inspect generated examples. Record final evaluation separately.
- At the budget boundary, inspect the learning curve and quality. If the gate
  is not met, report baseline reproduction as incomplete and choose any further
  budget explicitly. Do not infer that ISRU helps or fails from this run.
- Stop on nonfinite training loss or process failure, retain logs/checkpoints,
  and diagnose before restarting. Monitor at 30-second intervals and preserve
  any interrupted state. A sleeping/offline host is not a completed experiment.

## Controls and future comparison
Only the ordinary baseline launches now. ISRU becomes eligible after baseline
quality is established. The candidate will face the same recipe and data, with
parameter/compute differences made explicit; neither same-width nor same-step
training alone guarantees a fair resource comparison.
