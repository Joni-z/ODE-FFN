# CelebA training recipe and baseline decision

12 November 2026. Recommendation only: no new GPU job has been submitted.

Keep author-derived JiT as the immediate FFN comparison. The first CelebA64
allocation used a declared local transfer recipe, with important differences
from the published ImageNet setting. Review those differences before buying
more training time. Use AsymFlow as the next parameterization control;
PixelDiT2 is a later comparison with additional representation machinery.

## What the official JiT recipe actually specifies

The current author repository still points to the same revision used here,
`cbc743a2ada5e9762697da2c83f8c4f8379e8c17`. Updating the source alone would
not change our recipe. Its [README][jit-readme] gives an eight-GPU ImageNet
example; [main_jit.py][jit-main] computes the absolute learning rate from the
global batch. [Table 9 of the paper][jit-paper] specifies resolution-dependent
noise scaling. Neither source publishes an unconditional CelebA recipe.

| Setting | Author ImageNet256 example | Completed first CelebA allocation |
|---|---|---|
| Model and resolution | JiT-B/16, 256 | Full JiT-B body, local patch4 adapter, 64 |
| Global batch | 8 x 128 = 1,024 | 64 on one GPU |
| Learning rate | `blr=5e-5`; actual `2e-4` | Explicit absolute `1e-4` |
| Noise standard deviation | 1 at 256; 2 at 512 | 1 at 64 |
| Optimizer and schedule | AdamW (.9,.95), no decay, 5-epoch warmup, constant | Same choices |
| Training horizon | 200-epoch ablations; 600-epoch main runs | Target 200; K8 paused at 28, SwiGLU at 44 |
| Conditioning | 1,000 classes, label drop .1, guided sampling | Unconditional null label, drop 1, CFG 0 |

**Learning rate.** The implemented rule is
`actual_lr = 5e-5 * global_batch / 256`. At batch 64 it gives `1.25e-5`;
our `1e-4` is eight times that value. This identifies a recipe difference,
not evidence that a lower learning rate will win on CelebA. The published
absolute `2e-4` should not be copied while ignoring its batch of 1,024.

**Noise scale.** The paper gives `noise_std = image_size / 256`. Extending
that rule to 64 gives .25, and to 128 gives .5. The old 64-pixel run uses
four times the former amplitude, or sixteen times its variance. This is a
downward extrapolation from published 256/512/1024 settings, not a measured
CelebA optimum. Changing it changes the bridge, targets and numerical loss
scale; losses across the two recipes must not be ranked directly.

**Patch geometry.** The author's [patch embedder][jit-model] has two linear
convolutions with a 128-dimensional intermediate representation. At patch16,
an RGB patch has 768 coordinates: `768 -> 128 -> 768` is a genuine rank
restriction. At patch4 it has only 48: `48 -> 128 -> 768` imposes no additional
patch-rank restriction beyond the input dimension. Our current port keeps
the full transformer body and 256 image tokens, but changes this specific
premise of JiT. That does not invalidate the paired FFN experiment or explain
its quality by itself; it motivates a more deliberate transfer configuration.

## Recommended next configuration

The [machine-readable proposal](proposed-celeba128.json) is **not launched**
and is not an executable replacement for the existing Slurm file.

- Original aligned CelebA, official train split, author crop to **128x128**
  and horizontal flip. Keep validation/test splits separate.
- Native **JiT-B/16** definition: 12 blocks, width 768, 12 heads, bottleneck 128,
  32 context tokens starting at block4. This gives **64 image tokens** and
  restores the 768-to-128 patch compression without upscaling CelebA to 256.
- Proposed global/physical batch **256** on one GPU, subject to a memory
  and throughput probe. Use **`--blr 5e-5`**, giving actual LR **`5e-5`**.
  If batch must change, recalculate LR explicitly before registration.
- Proposed noise scale **.5**, logit-normal time mean **-.8**, std **.8**,
  denominator clip **.05**. Keep the author's loss and sampler.
- AdamW (.9,.95), weight decay 0, warmup 5 epochs, constant LR; keep the
  existing per-update EMA decays .9999/.9996 in both arms. A changed batch
  changes their image-count averaging window; this is a transfer choice,
  not exact reproduction of the ImageNet EMA window.
- Native SwiGLU versus the selected **ISRU-D K8**, same seed0 and recipe.
  Do not add Muon, REPA, a learned clock or a spectral constraint here.
- Proposed horizon 200 epochs, with checkpointed two-hour allocations.
  Fewer tokens suggest lower transformer-body work than the old B/4 port;
  actual GPU memory, throughput and time to useful FID remain unmeasured.

This is a proposed local CelebA transfer. Native B/16, the LR formula and
the noise-resolution rule have author sources; 128 resolution, batch 256
and their joint effectiveness on CelebA do not have an author validation.
The current entrypoint still enforces CelebA64: implementing this proposal
requires a small dataset-guard change and a separate launcher/protocol.

## Newer alternatives checked

| Baseline | Relevant released implementation | Decision for this project |
|---|---|---|
| [AsymFlow][asym-paper], v3 29 Oct 2026 | [Author code][asym-doc] reuses JiT, predicts full-dimensional data plus low-rank noise, and recovers velocity analytically. Official scratch config is class-conditional ImageNet256/H16. No native CelebA64 preset was found in the inspected tree. | Best next control. First compare prediction parameterizations with native SwiGLU, then swap K8 within the same parameterization. PCA must be fitted using training images only; copy its time convention and recovery equations together. |
| [PixelDiT2][pixel-paper] | [Author C2I release][pixel-readme] uses DINOv3 grounding, also at sampling, and DINOv2 REPA during training. B/L are architecture definitions without released checkpoints; the supplied training presets are H/16 ImageNet256/512. | Not the first replacement. Extra encoders, loss and grounding would change several mechanisms at once. A small-data port is still necessary. |

AsymFlow's ImageNet256/H16 result is FID 1.76 without REPA and 1.57 with REPA;
those settings must be distinguished. Its time direction is data-to-noise,
opposite the local JiT convention, so copying its time mean alone would be
wrong. FFN replacement is compatible with its JiT blocks, but it is primarily
a prediction-parameterization change, not a wholly new backbone.

## Execution order and preserved evidence

1. Inspect fixed-sampler samples from the existing SwiGLU at epoch 44 and K8 checkpoints;
   use identical evaluation settings and retain their exact identities.
   No image-quality measurement has yet been made on these checkpoints.
2. Prepare and profile the proposed 128/B16 setting. Register a final batch,
   recipe and budget before launching a fresh comparison. It cannot be
   presented as continuation of the existing 64/B4 experiment.
3. Establish the ordinary image baseline with FID50k below 50 before making
   FFN-quality claims. Evaluate both arms against the same deterministic
   training-split crop reference and sampler, with recorded EMA selection.

The old checkpoints, their frozen training runtime 882c46c and both finalized
reports remain unchanged. The separately prepared [continuation validation
receipt](continuation-validation.json) verifies bounded self-requeue of the
old recipe, not the proposed 128/B16 setting. Its GPU continuation is held
while this recipe decision is open.

The literature search covered three frozen paper IDs; [coverage.json](coverage.json)
records all three, source access and limitations. No local paper library or
arXiv API helper was available; primary papers and official code were inspected.

[jit-readme]: https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/README.md
[jit-main]: https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/main_jit.py
[jit-model]: https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/model_jit.py
[jit-paper]: https://arxiv.org/html/2511.13720v2#A1
[asym-paper]: https://arxiv.org/html/2605.12964v3
[asym-doc]: https://github.com/Lakonik/LakonLab/blob/10b8e86eb4aec656436d719dfc298ac04c83e9ae/docs/AsymFlow.md
[pixel-paper]: https://arxiv.org/html/2609.24919v2
[pixel-readme]: https://github.com/NVlabs/PixelDiT/blob/f7d7c35f71893d81c6e71d2576ea80601a1e8529/c2i/README.md
