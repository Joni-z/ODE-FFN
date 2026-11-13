# Source and local changes

Source: [LTH14/JiT](https://github.com/LTH14/JiT/tree/cbc743a2ada5e9762697da2c83f8c4f8379e8c17).
Commit: `cbc743a2ada5e9762697da2c83f8c4f8379e8c17`.
MIT license, copyright2025 Tianhong Li, copied in [LICENSE](LICENSE).
[UPSTREAM.json](UPSTREAM.json) records original file SHA256s;
[UPSTREAM.patch](UPSTREAM.patch) shows the complete local patch.

## Copied core

Model/attention/block equations, normalization, embedding, timestep law,
velocity loss around clean prediction, initialization, optimizer, LR schedule,
two EMAs, base checkpoint fields, Euler/Heun sampler, pixel conversion and FID
implementation come from upstream. `Denoiser.forward`, all sampling methods
and `update_ema` are copied verbatim. Native SwiGLUFFN is copied verbatim.
No old controller, custom metric, spectral constraint, clock variant, solver
search or extra loss enters active training.

## Local changes

| File | Change |
|---|---|
| `denoiser.py` | FFN hook after native initialization. Default SwiGLU is a no-op. |
| `ffn.py` | Existing Free ISRU and selected frozen-diagonal ISRU-D, with order8 default; swap only block.mlp and preserve training RNG. P uses the original Linear initializer distribution. |
| `model_jit.py` | Allocate bias on the input device; add B/2 for CIFAR and B/4 for CelebA64 as patch-size-only adapters of the author's complete B body. |
| `util/model_util.py` | Nonpersistent buffers move original RoPE values with .to(device), retaining official checkpoint keys. |
| `main_jit.py` | FFN/CIFAR flags, torchvision CIFAR10 loader, single-process execution, non-CUDA eager mode, device-correct EMA/cache calls, explicit trusted-checkpoint loading for modern PyTorch, FFN resume guard, CLI help, a picklable crop for Windows workers and epoch-wise sampler shuffle in single-process mode, optional sample retention, and evaluation after completed epoch multiples. The current default interval is 20 epochs; retain full numbered evaluation checkpoints, accept checkpoint-file resume and preserve training RNG around seeded evaluation. |
| `engine_jit.py` | Gate CUDA AMP/synchronization, place labels on model device, guard barriers; pass standard CIFAR train reference to upstream FID. Non-CUDA FID uses CPU Inception. Optional sample retention adds epoch-specific folders. CIFAR metric batches are capped by generation batch size for limited VRAM. |
| `util/misc.py` | Add a barrier helper, atomic checkpoint writes, optional single-GPU RNG state, P-history snapshots and progress JSON. |
| `util/datasets.py` | Official torchvision CelebA training split with a constant discarded label; image-only wrapper for the standard FID library. |

The CelebA extension supports64px/B4 and128px/native B16, adds CLI/order and loader flags, a single-GPU runtime
budget checked between completed epochs, epoch loss/throughput records and
RNG restoration. `engine_jit.py` returns its existing metric averages and
passes deterministic crops at the selected CelebA resolution to the unchanged FID library.
The loss, EMA and sampling equations remain verbatim. Short Slurm allocations
retain full checkpoints and save all learned P tensors. Evaluation is separate
from the initial bounded training allocations.
Resume additionally rejects changes to the model, resolution, batch, bridge,
seed and optimizer/EMA recipe rather than silently loading old optimizer state
into a new experiment. This does not change the author's training equations.

`util/crop.py`, `util/lr_sched.py`, `prepare_ref.py`, `environment.yaml` and
`LICENSE` are unchanged. Mac requirements form a separate execution profile,
pinning the author's torch-fidelity fork at
`bfe72b995473e4fee9351705d7b005f17442a4ca`.

## Limits

The author's public recipes target ImageNet, tested on8H200 GPUs; they do
not establish a32x32 unconditional CIFAR result. Meta's
[official CIFAR example](https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/examples/image/README.md)
uses a U-Net, not a Transformer FFN benchmark. Its reported result is not
transferred to our model.

CPU/MPS and CIFAR ports differ from the original CUDA paper environment.
B/2 and B/4 retain the author's B dimensions. The CelebA adapter is a declared
local dataset/resolution port. Resume is epoch-based; the new single-GPU
checkpoint also records RNG state. No exact mid-batch continuation is claimed.
Native SwiGLU on CelebA128 now reaches FID50k 35.91785 at 100 epochs under the
fixed standard evaluation ([receipt](docs/celeba128-20261112/FID100-swiglu.json)).
This validates the CelebA128 ordinary baseline; K8 FID is still running.
