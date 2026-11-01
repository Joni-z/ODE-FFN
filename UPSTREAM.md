# Source and local changes

Source: [LTH14/JiT](https://github.com/LTH14/JiT/tree/cbc743a2ada5e9762697da2c83f8c4f8379e8c17).
Commit: `cbc743a2ada5e9762697da2c83f8c4f8379e8c17`.
MIT license, copyright2025 Tianhong Li, copied in [LICENSE](LICENSE).
[UPSTREAM.json](UPSTREAM.json) records original file SHA256s;
[UPSTREAM.patch](UPSTREAM.patch) shows the complete local patch.

## Copied core

Model/attention/block equations, normalization, embedding, timestep law,
velocity loss around clean prediction, initialization, optimizer, LR schedule,
two EMAs, checkpoint format, Euler/Heun sampler, pixel conversion and FID
implementation come from upstream. `Denoiser.forward`, all sampling methods
and `update_ema` are copied verbatim. Native SwiGLUFFN is copied verbatim.
No old controller, custom metric, spectral constraint, clock variant, solver
search or extra loss enters active training.

## Local changes

| File | Change |
|---|---|
| `denoiser.py` | FFN hook after native initialization. Default SwiGLU is a no-op. |
| `ffn.py` | Existing free-ISRU response and Taylor initialization; swap only block.mlp and preserve training RNG. P has the professor Linear initializer distribution, without claiming old random-draw positions. |
| `model_jit.py` | Allocate bias on the input device; add B/2 as a patch-size-only CIFAR port of the author's B body. |
| `util/model_util.py` | Nonpersistent buffers move original RoPE values with .to(device), retaining official checkpoint keys. |
| `main_jit.py` | FFN/CIFAR flags, torchvision CIFAR10 loader, single-process execution, non-CUDA eager mode, device-correct EMA/cache calls, explicit trusted-checkpoint loading for modern PyTorch, FFN resume guard, CLI help, a picklable crop for Windows workers and epoch-wise sampler shuffle in single-process mode, and optional sample retention. |
| `engine_jit.py` | Gate CUDA AMP/synchronization, place labels on model device, guard barriers; pass standard CIFAR train reference to upstream FID. Non-CUDA FID uses CPU Inception. Optional sample retention adds epoch-specific folders. |
| `util/misc.py` | Add a barrier helper that is a no-op without a process group. |

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
B/2 retains the author's B dimensions; no tiny invented model is labeled
reproduced. Its runtime/capacity budget needs measurement before a long Mac
run. Upstream resume is epoch-based; exact mid-batch/RNG replay is not claimed.
Baseline FID below50 under fixed standard evaluation is still required.
