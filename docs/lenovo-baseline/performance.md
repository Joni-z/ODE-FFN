# Lenovo bounded performance study

Authorized on1November2026 after the runtime audit. Preserve a completed-epoch
checkpoint, stop the long-run child, and measure the copied official epoch loop.
No new architecture, loss, optimizer, LR or convergence claim is introduced.

## Outcome

Formal training is **paused after four completed epochs (25,000 updates)**.
The saved2.60GB checkpoint includes the model, both EMAs and Adam. Original
and backup SHA256 remained identical after every diagnostic. See [pause.json](pause.json).
The final process check found no training/profile process and GPU0%,391MiB.

All timings below exclude20warmup updates and cover100updates, reusing the
frozen official epoch loop and real CIFAR images.

| Configuration | ms/update | Images/s | Peak allocated GiB | Training-only hours for200epochs |
|---|---:|---:|---:|---:|
| Batch8, native AdamW | 195.97 | 40.82 | 5.299 | 68.0 |
| Batch8, native + phase timers | 195.30 | 40.96 | 5.299 | 67.8 |
| Batch12, native AdamW | 276.78 | 43.36 | 6.333 | 64.1 |
| Batch16, native AdamW | 1171.62 | 13.66 | 7.359 | 203.4 |
| Batch8, PyTorch fused AdamW | 178.64 | 44.78 | 5.300 | 62.0 |
| Batch12, PyTorch fused AdamW | 260.67 | 46.04 | 6.333 | 60.3 |

The phase timers changed measured batch8 wall time by less than0.4%. The
native batch8 split was49.2ms forward/loss,90.4ms backward,40.1ms Adam,15.0ms
EMA. These are CUDA-stream intervals; host-call times overlap them and must
not be added. The separate three-step operator profiler is more intrusive and
its per-operator table is not used as a disjoint kernel-time decomposition.

Batch16 slowed severely even after warmup. During that case, the Windows
adapter counters showed7.61GiB dedicated and0.76GiB shared GPU memory. This
is consistent with memory oversubscription, but it is adapter-wide evidence,
not a per-process paging trace. Do not increase the batch to16 on this path.

Fused AdamW reduced its own phase from40.1to23.7ms at batch8; total throughput
increased9.7%. On182parameter tensors with cloned real gradients and optimizer
states, the maximum one-step parameter difference was7.45e-9; relative update
RMS error was1.66e-6. Both moment comparisons passed the registered limits.
This is numerical agreement for an optimizer step, not bitwise equivalence or
a convergence result. The plain/instrumented100-step runs themselves also
had different model hashes despite identical CUDA RNG states; their mean losses
differed by1.21e-6. No exact trajectory identity is claimed.

The fastest tested setting was batch12 + fused AdamW,12.8% more images/second
than plain batch8. Its extrapolation is still60.3training hours for200epochs,
plus evaluation. It is only2.8% faster than fused batch8, while changing batch
changes optimizer-update count, warmup and EMA horizon per image.

## Continuation decision

Keep the long run paused. These measurements do not justify automatically
resuming its unvalidated200-epoch CIFAR allocation. The core B model and token
count remain expensive on this8GB laptop GPU. Compilation/attention changes
were not benchmarked, so they carry no claimed speedup.

If continuing this exact model/recipe is chosen, keep batch8 and prefer the
tested built-in fused AdamW after explicitly integrating and checking resume
state placement. This preserves the physical batch for a small execution gain,
but still projects to62training hours. The active trainer was not patched.
If a much shorter turnaround is required on Lenovo, select and justify a
smaller configuration separately, then establish its baseline quality. No
new model, shorter convergence budget or FFN experiment was launched here.

Input source: `D:/Research/ODE-FFN/runtimes/jit_cifar10_swiglu_20261101`.
Preserved checkpoint:
`D:/Research/ODE-FFN/runs/jit_cifar10_swiglu_20261101/preserved-before-profile/checkpoint-last.pth`.
Raw diagnostic outputs: `D:/Research/ODE-FFN/runs/lenovo_profile_20261101`.
Curated evidence: [performance-results.json](performance-results.json).
Measurement code: [profile_cuda.py](profile_cuda.py). Both reports, the formal
training code/environment, and original checkpoint remain unchanged.

## Protocol
- Source: frozen JiT port3b07a57, native SwiGLU, BF16 AMP, seed0.
- Resume disposable measurements from the preserved checkpoint including Adam
  and both EMA states; never save profiling weights over training outputs.
- Real torchvision CIFAR training images and native augmentation/normalization.
- Batch8,12,16 sequentially,20warmup updates and100timed updates each.
- Warmup and timing call the original train_one_epoch. Timed phase wrappers
  record CUDA events and host times; no synchronization is added inside phases.
- Measure wall time with explicit synchronization around the full region;
  also measure batch8 without phase instrumentation as an overhead control.
- Record driver, power, memory, loss finiteness, sample count and any OOM.
  Eight cold steps are not sufficient to select an optimal batch.
- Profiled component costs will determine the next bounded check. Do not
  change precision, attention kernels or the experimental recipe preemptively.

Timing uses the registered post-warmup LR1e-4 for every disposable run. Data
augmentation, state loading and the epoch loop are imported from the frozen
source. The short windows are measurement inputs, not additional training
epochs to append to the research checkpoint. Plain and instrumented batch8
will compare model and RNG hashes as a timing-wrapper correctness witness.

Implementation references: [CUDA events in PyTorch2.5.1](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/streams.py)
and [the profiler API](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/autograd/profiler_util.py).
The measurement procedure follows the installed optimize-for-gpu skill;
software/workflow attribution: Kassis, T., Agarwal, V., He, Y., Patel, D., and
Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural
Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
The current arXiv record (v2) was checked; research reports were not edited.

Status: completed. The pause and all six measurements succeeded. No profiling
weights were saved into the formal run.

## Evidence-triggered follow-up: native fused AdamW
The batch8 phase timing attributed40.10ms of195.30ms/update to Adam. Test only
PyTorch2.5.1's built-in AdamW(fused=True), with unchanged betas, LR, epsilon,
weight decay and incoming checkpoint state. Set fused/foreach before
load_state_dict so step counters follow PyTorch's documented placement logic.
Run20warmup/100timed updates at batch8 and12. Compare one-step native/fused
updates for all active parameter tensors with cloned real gradients and Adam
states. This compares optimizer arithmetic, not two full training trajectories.
Predetermined limits: max parameter error1e-6, relative update RMS error0.005,
and relative first/second-moment RMS errors1e-5. No bitwise-equivalence claim.
Any throughput gain remains separate from a convergence result. Formal training
remains paused; no kernel, LR or batch change is installed in its frozen source.

References: [official AdamW implementation](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py)
and [checkpoint state placement](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/optimizer.py).
