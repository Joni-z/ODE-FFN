# Lenovo bounded performance study

Authorized on1November2026 after the runtime audit. Preserve a completed-epoch
checkpoint, stop the long-run child, and measure the copied official epoch loop.
No new architecture, loss, optimizer, LR or convergence claim is introduced.

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

Status: preparing; a guarded pause waits for the fourth completed epoch, verifies
and copies its checkpoint, then stops only the bound training process tree.
