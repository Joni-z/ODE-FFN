# Fold a frozen image K2 checkpoint and measure the actual saving

Owner: Zhizhe Zhang. Registered before new timing outcomes. This is an
inference/export task, not a new training family or successful ODE successor.

The cycle19 constant-clock K2 endpoint has FID179.7179 versus native183.6981.
Cycle21's float64-constructed/float32-cached version has FID179.7184. These
are existing screening outcomes, not independent new quality results. Use
the six exact cached T32 matrices from cycle21, not newly fitted constants.
Each block becomes `M2*SiLU(M1*h)`. Keep the backbone, all its learned tensors,
actual buffers and external conditioning unchanged. Export without dormant
P/controller tensors. Keep original P/controllers in the sealed parents.

Package the frozen image builder/JiT Python source, compact state dictionary,
runtime buffers, loader and provenance manifest outside Git. Loading must
work in a fresh process using only this bundle plus torch/einops/numpy, without
accessing historical runtime paths. It is an inference-only artifact and does
not support resuming the old optimizer or FFN auxiliary training losses.

## Fixed controls and correctness

Six arms: native dynamic K2; constant-clock literal K2; cached-layer K2 with
the original surrounding auxiliary computation; compact folded K2; ordinary
SwiGLU129; ordinary SiLU193. The latter two also remove unused FFN diagnostic
computations, preserving actual prediction functions and weights. Thus cached
versus literal measures folding, compact versus cached measures removal of
unused inference machinery, and native versus compact includes a clock
intervention. Do not attribute the whole speed difference to matrix folding.

Before timing, restore exact parent checkpoints/buffers and verify all parent
seals. Reproduce CPU predictions on the same1024validation pairs and first64
Heun128starts for the compact K2 against cycle21's saved folded endpoint.
Use bitwise comparison with the same batch64. Ordinary compact predictions
must reproduce their respective saved1024CPU predictions bitwise. Retain
native and literal predictions against their own parents. No FID recompute
or new test inputs. On MPS compare prediction max<=5e-4/RMS<=5e-5 and samples
max<=.05/RMS<=1e-3 against the corresponding CPU results. A failed check
blocks that backend's equivalence claim; preserve outputs and do not tune
tolerances or choose another version after outcomes.

## Finite benchmark

Mac mini CPU one thread and available MPS, float32, eval/no_grad, no compile,
autocast or quantization. Input batches1/16/64 are fixed prefixes of the
existing validation bank. For each backend/size/arm: three warm-up forwards,
seven rounds of five forwards, with arm order cyclically rotated each round.
Synchronize MPS before and after each region. Retain all trials and report
medians/ranges, never the best trial. Device-resident timings exclude transfers;
a separate batch64 end-to-end timing includes CPU/device transfers and output
conversion (seven rounds of two calls). Report whole-model timing, not just
the FFN. Timing order mitigates drift but is not randomized replication.

Time one full first64-image Heun128 generation for native, cached and compact
K2 on each backend, including input transfer and CPU output. No repeated
generation to obtain a favorable timing. Save all samples and check semantics
before comparing speed. Count remaining stored parameter/buffer bytes and
actual inference dense MACs; do not equate MAC counts with latency or label
process RSS as per-arm peak GPU memory. Monitor process RSS/CPU every5seconds.
Wall ceiling1200seconds. No new dependency installation or remote machine.

The export is retained if its CPU function checks pass. A measured speed claim
requires a smaller median full-model time in that named backend/batch and a
correct sample path. All sizes and both backends remain visible even when
slower. This can deliver a useful compressed endpoint, but does not satisfy
the still-open goal of a better trained ODE architecture.
