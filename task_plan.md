# Task: launch parameter-matched ISRU-D on Torch

User authorized 14 November 2026: qualify the fixed native baseline, widen K8
to approximately its parameter count, and train K8 plus K1 on Torch; seed 0,
100 epochs, standard FID50k every 20 epochs, checkpoints and monitoring.
K1 uses the same widened architecture as K8, changing only polynomial order.

- [x] Qualify native SwiGLU baseline from completed source-bound FID35.91785.
- [x] Register exact width, operator, initialization, fixed recipe and outputs.
- [x] Implement minimal FFN/CLI and fresh launch support; verify semantics,
      equal parameters/common initialization, unchanged original controls.
- [ ] Push code; freeze new runtime on Torch; validate CPU and full-model CUDA
      training/sampling with finite gradients, memory and throughput.
- [ ] Set measured time limits including five FIDs; submit both seed0 arms.
- [ ] Attach persistent job/GPU/FID monitor; verify jobs or pending reasons.
- [ ] Publish source/config/readiness/job receipts and current status to GitHub.

Outputs: docs/celeba128-wide-20261114/. Old runs, baseline and reports preserved.
No narrow SwiGLU is selected. No new optimizer or spectral constraint.

Implementation: internal width1200, two bias-free Xavier projections, existing
P initialization and frozen input diagonal. New feature source17b61a5 pushed.
Local host has no Torch; numerical tests use the existing Torch venv.

Validation: source17b61a5 frozen onTorch; original upstream compatibility
PASS. Fresh documented CPU check and periodic evaluation fixture24/24 PASS.
CUDA readiness23013985 running onH100 gh003; time limits await measurement.
