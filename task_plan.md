# Task: launch parameter-matched ISRU-D on Torch

User authorized 14 November 2026: qualify the fixed native baseline, widen K8
to approximately its parameter count, and train K8 plus K1 on Torch; seed 0,
100 epochs, standard FID50k every 20 epochs, checkpoints and monitoring.
K1 uses the same widened architecture as K8, changing only polynomial order.

- [x] Qualify native SwiGLU baseline from completed source-bound FID35.91785.
- [x] Register exact width, operator, initialization, fixed recipe and outputs.
- [x] Implement minimal FFN/CLI and fresh launch support; verify semantics,
      equal parameters/common initialization, unchanged original controls.
- [x] Push code; freeze new runtime on Torch; validate CPU and full-model CUDA
      training/sampling with finite gradients, memory and throughput.
- [x] Set measured time limits including five FIDs; submit both seed0 arms.
- [x] Attach persistent job/GPU/FID monitor; verify jobs or pending reasons.
- [x] Publish source/config/readiness/job receipts and current status to GitHub.

Outputs: docs/celeba128-wide-20261114/. Old runs, baseline and reports preserved.
No narrow SwiGLU is selected. No new optimizer or spectral constraint.

Implementation: internal width1200, two bias-free Xavier projections, existing
P initialization and frozen input diagonal. New feature source17b61a5 pushed.
Local host has no Torch; numerical tests use the existing Torch venv.

Validation: source17b61a5 frozen onTorch; original upstream compatibility
PASS. Fresh documented CPU check and periodic evaluation fixture24/24 PASS.
CUDA readiness23013985 running onH100 gh003; time limits await measurement.

CUDA23013985 PASS H100 gh003. Formal K8/K1 jobs23015406/23015408 submitted.
K8 allocated H100 gh011; requested cap reduction20h->10h was rejected by the cluster;20h remains.
K1 initialcap9h.
Monitoring460860,60seconds, tracks FID/checkpoints; per-jobGPU telemetry30s.

Launch confirmed: K8 completed epoch1, entered epoch2,~.149s/update, GPU98–99%.
K1 remains queued with QOSGrpGRES. Both monitored; no duplicate submissions.
Reports1/2 and baseline source/checkpoint unchanged.
