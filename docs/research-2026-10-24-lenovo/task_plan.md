# Lenovo deployment and focused ODELayer continuation

## Objective
Deploy the current project in a dedicated directory on Lenovo's D drive, provision an isolated CUDA environment, and start a monitored seed-0 experiment that follows the October 23 mathematical review.

## Phases
- [x] Inspect Windows resources and select an isolated layout/environment.
- [x] Implement and validate the centered gate-curvature diagnostic/control.
- [x] Transfer source and exact datasets; old checkpoints are inaccessible, so use a matched fresh reference on Lenovo.
- [x] Verify native CUDA execution and the scientific controls.
- [x] Run checkpoint diagnosis; launch the justified minimum comparison with monitoring.
- [x] Record commands, provenance, status and results.

## Decisions
- Use the professor's original ODELayer and keep SwiGLU as the practical baseline.
- Seed 0 only, Gaussian and Ring8; no broad architecture or spectral sweep.
- Preserve dataset/checkpoint hashes and distinguish this Windows/CUDA environment from older CPU runs.
- Benchmark CPU/GPU before selecting concurrency; do not reserve idle GPUs.

## Status
Complete. Six seed-0 cells finished5000 updates each; Task Scheduler exit0. Benchmark selected2 CPU workers (278updates/s versus239 for2 CUDA workers). The predeclared diagnostic triggered the two gate-linearized controls. All36 checkpoints and100 diagnostic arrays audited locally; complete raw results backed up to the Mac mini. Findings and full comparison table are saved beside this plan. No further architecture experiments launched.

## Errors
The first PowerShell inventory returned non-UTF8 text. Explicit console UTF8 encoding is used for subsequent commands.
Default local Python had no torch; used the existing .venv-test for verification. The first direct Start-Process launch was reclaimed by Windows OpenSSH before the runner started; no experiment output was created. Switched to the on-demand Windows Task Scheduler and verified that Python survives command completion. Old145/torch SSH checks failed; checkpoint-derived claims will use explicitly identified fresh Lenovo runs. Two legacy directory symlinks were restored through Git's native Windows checkout behavior.
