# Lenovo baseline deployment

## Goal
Run the upstream-derived unconditional CIFAR-10 SwiGLU baseline on the recovered
Lenovo RTX 4060, with persistent logs, checkpoints and GPU monitoring. Establish
FID below 50 before interpreting an FFN comparison. Seed 0 only.

## Actions
- [x] Inspect GPU processes, disk, Python environments and Git checkout.
- [x] Sync a clean upstream-derived checkout through GitHub; preserve old runs.
- [x] Register a CUDA environment, training budget and fixed evaluation protocol.
- [x] Verify the copied code and measure actual full-model memory/throughput.
- [x] Run the documented readiness check with a fresh agent.
- [x] Launch a persistent Windows task and confirm increasing training steps.
- [x] Publish code, protocol and verified launch status to GitHub main.

## Scope
The user explicitly authorized Lenovo execution on 1 November 2026. This extends
the earlier Mac-only decision. Do not edit Report1/Report2 or stop the old Mac
run. Default native SwiGLU comes first. Do not revive the bespoke image trainer.
Do not claim that the local CIFAR port is an author-validated CIFAR recipe.

## Status
All readiness checks passed. The detached 3b07a57 baseline is running with
increasing updates and 99% GPU load. The verified launch record accompanies this status.

## Errors
Recorded in notes.md: unsigned old interpreter blocked, slow direct downloads,
bundle remote-ref clone failed, Windows-only colorama omitted, oversized SFTP
message rejected. All have concrete resolutions; no security setting changed.

## Follow-up: training status and interactive GPU monitor
- [x] Read current process status, training log, checkpoint inventory and recent speed.
- [x] Check the official nvitop Windows instructions and the user's SSH shell/PATH.
- [x] Install nvitop separately from the training environment and expose the command.
- [x] Verify a fresh SSH login and the interactive GPU display; preserve the job.
- [x] Publish a timestamped status update and the exact usage instructions.

## Runtime audit requested by the user
- [x] Compare the live launch and copied training path against the pinned official JiT recipe.
- [x] Verify current throughput, token count, batch budget, LR/EMA schedules, and evaluation cost.
- [x] Identify supported conclusions and options without changing the running job.
- [x] Publish a source-linked runtime audit and a concrete recommendation.

Scope: read-only runtime inspection and documentation. Preserve the running
checkpoint, training environment, and both reports. Any budget/architecture
change must be presented as a changed experiment, not a transparent speedup.
