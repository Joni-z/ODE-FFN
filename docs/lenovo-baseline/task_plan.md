# Lenovo baseline deployment

## Goal
Run the upstream-derived unconditional CIFAR-10 SwiGLU baseline on the recovered
Lenovo RTX 4060, with persistent logs, checkpoints and GPU monitoring. Establish
FID below 50 before interpreting an FFN comparison. Seed 0 only.

## Actions
- [x] Inspect GPU processes, disk, Python environments and Git checkout.
- [x] Sync a clean upstream-derived checkout through GitHub; preserve old runs.
- [ ] Register a CUDA environment, training budget and fixed evaluation protocol.
- [ ] Verify the copied code and measure actual full-model memory/throughput.
- [ ] Run the documented readiness check with a fresh agent.
- [ ] Launch a persistent Windows task and confirm increasing training steps.
- [ ] Publish code, protocol and verified launch status to GitHub main.

## Scope
The user explicitly authorized Lenovo execution on 1 November 2026. This extends
the earlier Mac-only decision. Do not edit Report1/Report2 or stop the old Mac
run. Default native SwiGLU comes first. Do not revive the bespoke image trainer.
Do not claim that the local CIFAR port is an author-validated CIFAR recipe.

## Status
Official Python and auxiliary packages installed. CUDA wheel transfer is ongoing;
full-model witness and persistent training launch remain pending.

## Errors
Recorded in notes.md: unsigned old interpreter blocked, slow direct downloads,
bundle remote-ref clone failed, Windows-only colorama omitted, oversized SFTP
message rejected. All have concrete resolutions; no security setting changed.
