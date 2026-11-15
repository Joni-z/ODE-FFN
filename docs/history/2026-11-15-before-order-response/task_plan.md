# Professor follow-up execution, 15 November 2026

Authorized: execute on 145 physical GPUs **4, 5, 6, 7**; Torch overflow only.
Training seed 0. Preserve Report1, Report2 and historical runtimes.

- [x] Inspect resources, warm environment and prior protocol/evidence.
- [x] Recover source-bound Ring8 data and replay missing weights; Product16 originals reused.
- [x] Register exact activation and C8 protocols, implement adapters and tests.
- [x] Verify native trainer continuation, derivatives, symmetry and real-checkpoint restoration.
- [x] Push source, run a fresh documented witness, launch with per-GPU monitoring.
- [x] Analyze Product16 actual parameter/update rank and condition separately.
- [x] Add replayed Ring8 symmetry diagnostics with separate provenance.
- [x] Complete paired C8 sampling and the two tanh20k trajectories, then common-bank evaluation.
- [x] Publish source-bound conclusions and choose the next architecture test from evidence.

P0: reuse spectra and frozen checkpoints; no new optimizer sweep.
P1: exactly two new scaled-tanh K1 trajectories, Ring8 and Product16.
P2: conditional image architecture work after these results.

Current: completed. Five primary trajectories and all15k/20k frozen evaluations
passed. The independent original-script20k SwiGLU audit exactly matches the
new entry's full numerical states. All GPUs4-7 are released and monitors ended.
[Findings](docs/research-assessments/2026-11-15-professor-execution/findings.md)
record learned spectra, matched tanh controls, C8interventions, exact-orbit
error decomposition, the oracle reference and the next ODE hypothesis.
No additional training remains queued; proposed next experiments are not launched.

Readiness attempt1: failed nonzero-gradient fixture because a constant readout
annihilated centered LayerNorm output. Corrected test-only readout to a
deterministic nonconstant row; record attempt2 separately. No formal training.
