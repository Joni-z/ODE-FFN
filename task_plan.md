# Professor follow-up execution, 15 November 2026

Authorized: execute on 145 physical GPUs **4, 5, 6, 7**; Torch overflow only.
Training seed 0. Preserve Report1, Report2 and historical runtimes.

- [x] Inspect resources, warm environment and prior protocol/evidence.
- [ ] Recover source-bound Ring8 checkpoints; Product16 and both datasets are on145.
- [ ] Register exact activation and C8 protocols, implement adapters and tests.
- [ ] Verify native trainer continuation, derivatives, symmetry and real-checkpoint restoration.
- [ ] Push source, run a fresh documented witness, launch with per-GPU monitoring.
- [ ] Analyze actual parameter/update rank and condition separately.
- [ ] Complete paired C8 sampling and the two tanh20k trajectories, then common-bank evaluation.
- [ ] Publish source-bound conclusions and choose the next architecture test from evidence.

P0: reuse spectra and frozen checkpoints; no new optimizer sweep.
P1: exactly two new scaled-tanh K1 trajectories, Ring8 and Product16.
P2: conditional image architecture work after these results.

Current: implementation preparation. 145 GPUs4-7 idle; /data2 has47GB free.
Lenovo SSH timed out; searching existing mirrors for Ring8 weights.
