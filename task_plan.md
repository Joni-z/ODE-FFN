# Professor follow-up execution, 15 November 2026

Authorized: execute on 145 physical GPUs **4, 5, 6, 7**; Torch overflow only.
Training seed 0. Preserve Report1, Report2 and historical runtimes.

- [x] Inspect resources, warm environment and prior protocol/evidence.
- [ ] Recover source-bound Ring8 checkpoints; Product16 and both datasets are on145.
- [x] Register exact activation and C8 protocols, implement adapters and tests.
- [x] Verify native trainer continuation, derivatives, symmetry and real-checkpoint restoration.
- [x] Push source, run a fresh documented witness, launch with per-GPU monitoring.
- [x] Analyze Product16 actual parameter/update rank and condition separately.
- [ ] Add recovered/replayed Ring8 diagnostics with separate provenance.
- [ ] Complete paired C8 sampling and the two tanh20k trajectories, then common-bank evaluation.
- [ ] Publish source-bound conclusions and choose the next architecture test from evidence.

P0: reuse spectra and frozen checkpoints; no new optimizer sweep.
P1: exactly two new scaled-tanh K1 trajectories, Ring8 and Product16.
P2: conditional image architecture work after these results.

Current: five registered seed0 pipelines launched on145 GPUs4-7 atda8b8a4.
Queue PIDs3199710/3199712/3199721/3199728. Missing Ring8 weights are replayed
from the exact source/data recipe, rather than labelled recovered originals.
Readiness3 passed; frozen evaluator restores original Product16 output exactly.

Readiness attempt1: failed nonzero-gradient fixture because a constant readout
annihilated centered LayerNorm output. Corrected test-only readout to a
deterministic nonconstant row; record attempt2 separately. No formal training.
