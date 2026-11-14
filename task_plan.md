# Task: continue widened K1/K8 from100 to160 epochs

Authorized14November2026: user requests continuing the two unfinished
convergence trajectories. Bound this stage at160, with FID50k120/140/160.
Preserve baseline100, original100epoch evidence and frozen source17b61a5.

- [x] Verify last checkpoints exactly match the100epoch model/optimizer/EMA/RNG
      anchors and can resume with epoch target160 and constant LR5e-5.
- [x] Register continuation, estimate full allocation including three FIDs.
- [ ] Submit both continuations with same source/recipe and no fresh start.
- [ ] Attach new job/progress/FID monitor; verify resume starts at epoch101.
- [ ] Publish updated status, registration and launch receipts to GitHub/Torch.

No architecture, optimizer, EMA, sampler, dataset or baseline change.

Checkpoint model, optimizer, twoEMAs and RNG exactly match100epoch anchors.
Warmup is finished; constant LR5e-5 continues. Budget helper syntax typo was
corrected before creating the record or submitting jobs.
