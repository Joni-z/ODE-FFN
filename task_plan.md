# Task: continue widened K1/K8 from100 to160 epochs

Authorized14November2026: user requests continuing the two unfinished
convergence trajectories. Bound this stage at160, with FID50k120/140/160.
Preserve baseline100, original100epoch evidence and frozen source17b61a5.

- [x] Verify last checkpoints exactly match the100epoch model/optimizer/EMA/RNG
      anchors and can resume with epoch target160 and constant LR5e-5.
- [x] Register continuation, estimate full allocation including three FIDs.
- [x] Submit both continuations with same source/recipe and no fresh start.
- [x] Attach new job/progress/FID monitor.
- [x] Verify both logs restore original checkpoint/optimizer after100epochs.
- [x] Verify K1 completed160,101600updates,finite model/twoEMAs and FID27.78868.
- [ ] Collect K8 final160epoch FID when the remaining run completes.
- [x] Publish updated status, registration and launch receipts to GitHub/Torch.

No architecture, optimizer, EMA, sampler, dataset or baseline change.

Checkpoint model, optimizer, twoEMAs and RNG exactly match100epoch anchors.
Warmup is finished; constant LR5e-5 continues. Budget helper syntax typo was
corrected before creating the record or submitting jobs.

First submit guard expected a status key but doc receipt uses pass:true;
normalized the receipt check before any real submission. No jobs existed
from that failed preflight; duplicate guards remain in place.

Submitted K8/K1 jobs23085012/23085013, source17b61a5, target160, monitor2415208.
Both resumed correctly. K1 completed; K8 running on L40Sgl060.

Submission/resume verified. K8 snapshot142/160, FID140=30.08220, GPU99–100%;
K1 complete160, FID27.78868. The live monitor continues independently.
