# Current task: standard FID evaluation every20epochs

User request13November2026: use a standard FID implementation; measure the
existing models now and evaluate every20completed epochs during training.

- [x] Inspect existing evaluations22993694/22993695; both running, preserve them.
- [x] Inspect saved weights: only checkpoint-last.pth at100epochs in each run;
  P-only snapshots cannot reconstruct checkpoints20/40/60/80.
- [x] Verify18FID/Inception/resize definitions match official torch-fidelity.
- [x] Enable native online evaluation every20epochs and retain full milestone
  checkpoints, without changing the training objective, FFN, optimizer or sampler.
- [x] Actual tiny100-epoch entrypoint regression PASS for scheduling, full checkpoints and RNG;
  check syntax and frozen/source/report preservation.
- [x] Native checkpoint100FID50k35.91785 verified, job22993694 completed0:0.
- [x] Publish code/evidence as0dbc9ad and fast-forward the active Torch checkout.
- [x] K8 job22993695 completed0:0; checkpoint100FID50k43.90767 retrieved.
- [x] Verify50,000PNG images per arm, matching evaluation keys and frozen checkpoint identities.
- [x] Publish both completed metrics and checks; active Torch sync follows this final result commit.

No duplicate evaluations or replay training. Existing jobs use frozen65864ad;
code changes apply to subsequent training, not the already-completed100epochs.
Fixed evaluation: firstEMA.9999, seed0, authorHeun50,CFG0,50000images and official
CelebA training split through deterministic128px crops. Training seed0only.

Prior complete training/P evidence: docs/celeba128-20261112/results.md.

## Implementation notes

Future K8/native allocations include five FID50k evaluations (provisional14h/8h);
current fixed-checkpoint3h allocations are unchanged. No new training submitted.

## Resolved preparation errors

A source transfer initially added a trailing blank line; exact pinned author
SHA256 was restored before regenerating UPSTREAM.patch. A failed patch context
made no changes; the corrected narrow patch applied. No running job was altered.
