# Fixed image-training budget comparison

Owner: Zhizhe Zhang. Registered before new continuation outcomes. This asks
whether the existing image-model frontier is stable from5,000to20,000updates.
It does not select a new ODE architecture or reopen a failed modified family.

The image learning curves have not plateaued: from2,500to5,000updates,
weighted validation MSE drops0.190996to0.176842 for nativeK2,
0.184319to0.171234 for Swi129, and0.186572to0.171526 for SiLU193.
Late improvement is not a prediction of an ODE reversal; its slope does not
show a favorable catch-up. The uncertain premise is whether our fixed5,000-step
ranking remains useful for selecting an architectural direction at larger
budget. Existing finite-budget conclusions and all prior failed gates stay intact.

## Exactly seven unchanged arms

Continue every native image arm from cycles17and20: K1,K2,Swi86,Swi129,
SiLU64,SiLU131,SiLU193. No width or favorable checkpoint is selected.
Start from each final5,000model, actual Adam state, data/global RNG and
nonpersistent buffers. All seven have the same consumed-stream digest.
Final endpoint20,000 is fixed, with10,000 and15,000 intermediate snapshots.
No early success selection, post-result extension or best-checkpoint ranking.

Keep unconditional CIFAR32, all model shapes, image normalization/splits,
batch64, time sampler, clean prediction and delta=.05 cap, AdamW1e-4,
betas(.9,.99), zero weight decay, clip1, constantLR, seed0, no EMA/AMP,
augmentation or new loss. No frozen-clock intervention enters training.
Use the exact original update/draw/optimizer/restore functions, data banks
and original constructors. Existing datasets/checkpoints remain read-only.

## Measurements and interpretation fixed before outcomes

Primary: paired weighted validation MSE at20,000 on the existing1,024pairs.
Also retain clean MSE, all windowed losses/gradients/clipping and validation
every1,000updates; all five denoising panels; original P/eigen/singular-value
histories and clocks at5,000/10,000/15,000/20,000. Save ordinary matrix histories.
Checkpoint every1,000updates, retaining optimizer/RNG/buffers and stream state.
Generate the common1,024starts with unchanged CPU Heun128 at10,000and20,000. Reuse the
fixed2,048reference features and existing MPS Inception evaluator for all seven;
these remain exposed small-sample screening FIDs, not a fresh test estimate.

Report every arm at5k/10k/15k/20k and all parameter/work comparisons. For each
fixed ODE-versus-ordinary contrast, report continuous error/FID differences
and any sign reversal. There is no convergence claim from a downward slope,
and a budget-dependent ranking does not identify its optimization cause.
If primary rankings persist, retain that architecture-selection evidence
within this finite budget range. If they reverse, scope the earlier conclusion
to5k and require a separately registered explanation/confirmation; do not call
this a new ODE method. FID-only gains do not override fitting, K1 or ordinary
controls. Synthetic transfer failures remain failures under their old protocols.

A prospective follow-up lead at the primary20k endpoint requires at least5%
lower FID than both declared ordinary controls, with weighted and clean MSE
no more than5%worse. ForK1 retain Swi86/SiLU131; forK2 retain Swi129/SiLU193.
Report their known parameter/work mismatches.10k is stability context, not
an interchangeable success endpoint. A practicalK1lead does not require K2
to win; a quadratic-mechanism claim separately requires the K1control. Even
a qualifying lead is not a new method or confirmation. Failure ends this
budget question without another endpoint/optimizer/width rescue.

## Execution and readiness

Mac mini CPU float32, at most two one-thread children; no GPU reserved for
training. Lenovo SSH currently times out. Reuse image-training@475428fb and
image-metrics@dc729a50 without installation. Full-state physical disk resume
must agree for each arm: direct three updates versus one plus reload plus two,
including moments, runtime buffers, RNG, consumed stream and predictions.
Replay each5k prediction against its saved original before any formal update.
Check the actual entrypoint with a fresh agent following the documented command.

Monitor parent/child identity, actual CPU/RSS and per-arm progress every5seconds;
status every100updates. The parent may live across conversation turns. A polling
timeout does not imply failure and must not trigger a duplicate run. Cell ceiling
7,200seconds; total training ceiling28,800seconds. Expected wall time1–2hours,
estimated from recorded native runs; no duration guarantee or GPU-speed claim.
Preserve crashes and partial states. No auto-retry or shortened endpoint claim.

Preflight command after preparation:

```text
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/image-budget-continuation/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-budget
```

No formal launch until all seven native/reload checks and documented-entrypoint
verification pass. Full parent/source preservation and independent final
prediction/panel/sample/stream/P-history replay precede quality claims.
This is an empirical premise test on existing models; the broader useful-ODE
successor goal remains unachieved.
