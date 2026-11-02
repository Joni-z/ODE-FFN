# Cycle6 registration: does the raw-K1 lead survive a longer matched budget?

Registered after cycle5 rejection and the saved-sample structure audit, before
new training. No new gate/optimizer/architecture is selected. Current20k curves
do not establish convergence; a low aggregate sampled distance can also hide a
blurred mixture coordinate. Further D-shape searches would not resolve either.

## Single factor and eight trajectories

Change only the fixed update budget from20k to80k for existing models. Use the
unchanged copied trainer, loss, caches, batch512, seed0, constant learning rates,
optimizer groups and Heun32. No scheduler, extra loss, projection, new parameters,
packages or image training. This explicitly extends the earlier program's20k
budget; do not compare80k candidates with20k controls as an architecture gain.

| Task | Raw K1 | Matched SwiGLU64 | SwiGLU128 | Stronger SwiGLU192 |
|---|---|---|---|---|
|Ring8 / Lenovo|Muon.02|Muon.02|Muon.02|Muon.02|
|Product16 /145|Muon.006|AdamW.001|AdamW.0003|AdamW.0003|

These fixed configurations are the earlier lowest-validation choices within
the shown families, selected before the terminal-sample audit. No new rate
sweep or test-driven choice. RawK1 and Swi64 match both parameters and dense
MACs closely; Swi192 preserves the stronger ordinary frontier. The rejected
absK1 family stays closed. Bilinear remains a frozen diagnostic reference.

Swi128 retains the intermediate ordinary frontier as well; the three ordinary
widths need not keep their20k ranking at80k. Four tasks per host also allow two
useful controls to occupy the last batch instead of leaving a three-cell tail.
The purpose is to distinguish a short-budget field lead from a persistent useful
tradeoff, not to nominate a new method merely because a longer run improves.
Retain all eight outcomes even if an ordinary control overtakes raw. No accepted
advantage from field fitting alone, a blurred marginal or a weak-parent win.

## Prefix identity and verification

Use fresh isolated outputs, preserving the original20k runs. This intentionally
repeats the first20k to verify an exact numerical prefix under the new budget;
configuration/model identifiers change only metadata. Every new20k checkpoint
must match its archived parent in model tensors, nonpersistent buffers,
optimizer tensors/counters, and data/global RNG states. Hash-bind both configs,
data manifests and original parent checkpoint. A mismatch must be diagnosed
before trusting later results; never reset the old run or silently retry.

Before formal launch, fresh documented CUDA checks compare actual8-update
old-budget/new-budget native states and continue the new checkpoint8->16.
No new mathematical layer implementation needs to be verified. During formal
training inspect prefixes once the loop has advanced beyond their save point.
Monitor processes/resources every30s with the existing queue.

## Readouts and execution

Use the final80k checkpoint as primary; retain20/40/60/80k validation curves and
saved P histories as descriptive trajectories, not independent repetitions or
checkpoint-selection opportunities. Replay80k field and test SW2, then inspect
the same known Product16 marginal examples/gap counts without replacing primary
metrics or selecting a checkpoint from them. No image-quality claim.
Extend only the existing read-only matrix/update logger to40k/60k/80k; retain
all earlier logging points. No new gradient or optimizer computation is inserted.
Final examples include all4selected models per task. Product16 uses the same
0/7/15 latent-axis histograms and predeclared gap readouts. Ring8 displays fixed
indices0,4,...,4092 (1024points) from each4096-sample final array, with the same
target indices and frame[-4,4]^2; report any outside-frame mass. This is a visual
subsample only; primary metrics keep all4096samples.

The earlier controls already establish the20k recipe. Launch matched Swi64 and
raw together, then fill the next slot with Swi192, two workers per freeGPU0 on
each host. This permits overlap while retaining all strong controls. Actual
wall time and the extra budget are reported separately from model quality.

- [x] Implement a budget-only entry/grid, source-bound prefix auditor and native
      actual-entry witness using unchanged parent builders/trainer.
- [x] Freeze/push; deploy new worktrees and pass fresh documented CUDA checks.
- [x] Launch all8 with monitors and verify all8 exact archived20k numerical prefixes.
- [ ] Finish/replay80k endpoints, retained learning curves and P/sample readouts.
- [ ] Decide whether raw's lead survives matched ordinary convergence, then choose
      the next research question without reopening stopped feature families.

Native readiness passed at b33c837 on both hosts: all4actual8-update old/new
budget numerical states match and all4new runs resume8-to16. Formal queues now
run2workers perhost; independent post-SSH increasing steps and91–93% GPU use
verified.20k prefix and80k quality checks are still outstanding.
