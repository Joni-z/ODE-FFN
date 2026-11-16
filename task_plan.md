# ODE mechanism investigation

Goal active: determine which ODE constraints help and identify the mechanism.
Keep seed 0, the pinned native code and finalized reports. Mac mini is
available freely. On145, use only idle physical GPUs4-7;0-3are forbidden.
Never stop or share a foreign task.

## Completed evidence

- [x] Coefficient/state/stage training and fair ordinary controls. Fixed
  Taylor coefficients do not add a consistent benefit; state feedback helps
  the within-family Ring8 comparison.
- [x] Shared versus released-stage operators at identical dense work. Sharing
  improves generation on both tasks; Product16 field MSE moves oppositely.
- [x] Full solver refinement and frozen symmetry intervention. Ring8 reverses
  the sharing comparison after C8projection; Product16 retains the gap.
- [x] Exact density transport identity and independently checked observer.
- [x] Full population observation:8cases,2fields,4x256sources,1024/2048steps.
  All128rows independently reconstruct within4.62e-14. All16cells include
  some resolution-sensitive paths; the pilot did not certify those sources.
- [x] Register bounded all-source adaptive refinement4096/8192/16384, with
  original thresholds and no point removal. Fresh CPU/CUDAentries pass.
- [x] Mac Ring8 FreeISRU/SwiGLU refinement and independent merge verification:
  all4cells qualify; max reconstruction error1.16e-14.
- [x] Prepare paired fixed-weight statistics and validate common-path/bank
  resampling plus exact native SW2replay. Confirmation time profiles stay hidden.
- [x] Independently verify all four Product16 models, both fields. Stage-minus-
  shared KL is negative with adjusted intervals excluding zero; reject the
  proposed density explanation of the earlier large-bank SW2 sharing lead.
- [x] Diagnose Product16 component widths, centroids and pairwise dependence;
  calibrate the finite-sample SW2 floor using an exact target-law endpoint map.

## Current work

- [x] Finish six CUDAadaptive cases; queue3584295, source405a201. Keep all
  sources, selected resolutions, previous estimates and unresolved flags.
- [x] Independently reconstruct all eight adaptive merges and every source
  gate. Do not rank an unresolved density contrast.
- [x] Run four paired shared/stage density contrasts, their MCuncertainty
  and resolution sensitivity. Compare SW2on these same prebound inputs.
- [x] Implement the registered mean-update-preserving stage release. Verify
  its full-model/native-continuation identities before the two matched20k
  training controls. This tests sharing versus optimizer coordinates.
  - [x] Add only the stage-coordinate adapter, grid and native evaluator wrapper.
  - [x] Check zero-contrast function/input/common-gradient pairing, arbitrary
    matrix reconstruction, real optimizer mean invariance and clipping factors.
  - [x] Pass native8-to16 continuation and a fresh documented CPU witness.
  - [x] Freeze/publish, run two20k CPU trajectories with the existing monitor,
    and evaluate on the already bound common eight-bank arrays.
- [x] Recompute all ten models, finish native same-function optimizer probes,
  inspect both scientific figures and record cost/paired-bank statistics.
- [x] Assemble the independently verified coordinate-control evidence,
  updated snapshot and inspected figures for publication.
- [ ] Measure new-checkpoint solver sensitivity and joint density/widths on
  the existing sources. Reuse the sixteen completed reference cells; no
  response search, new training or image promotion is authorized by these scores.
  - [x] Extend the existing portable loader and density entry checks to the
    two registered contrast checkpoints; bind the model-builder source hashes.
  - [x] Copy the existing P2source arrays byte-for-byte and generate only the
    two new native CPU anchors. Preserve all earlier portable/evidence roots.
  - [x] Verify derivatives, native endpoints and real CLI entries; obtain a
    fresh documented witness before the bounded observation queue.
  - [x] Use Mac CPUwhile145physical4-7are occupied; benchmark the new cells
    and freeze source before monitored1024/2048 observations.
  - [ ] Independently collect all32new formal rows and refine unresolved
    sources under the original unchanged gates, with no point removal.
  - [ ] Run outer128/256/512 and inner4/8/16checks on the original sharing
    evaluation arrays, independently recompute scores and compare the same banks.

The Product16 oracle-window branch is rejected by its signed density result;
do not select a favorable time bin. Keep confirmation time profiles unopened.

## Current execution

Full producer queues35441and3447050are terminal, with all workers exit0.
Mac adaptive queue36629and workers36631/36632are terminal, exit0.
CUDAadaptive queue3584295is terminal, with all six workers exit0; all
sixteen combined model/field cells qualify independently within4.98e-14.
The full paired statistics and rendered figures are complete. Our CUDA
contexts are released; only idle GPUs4-7were used.
Initial workers3584335/3584336/3584337/3584338; Product16ordinary controls
follow in the same monitored queue. Re-poll live handles after timeouts.

The completed405a201adaptive engine and3a7d5feformal producer remain frozen.
New statistics are separate files; no training or image experiment is added.
Timing-only small-subset probes show that future tiny Ring8refinements should
use Mac CPU; Product16projected subsets still favor GPU. Do not change a
running producer based on an unverified estimate.

Canonical evidence: docs/research-assessments/2026-11-16-ode-mechanism/.
Raw root: /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport.
Current registration: density-adaptive-followup.md; execution: density-adaptive-execution.md.
Previous plan: docs/history/2026-11-16-density-pre-adaptive-task-plan.md.

Previous goal turn completed all-source density qualification and rejected
the proposed Product16 explanation. Current coordinate control is complete:
queue40445and workers40447/40448are terminal, both exit0; watcher40446has
completed all new evaluations. New-versus-shared pooled SW2isRing8.104/.091,
Product16.072/.064. Ring8paired support is5/8for shared and unresolved;
Product16is8/8for shared. Same-function native AdamWprobes establish
coordinate effects, but do not replay historical optimizer momentum.
Goal remains active. New-control solver/density measurements remain pending.

Current continuation is progress: coordinate-control results were published
at4f359d1. Fresh145inventory finds4-7occupied (13--15GiBeach); noGPUis
allocated. Implement the bounded observation extension next; keep density_ops.py,
native loss/sampler and all finalized report files unchanged.

Author preparation is complete: two new checkpoints replay the full native
field and bank0exactly; two new and all eight original model entries pass.
Eight full-native-map logdet checks pass within3.68e-8at the finest stencil.
The fresh documented witness passes four commands, all exits0. Independent
native-law/NumPyentry audits pass, max1.78e-14; eight bounded numerical scores
reproduce within2.78e-17. Sequential256-source benchmarks pass independently.
At frozenfd7f09b, densityqueue43084(workers43093/43094) and numericalqueue43096
(initial workers43098/43099) are live with advancing integration and actual
CPUutilization. Four numerical tasks use all32768sources; two new density
cases retain all1024sources each, two fields and1024/2048steps.
Protocol: coordinate-control/observation-protocol.md. No current blocker.

Execution note: a publication-only Python helper had an unmatched brace and
failed before executing. Corrected it once; no observation job was restarted.
