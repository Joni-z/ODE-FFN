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
- [ ] Implement the registered mean-update-preserving stage release. Verify
  its full-model/native-continuation identities before the two matched20k
  training controls. This tests sharing versus optimizer coordinates.
  - [ ] Add only the stage-coordinate adapter, grid and native evaluator wrapper.
  - [ ] Check zero-contrast function/input/common-gradient pairing, arbitrary
    matrix reconstruction, real optimizer mean invariance and clipping factors.
  - [ ] Pass native8-to16 continuation and a fresh documented CPU witness.
  - [ ] Freeze/publish, run two20k CPU trajectories with the existing monitor,
    and evaluate on the already bound common eight-bank arrays.

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

The running405a201adaptive engine and3a7d5feformal producer remain frozen.
New statistics are separate files; no training or image experiment is added.
Timing-only small-subset probes show that future tiny Ring8refinements should
use Mac CPU; Product16projected subsets still favor GPU. Do not change a
running producer based on an unverified estimate.

Canonical evidence: docs/research-assessments/2026-11-16-ode-mechanism/.
Raw root: /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport.
Current registration: density-adaptive-followup.md; execution: density-adaptive-execution.md.
Previous plan: docs/history/2026-11-16-density-pre-adaptive-task-plan.md.

Previous goal turn: progress, completing all-source density qualification and
rejecting the proposed Product16 explanation. Current turn implements the
registered coordinate control and tests it before the two bounded trainings.
Goal remains active. The density explanation is rejected; the registered
mean-update control remains to isolate sharing from optimizer coordinates.
