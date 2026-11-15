# ODE mechanism: current evidence and next action

Goal: identify the useful architectural constraints, with seed0and unchanged
native trainers/samplers/metrics. Reports1/2are final; all96protected files
remain unchanged. Mac mini is available;145physical0-3are forbidden and4-7
may be used only when empty. Never stop or share another user's task.

## Completed evidence

- Entrance-frozen exp(PD(h)) is not a hidden-only semigroup. Restart defect
  is much larger than Taylor8truncation on learned states. True state flow
  y'=P ISRU(y) has derivativeP diag(D^3), notPD.
- Fixed/free Taylor coefficients have similar generation scores; fixed/free
  midpoint stages also match at trained work. Coefficients are active.
- State feedback improves within-family Ring8sampling, but Product16ordinary
  controls remain stronger. Fixed midpoint has order2; free stages approach1.
- Shared/released stage matrices, at identical initial function and dense
  work: confirmationSW2Ring8 .091/.111, Product16 .064/.073. Sharing wins
  all8banks each. Released stages add73,728params; their actions are active.
  Product16field fitting improves with release but sampling gets worse.
- Shared P still has state-dependentJ_f=P diag(q'); temporal stationarity
  leaves spatial anisotropy and nonlinear state adaptation free. Released
  fields have nonzero Lie brackets and ordering sensitivity. Both are ODEs.
- Same-host CPUcost/parameter checks pass. Free ISRU has the same parameter
  count as shared flow but about one quarter of its matrix work and wins
  Product16. No practical image-promotion gate is met.
- All25resolution/C8cases and78scores independently recompute. C8projection
  removes/reverses state flow's Ring8lead over ordinary controls on one
  paired stratified bank. The released-stage model was absent from that cohort.
- Full P0GPUrefinement on idle145GPU4: all16pooled/128bank scores independently
  recompute within2.78e-17. Native CUDA/CPUscore difference<=3.44e-8. Outer
  Heun128->512leaves the sharing gap. Inner4->16retains pooled sharing lead,
  but Ring8bank support falls8/8->5/8; Product16stays8/8. Sampling/metric
  compute summed53.824seconds. Worker2734223is terminal.
- Oracle variational transport is accurate locally but fails as a full-model
  explanation. Paired endpoint displacement is not SW2. Preserve that rejection.

## Canonical artifacts and raw inputs

Study:docs/research-assessments/2026-11-16-ode-mechanism/
Primary:primary-findings.md; sharing-results/analysis-report.md;
sharing-results/full-resolution/analysis.md; numerical-findings.md.
Raw root:/Users/mr.z/ode-workspace-prep/coefficient-20261116
All eight comparison folders/configs/checkpoint bindings:sharing-anchors.json.
Data:/Users/mr.z/ode-workspace-prep/data/synthetic, v1andproduct16_v1.
Warm CPU Python:/Users/mr.z/ode-workspace-prep/.venv-test/bin/python.
Pure NumPy/SciPy analysis:systempython3(noTorch).
145warm Python:/data2/zz5070/miniconda3/envs/py312/bin/python,Torch2.8.0+cu128.
145source7735abe:/data2/zz5070/ODE_FFN/mechanism-sharing-20261116/repo
145portablefour-focal bundle:../portable; full outputs:../full-refinement.
Current ordinary references stay in their original CPUresponse-screen folders;
P1must export them too. Finalized report directories and old runtimes stay fixed.

## P1next

Follow sharing-symmetry-followup.md. Four cells on each task: shared P,
released P, Free ISRU and native SwiGLU; four new4096-point banks, stream2026112200.
UseC8rotations forRing8and latent sign flips onaxes0,7,15forProduct16. Verify
Qorientation from the native law and exact oracle equivariance. Compare
original/projected full fields and sampling on identical group-stratified
inputs, preserving dependent-orbit accounting. No claim of intrinsic hidden
ODEequivariance. If projection removes the sharing lead, it supports this
error-component explanation; if not, symmetry alone is insufficient.

Generic sharing_dispatch.py now selects only an idle145physical4-7UUID;
exclusive_gpu rechecks occupancy during execution. Keep new source frozen
and get a fresh documented CUDAwitness before full deployment. No new
packages needed. The full goal remains active; P1is prepared, not launched.

P1CPU entry receipts now explicitly label witness=true,16base points,2banks;
these values are engineering checks, never evidence for a model ranking.
Independent verification also reconstructs both selected groups from the
native data specification. Premature remote extraction failed while SCP
was still active; its partial directory will be retained separately.

P1CUDAreadiness passed at5e78e6a: max field5.67e-15, native64sample1.60e-14,
shortcut5.90e-13; all8shortcuts qualified. Queue2996899uses permitted idle
GPU4/5;6/7became occupied and were not used. No training added.
