# Design, timing and audit scope

## Design and units

Seven fixed Product16 cells use training seed 0, width 64/depth 3, 5000 updates, batch 512, cached-target/independent-Gaussian pairs, uniform time, AdamW3e-4, zero weight decay and clip1. Only the declared FFN setting varies. Common surrounding tensors, Fourier buffers and first ODE encoder tensors are preserved; different FFNs do not have identical hidden initial functions or head parameter responses. The common zero head must not be treated as NTK pairing.

Outcomes are fixed training-trajectory comparisons. Checkpoints, five time panels, P matrices and sampling projections are repeated measurements, not independent training repetitions. No p-values or training-population confidence intervals are claimed. The original/SwiGLU compatibility reruns reproduce their actual prior states and are not new independent evidence.

## Prototype and pipeline checks

Four tests verify exact VJP values, mixed input/parameter derivatives, local symmetry, the omitted-clock identity, ordinary tied-ridge gradients, whole-model parameter finite differences and preserved initialization. A single-vector shape bug was found before profiling and corrected only in the new wrapper; no old helper was changed. Worktree and frozen-source tests pass.

The pre-outcome resource screen passes all three fixed conditions. Actual isolated group timings use a deterministic batch512 and nonzero receiving head, median over five groups, with no optimizer update. The complete timings and source hashes are in runtime/profile.json. Same dense MACs do not imply same pointwise or wall-time cost.

All seven three-update pipeline cells complete. Original/SwiGLU128/SwiGLU192 are separately executed through the actual old training function at three updates: six 0/3-step model, optimizer, buffer, data RNG, Torch RNG and input-digest states match bitwise. Both local models move every one of their15 encoder parameter tensors and capture all three local-FFN probe visits. READY-cold.json binds those checks before formal training.

## Completed cold-run verification

The supervisor and all seven child cells are terminal COMPLETE; no recorded child PID remains. Two one-thread workers and ten-second process/memory telemetry were used. Lenovo remained offline. No GPU was allocated or remote job claimed.

The audit verifies all 163 frozen source files and actual input/checkpoint/bank bindings, regenerates the entire training hash/RNG stream, restores28 checkpoints, and exactly matches12 historical native-control model/optimizer/buffer/RNG states. All7 final fields,35 prediction/Jacobian panels and7 common-source generated sample arrays replay bitwise. SW2 is independently recalculated in NumPy float64 within the declared numerical tolerance.

All48 original/local P spectrum records are reconstructed from checkpoint matrices, including singular values and complex spectra. All24 local FFN states across the two local arms, three blocks and four checkpoints are compared with a separately autograd-differentiated original encoder, including complete input and parameter loss gradients. Maximum local value/gradient discrepancies are3.55e-14/1.42e-13. Two full trained-model finite directions, one P and one clock projection, have maximum relative error1.20e-12; perturbed weights are restored.

The separate plot/screen script recomputes every decision inequality as normalized ratios directly from all seven raw results and verifies agreement with the audited table and decision. Both quality routes and derivative/sampling guards fail. This is a prospective design-selection convention, not statistical significance or automatic scientific proof.

## Runtime and artifacts

```text
/Users/mr.z/ode-workspace-prep/architecture-search/local_adjoint_20261026
  source/                 frozen prototype (37 files)
  profile.json            actual seven-setting resource/timing measurements
  source-v2/              frozen formal source (163 files)
  preflight/              all seven three-update cells
  old-native-preflight/   actual old-function compatibility execution
  READY-cold.json         verified preflight receipt
  output/                 completed seven-cell formal training and monitoring
  analysis/               full replay, trajectories and independent screen
```

The audit's scalar-conversion warning concerns reporting a detached numerical error after differentiation and does not change any model, gradient or check. A status-inspection command used system Python without psutil; the existing venv supplied it, and actual terminal processes were verified. A preflight observation handle was missing after terminal completion, so the completed directory/states were inspected rather than restarted.

The endpoint plot's initially crowded scientific-notation tick labels were replaced with ordinary numeric ticks for Jacobian/SW2. No data or evaluation changed. Prior coordinate/affine/depth reports and submitted Report1 remain untouched.
