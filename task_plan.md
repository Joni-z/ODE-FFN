# Task: publish the completed wide K8/K1 comparison

Status request14November2026; no new experiment selected.

- [x] Bind final FIDs to original logs/TensorBoard/checkpoints and source;
      collect all five evaluation points and verify terminal job state.
- [x] Summarize quality and progress with a FID/training-loss figure.
- [x] Update current status, notes and evidence; push GitHub and sync Torch.

Both jobs completed0:0. Reported final FID: wideK8 38.1357422, wideK1
37.9248619, fixed native35.9178543. Validation complete: both jobs0:0, all final/checkpoint/FID checks pass.

Verified model/twoEMA finite, all63500steps, five full checkpoints,101P
snapshots per arm and all10console/TensorBoardFID points. Collector hash
helper adapted for Python3.10 (file_digest unavailable). No training changes.
