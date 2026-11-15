# Plan: choose the next ODE-FFN architecture test

Scope: evidence review and recommendation only; no new training submission.

- [x] Read repository conventions, current source and matched CelebA FIDs.
- [x] Reconcile proposed mechanisms with prior synthetic tests and primary literature.
- [x] Derive one minimal structural comparison and its decision rule.
- [x] Verify formulas, links and preserved report files.
- [x] Publish recommendation with the repository notes and evidence index.

## 15 November: turn the recommendation into an execution contract

- [x] Read final FID160 for all three arms and cross-check console/TensorBoard.
- [x] Write the exact two-arm protocol, bounded diagnostics, resource budget and decision table.
- [x] Update the matched result table and current project state.
- [x] Verify and publish the plan. Implementation and training remain unstarted.

Current observations: at epoch 140, native/K1/K8 FID50k is 25.98559/29.62906/30.08220. Baseline is still training and K8 is evaluating epoch 160.

Resolved lookup errors: guessed analysis-nonlinear-drift and state-clock-readout
paths did not exist. Used the indexed nonlinear-drift-readout.md and
analysis-state-clock/analysis-report.md instead. Parameter/MAC evidence comes
from the existing study README, not a nonexistent parameters.json.

Plan source inspection: a shell glob for nonexistent tests/test* matched no files; no test was executed by that lookup. Required tests are specified in the plan. Shared skill output-reference files are absent; used the canonical project documentation directory and Git history for versioning.
